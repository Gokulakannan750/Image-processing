"""
detectors/yolo_detector.py
==========================
YOLOv8-based obstacle detector for real-time safety monitoring.

Runs inference in a background thread so the ArUco/barcode pipeline
is never blocked. The main thread always gets the most recently
completed inference result with zero added latency to the critical path.
"""
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config.config_manager import config_manager
from .base_detector import BaseDetector, DetectionResult, ObstacleDetection
from utils.logger import get_logger

log = get_logger(__name__)

# COCO class IDs that represent physical obstacles on a farm
_OBSTACLE_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
}


class YoloDetector(BaseDetector):
    """
    Wraps YOLOv8n with a background inference thread.

    process_frame() is non-blocking: it hands the latest frame to the
    background thread and immediately returns the previous inference result
    drawn onto the current frame. This keeps end-to-end latency low on the
    main vision loop while YOLO runs asynchronously.
    """

    def __init__(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for YoloDetector. "
                "Install it with: pip install ultralytics"
            )

        model_path = config_manager.get("detectors.yolo.model", "yolov8n.pt")
        self._conf = config_manager.get("detectors.yolo.confidence", 0.50)
        self._danger_ratio = config_manager.get("detectors.yolo.danger_zone_ratio", 0.12)
        self._device = config_manager.get("detectors.yolo.device", "cpu")

        log.info("Loading YOLO model: %s on device=%s", model_path, self._device)
        self._model = YOLO(model_path)

        # If the model has its own class names (custom trained), use them.
        # Otherwise fall back to the hardcoded COCO obstacle subset.
        if hasattr(self._model, "names") and self._model.names:
            self._class_names = self._model.names
            self._filter_classes = None
            log.info("Custom model loaded with %d classes: %s",
                     len(self._class_names), list(self._class_names.values()))
        else:
            self._class_names = _OBSTACLE_CLASSES
            self._filter_classes = list(_OBSTACLE_CLASSES.keys())
            log.info("Using pretrained COCO model — filtering to %d obstacle classes.",
                     len(self._filter_classes))

        # Warm-up pass so the first real frame isn't slow
        self._model.predict(
            np.zeros((64, 64, 3), dtype=np.uint8),
            verbose=False,
            device=self._device,
        )

        self._lock = threading.Lock()
        self._pending_frame: Optional[np.ndarray] = None
        self._latest_obstacles: List[ObstacleDetection] = []
        self._new_frame_event = threading.Event()
        self._running = True

        self._thread = threading.Thread(
            target=self._inference_loop, name="yolo-inference", daemon=True
        )
        self._thread.start()
        log.info("YoloDetector ready — background inference thread started.")

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _inference_loop(self) -> None:
        while self._running:
            triggered = self._new_frame_event.wait(timeout=0.5)
            if not triggered or not self._running:
                continue
            self._new_frame_event.clear()

            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None

            if frame is None:
                continue

            try:
                obstacles = self._run_inference(frame)
                with self._lock:
                    self._latest_obstacles = obstacles
            except Exception as exc:
                log.error("YOLO inference error: %s", exc)

    def _run_inference(self, frame: np.ndarray) -> List[ObstacleDetection]:
        h, w = frame.shape[:2]
        frame_area = max(h * w, 1)

        results = self._model.predict(
            frame,
            conf=self._conf,
            classes=self._filter_classes,
            verbose=False,
            device=self._device,
        )

        obstacles: List[ObstacleDetection] = []
        for res in results:
            for box in res.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self._class_names:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                bbox_area = max((x2 - x1) * (y2 - y1), 0)
                is_critical = (bbox_area / frame_area) >= self._danger_ratio
                obstacles.append(
                    ObstacleDetection(
                        label=self._class_names[cls_id],
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        class_id=cls_id,
                        is_critical=is_critical,
                    )
                )
        return obstacles

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, DetectionResult]:
        t0 = time.time()

        # Hand the frame to the background thread (non-blocking)
        with self._lock:
            self._pending_frame = frame.copy()
            obstacles = list(self._latest_obstacles)
        self._new_frame_event.set()

        # Draw the most recent results onto the current frame
        self._draw_obstacles(frame, obstacles)

        result = DetectionResult()
        result.obstacles = obstacles
        result.frame_latency_ms = (time.time() - t0) * 1000
        return frame, result

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_obstacles(self, frame: np.ndarray, obstacles: List[ObstacleDetection]) -> None:
        for obs in obstacles:
            x1, y1, x2, y2 = obs.bbox
            color = (0, 0, 255) if obs.is_critical else (0, 140, 255)
            thickness = 3 if obs.is_critical else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            label = f"{obs.label} {obs.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            if obs.is_critical:
                cv2.putText(frame, "STOP", (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if not self._running:
            return
        self._running = False
        self._new_frame_event.set()
        self._thread.join(timeout=3.0)
        log.info("YoloDetector shut down.")

    def __del__(self) -> None:
        self.shutdown()
