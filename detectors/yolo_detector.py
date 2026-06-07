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
    # Objects that may be left in the alley (bags, cases on the ground)
    24: "backpack",
    26: "handbag",
    28: "suitcase",
}


class YoloDetector(BaseDetector):
    """
    Wraps YOLOv8n with a background inference thread.

    process_frame() is non-blocking: it hands the latest frame to the
    background thread and immediately returns the previous inference result
    drawn onto the current frame. This keeps end-to-end latency low on the
    main vision loop while YOLO runs asynchronously.
    """

    # After this many consecutive inference errors, YOLO is disabled for safety.
    _MAX_CONSECUTIVE_ERRORS = 5

    def __init__(self) -> None:
        self._model = None
        self._class_names: dict = {}
        self._filter_classes: Optional[list] = None
        self._conf = config_manager.get("detectors.yolo.confidence", 0.50)
        self._danger_ratio = config_manager.get(
            "detectors.yolo.danger_zone_ratio", 0.12
        )
        self._device = config_manager.get("detectors.yolo.device", "cpu")
        # Path-ahead region of interest. An obstacle only triggers a STOP if its
        # centre falls inside this central corridor — so a person/bicycle off on
        # the adjacent road is detected (orange) but does NOT stop the machine.
        # Fractions of frame width/height. roi_enabled=False = whole frame.
        self._roi_enabled = config_manager.get("detectors.yolo.roi_enabled", True)
        self._roi_half_w = config_manager.get("detectors.yolo.roi_half_width", 0.28)
        self._roi_top = config_manager.get("detectors.yolo.roi_top", 0.40)
        self._roi_bottom = config_manager.get("detectors.yolo.roi_bottom", 1.0)
        # Safety-critical classes (people, animals): STOP for these in the path
        # even when small/far — you must never drive toward a person. They use a
        # much smaller size threshold than vehicles/objects.
        self._priority_classes = set(
            config_manager.get(
                "detectors.yolo.priority_classes",
                ["person", "cat", "dog", "horse", "sheep", "cow"],
            )
        )
        self._priority_min_ratio = config_manager.get(
            "detectors.yolo.priority_min_ratio", 0.015
        )
        # Performance: run inference at a reduced image size and cap the rate so
        # the background thread does not saturate the CPU and starve the main
        # video loop. 6 FPS obstacle updates are ample for a slow field robot.
        self._imgsz = int(config_manager.get("detectors.yolo.imgsz", 416))
        max_fps = float(config_manager.get("detectors.yolo.max_fps", 6.0))
        self._min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self._faulted = False  # True → inference disabled, return empty results
        self._consecutive_errors = 0

        try:
            from ultralytics import YOLO
        except ImportError:
            log.error(
                "ultralytics not installed — YOLO obstacle detection disabled. "
                "Install with: pip install ultralytics"
            )
            self._faulted = True
            self._start_dummy_thread()
            return

        model_path = config_manager.get("detectors.yolo.model", "yolov8n.pt")
        log.info("Loading YOLO model: %s on device=%s", model_path, self._device)
        try:
            self._model = YOLO(model_path)
        except Exception as exc:
            log.error(
                "Failed to load YOLO model '%s': %s — obstacle detection disabled.",
                model_path,
                exc,
            )
            self._faulted = True
            self._start_dummy_thread()
            return

        # Use the model's own class names when present (both the pretrained COCO
        # model and any custom-trained model expose .names). We deliberately do
        # NOT filter to a fixed subset: detecting every class the model knows
        # means more objects in the path can stop the machine. Safety-critical
        # vs. ordinary classes are handled later via priority_classes.
        if hasattr(self._model, "names") and self._model.names:
            self._class_names = self._model.names
            self._filter_classes = None
            log.info(
                "YOLO model loaded with %d classes.", len(self._class_names)
            )
        else:
            # Fallback for an unusual model with no embedded names.
            self._class_names = _OBSTACLE_CLASSES
            self._filter_classes = list(_OBSTACLE_CLASSES.keys())
            log.info(
                "Model has no class names — using %d-class obstacle fallback.",
                len(self._filter_classes),
            )

        # Warm-up pass so the first real frame isn't slow
        try:
            self._model.predict(
                np.zeros((64, 64, 3), dtype=np.uint8),
                verbose=False,
                device=self._device,
            )
        except Exception as exc:
            log.warning("YOLO warm-up failed (non-fatal): %s", exc)

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

    def _start_dummy_thread(self) -> None:
        """Start a no-op thread so shutdown() is always safe to call."""
        self._lock = threading.Lock()
        self._pending_frame = None
        self._latest_obstacles = []
        self._new_frame_event = threading.Event()
        self._running = False
        self._thread = threading.Thread(target=lambda: None, daemon=True)
        self._thread.start()

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
                t_inf = time.time()
                obstacles = self._run_inference(frame)
                with self._lock:
                    self._latest_obstacles = obstacles
                self._consecutive_errors = 0  # reset on success
                # Throttle: yield the CPU so the main video loop stays smooth.
                spare = self._min_interval - (time.time() - t_inf)
                if spare > 0:
                    time.sleep(spare)
            except Exception as exc:
                self._consecutive_errors += 1
                log.error(
                    "YOLO inference error (%d/%d): %s",
                    self._consecutive_errors,
                    self._MAX_CONSECUTIVE_ERRORS,
                    exc,
                )
                if self._consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS:
                    log.critical(
                        "YOLO disabled after %d consecutive errors — "
                        "continuing without obstacle detection.",
                        self._MAX_CONSECUTIVE_ERRORS,
                    )
                    self._faulted = True
                    self._running = False

    def _in_roi(self, x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> bool:
        """True if the bbox is inside the path-ahead corridor (centre in the
        central x-band and reaching into the lower band). If ROI is disabled,
        everything counts as in-path (whole-frame behaviour)."""
        if not self._roi_enabled:
            return True
        cx = (x1 + x2) / 2.0
        left = (0.5 - self._roi_half_w) * w
        right = (0.5 + self._roi_half_w) * w
        in_x = left <= cx <= right
        in_y = y2 >= self._roi_top * h and y1 <= self._roi_bottom * h
        return in_x and in_y

    def _run_inference(self, frame: np.ndarray) -> List[ObstacleDetection]:
        if self._model is None:
            return []
        h, w = frame.shape[:2]
        frame_area = max(h * w, 1)

        results = self._model.predict(
            frame,
            conf=self._conf,
            classes=self._filter_classes,
            verbose=False,
            device=self._device,
            imgsz=self._imgsz,
        )

        obstacles: List[ObstacleDetection] = []
        for res in results:
            if res.boxes is None:
                continue
            for box in res.boxes:  # type: ignore[union-attr,attr-defined]
                cls_id = int(box.cls[0])
                if cls_id not in self._class_names:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                label = self._class_names[cls_id]
                area_ratio = max((x2 - x1) * (y2 - y1), 0) / frame_area
                in_path = self._in_roi(x1, y1, x2, y2, w, h)
                # People/animals: stop even when small/far. Others: only close.
                min_ratio = (
                    self._priority_min_ratio
                    if label in self._priority_classes
                    else self._danger_ratio
                )
                is_critical = in_path and area_ratio >= min_ratio
                obstacles.append(
                    ObstacleDetection(
                        label=label,
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
        if self._faulted:
            # Return empty result — obstacle detection unavailable but don't crash
            return frame, DetectionResult()

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

    def _draw_obstacles(
        self, frame: np.ndarray, obstacles: List[ObstacleDetection]
    ) -> None:
        # Draw the path-ahead STOP zone (only obstacles inside it can stop us).
        if self._roi_enabled:
            h, w = frame.shape[:2]
            lx = int((0.5 - self._roi_half_w) * w)
            rx = int((0.5 + self._roi_half_w) * w)
            ty = int(self._roi_top * h)
            by = int(self._roi_bottom * h) - 1
            cv2.rectangle(frame, (lx, ty), (rx, by), (180, 180, 60), 1)
            cv2.putText(
                frame, "STOP ZONE", (lx + 4, ty + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 60), 1,
            )

        for obs in obstacles:
            x1, y1, x2, y2 = obs.bbox
            color = (0, 0, 255) if obs.is_critical else (0, 140, 255)
            thickness = 3 if obs.is_critical else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            label = f"{obs.label} {obs.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            if obs.is_critical:
                cv2.putText(
                    frame,
                    "STOP",
                    (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_faulted(self) -> bool:
        """True when YOLO has self-disabled due to load or inference failures."""
        return self._faulted

    def shutdown(self) -> None:
        if not self._running:
            return
        self._running = False
        self._new_frame_event.set()
        self._thread.join(timeout=3.0)
        log.info("YoloDetector shut down.")

    def __del__(self) -> None:
        self.shutdown()
