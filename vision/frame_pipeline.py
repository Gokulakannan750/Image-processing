"""
vision/frame_pipeline.py
========================
Coordinates preprocessing, execution of the detector registry,
performance monitoring, and postprocessing.
"""

from typing import Tuple
import cv2
import numpy as np

from detectors.detector_registry import DetectorRegistry
from detectors.base_detector import DetectionResult
from vision.performance_monitor import PerformanceMonitor
from vision.lighting_normalization import LightingNormalizer
from vision.pose_stability_analyzer import PoseStabilityAnalyzer
from testing.environment_modes import EnvironmentSimulator, TestMode
from config.config_manager import config_manager
from utils.logger import get_logger

log = get_logger(__name__)


class FramePipeline:
    def __init__(self, registry: DetectorRegistry):
        self.registry = registry
        self.perf_monitor = PerformanceMonitor()
        self.normalizer = LightingNormalizer()
        self.stability_analyzer = PoseStabilityAnalyzer()

        # Determine test mode from config
        mode_str = config_manager.get("testing.active_mode", "NORMAL")
        try:
            self.test_mode = TestMode[mode_str]
        except KeyError:
            log.error(f"Invalid test mode: {mode_str}. Reverting to NORMAL.")
            self.test_mode = TestMode.NORMAL

        # Detector stride: run the (heavy) detector stack on 1 of every N frames
        # and re-use the last result on the others. This keeps the displayed
        # video smooth even when detection is slow. 1 = run every frame.
        self._stride = max(1, int(config_manager.get("processing.detector_stride", 1)))
        self._frame_count = 0
        self._last_result = DetectionResult()
        # Cap processing resolution so per-frame cost is bounded no matter how
        # large the camera/video frames are. 0 = use frames as-is.
        self._max_width = int(config_manager.get("processing.max_width", 0))

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, DetectionResult]:
        self.perf_monitor.tick()

        if frame is None or frame.size == 0:
            log.warning("Received empty frame in pipeline.")
            return frame, DetectionResult()

        # 0. Cap resolution: downscale large frames so cost stays bounded.
        if self._max_width and frame.shape[1] > self._max_width:
            s = self._max_width / float(frame.shape[1])
            frame = cv2.resize(frame, (self._max_width, int(frame.shape[0] * s)))

        # 1. Simulate environmental conditions (if enabled)
        processed_frame = EnvironmentSimulator.apply_mode(frame.copy(), self.test_mode)

        # Detector stride: on skipped frames, show the fresh frame (smooth video)
        # but re-use the previous detection result for navigation/safety.
        self._frame_count += 1
        run_detectors = (self._frame_count % self._stride) == 0 or self._stride == 1

        if not run_detectors:
            annotated_frame = processed_frame
            result = self._last_result
        else:
            # 2. Apply Lighting Normalization
            processed_frame = self.normalizer.process(processed_frame)

            # 3. Execute Detectors
            if not self.registry.is_empty():
                annotated_frame, result = self.registry.process_all(processed_frame)
            else:
                annotated_frame = processed_frame
                result = DetectionResult()
            self._last_result = result

        # 4. Stability Analysis
        stability_score = self.stability_analyzer.update(result.targets)
        result.tracking_stability_score = stability_score

        # 5. UI Overlays
        self.perf_monitor.draw_overlay(annotated_frame)

        if config_manager.get("testing.draw_mode_overlay", True):
            EnvironmentSimulator.draw_overlay(annotated_frame, self.test_mode)

        if config_manager.get("validation.enable_stability_overlay", True):
            color = (0, 255, 0) if stability_score > 0.8 else (0, 165, 255)
            cv2.putText(
                annotated_frame,
                f"STABILITY: {stability_score:.1%}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        return annotated_frame, result
