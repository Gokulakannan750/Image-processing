"""
detectors/detector_registry.py
==============================
Dynamic registry for computer vision detectors.
"""

from typing import Dict, Tuple
import numpy as np

from .base_detector import BaseDetector, DetectionResult
from utils.logger import get_logger
from config.config_manager import config_manager

log = get_logger(__name__)


class DetectorRegistry:
    def __init__(self):
        self._detectors: Dict[str, BaseDetector] = {}

    def register(self, name: str, detector_instance: BaseDetector) -> None:
        self._detectors[name] = detector_instance
        log.info("Registered detector: %s", name)

    def process_all(self, frame: np.ndarray) -> Tuple[np.ndarray, DetectionResult]:
        annotated_frame = frame.copy()
        combined_result = DetectionResult()

        for name, detector in self._detectors.items():
            annotated_frame, result = detector.process_frame(annotated_frame)
            combined_result.targets.extend(result.targets)
            combined_result.obstacles.extend(result.obstacles)

        return annotated_frame, combined_result

    def is_empty(self) -> bool:
        return len(self._detectors) == 0

    def is_yolo_faulted(self) -> bool:
        """Returns True if the YOLO detector is registered but has self-disabled."""
        yolo = self._detectors.get("yolo")
        return bool(yolo and getattr(yolo, "is_faulted", False))

    def shutdown(self) -> None:
        for name, detector in self._detectors.items():
            if hasattr(detector, "shutdown"):
                detector.shutdown()
                log.info("Shut down detector: %s", name)


def build_detectors_from_config() -> DetectorRegistry:
    from .aruco_detector import ArucoDetector
    from .barcode_detector import BarcodeDetector
    from .feature_detector import FeatureDetector
    from .yolo_detector import YoloDetector
    from .pole_detector import PoleDetector
    from .row_follower import RowFollowerDetector

    registry = DetectorRegistry()

    if config_manager.get("detectors.row_follower.enabled", False):
        registry.register("row_follower", RowFollowerDetector())

    if config_manager.get("detectors.orchard.enabled", False):
        from .orchard_follower import OrchardFollowerDetector

        registry.register("orchard", OrchardFollowerDetector())

    if config_manager.get("detectors.aruco.enabled", False):
        registry.register("aruco", ArucoDetector())

    if config_manager.get("detectors.barcode.enabled", False):
        registry.register("barcode", BarcodeDetector())

    if config_manager.get("detectors.feature.enabled", False):
        registry.register("feature", FeatureDetector("target.jpg"))

    if config_manager.get("detectors.yolo.enabled", False):
        registry.register("yolo", YoloDetector())

    if config_manager.get("detectors.ground_obstacle.enabled", False):
        from .ground_obstacle import GroundObstacleDetector

        registry.register("ground_obstacle", GroundObstacleDetector())

    if config_manager.get("detectors.pole.enabled", True):  # Default to True for new architecture
        registry.register("pole", PoleDetector())

    if registry.is_empty():
        log.warning("No detectors enabled in configuration!")

    return registry
