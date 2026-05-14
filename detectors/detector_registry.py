"""
detectors/detector_registry.py
==============================
Dynamic registry for computer vision detectors.
"""
from typing import Dict
import numpy as np

from detectors.base_detector import BaseDetector, DetectionResult
from utils.logger import get_logger
from config.config_manager import config_manager

log = get_logger(__name__)

class DetectorRegistry:
    def __init__(self):
        self._detectors: Dict[str, BaseDetector] = {}
        
    def register(self, name: str, detector_instance: BaseDetector) -> None:
        self._detectors[name] = detector_instance
        log.info("Registered detector: %s", name)
        
    def process_all(self, frame: np.ndarray) -> tuple[np.ndarray, DetectionResult]:
        annotated_frame = frame.copy()
        combined_result = DetectionResult()
        
        for name, detector in self._detectors.items():
            annotated_frame, result = detector.process_frame(annotated_frame)
            combined_result.targets.extend(result.targets)
                
        return annotated_frame, combined_result

    def is_empty(self) -> bool:
        return len(self._detectors) == 0

def build_detectors_from_config() -> DetectorRegistry:
    from detectors.aruco_detector import ArucoDetector
    from detectors.barcode_detector import BarcodeDetector
    from detectors.feature_detector import FeatureDetector
    
    registry = DetectorRegistry()
    
    if config_manager.get("detectors.aruco.enabled", False):
        registry.register("aruco", ArucoDetector())
        
    if config_manager.get("detectors.barcode.enabled", False):
        registry.register("barcode", BarcodeDetector())
        
    if config_manager.get("detectors.feature.enabled", False):
        registry.register("feature", FeatureDetector("target.jpg"))
        
    if registry.is_empty():
        log.warning("No detectors enabled in configuration!")
        
    return registry
