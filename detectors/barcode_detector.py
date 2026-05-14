"""
detectors/barcode_detector.py
=============================
Detector for standard 1D/2D barcodes using pyzbar.
"""
from typing import Tuple

import cv2
import numpy as np

from config.config_manager import config_manager
from .base_detector import BaseDetector, DetectionResult, DetectionTarget
from utils.logger import get_logger

log = get_logger(__name__)

try:
    from pyzbar import pyzbar
    _PYZBAR_AVAILABLE = True
except ImportError:
    log.warning("pyzbar not installed. Barcode detection will fail.")
    _PYZBAR_AVAILABLE = False


class BarcodeDetector(BaseDetector):
    def __init__(self) -> None:
        self.min_pixel_area = config_manager.get("detectors.barcode.min_pixel_area", 20000)
        log.info("BarcodeDetector initialised — min trigger area: %d px^2", self.min_pixel_area)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, DetectionResult]:
        result = DetectionResult()

        if not _PYZBAR_AVAILABLE:
            return frame, result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        barcodes = pyzbar.decode(gray)

        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            area = w * h
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type

            text = f"{barcodeData} ({barcodeType}) Area: {area}"
            cv2.putText(
                frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2,
            )

            is_turn = area > self.min_pixel_area
            center_x = float(x + w / 2)
            center_y = float(y + h / 2)

            target = DetectionTarget(
                id=f"Barcode: {barcodeData}",
                distance_m=None, # Cannot easily estimate absolute distance
                center_x=center_x,
                center_y=center_y,
                priority=2,
                is_turn_trigger=is_turn
            )
            result.targets.append(target)

        return frame, result
