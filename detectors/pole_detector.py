"""
detectors/pole_detector.py
==========================
CV detector for finding red 1-meter poles at row ends.
"""

from typing import Tuple
import cv2
import numpy as np
import time

from config.config_manager import config_manager
from .base_detector import BaseDetector, DetectionResult, DetectionTarget, MarkerType
from utils.logger import get_logger

log = get_logger(__name__)


class PoleDetector(BaseDetector):
    def __init__(self) -> None:
        # Load HSV thresholds from config or use defaults for red
        # Red can wrap around the hue cylinder in OpenCV (0-10 and 160-180)
        self.hsv_lower1 = np.array(
            config_manager.get("detectors.pole.hsv_lower1", [0, 100, 100])
        )
        self.hsv_upper1 = np.array(
            config_manager.get("detectors.pole.hsv_upper1", [10, 255, 255])
        )

        self.hsv_lower2 = np.array(
            config_manager.get("detectors.pole.hsv_lower2", [160, 100, 100])
        )
        self.hsv_upper2 = np.array(
            config_manager.get("detectors.pole.hsv_upper2", [180, 255, 255])
        )

        # Camera info for distance estimation
        cam_mat = config_manager.get("detectors.aruco.camera_matrix")
        if cam_mat:
            self.camera_matrix = np.array(cam_mat, dtype=np.float32)
            self.focal_length = self.camera_matrix[0, 0]  # fx
        else:
            self.focal_length = 800.0  # Fallback focal length

        # Known real-world width of the pole in meters (e.g., 0.1m / 10cm)
        self.pole_real_width_m = config_manager.get("detectors.pole.width_m", 0.1)
        self.turn_trigger_distance = config_manager.get(
            "detectors.pole.turn_trigger_distance_m", 1.5
        )

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, DetectionResult]:
        start_time = time.time()
        result = DetectionResult()

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create masks for red color (two ranges)
        mask1 = cv2.inRange(hsv, self.hsv_lower1, self.hsv_upper1)
        mask2 = cv2.inRange(hsv, self.hsv_lower2, self.hsv_upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 500:  # Minimum pixel area to be considered a pole
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Check aspect ratio (poles are tall and thin)
            aspect_ratio = h / float(w)
            if (
                aspect_ratio < 2.0
            ):  # If it's not at least twice as tall as it is wide, it's not a pole
                continue

            # Distance estimation using pinhole camera model
            # Distance = (Real Width * Focal Length) / Pixel Width
            distance_m = (self.pole_real_width_m * self.focal_length) / w

            center_x = x + w / 2.0
            center_y = y + h / 2.0

            # Calculate confidence based on area
            confidence = min(1.0, area / 5000.0)

            target = DetectionTarget(
                id=f"RED_POLE_{idx}",
                distance_m=float(distance_m),
                center_x=float(center_x),
                center_y=float(center_y),
                confidence=float(confidence),
                latency_ms=(time.time() - start_time) * 1000,
                priority=1,
                is_turn_trigger=distance_m <= self.turn_trigger_distance,
                marker_type=MarkerType.NORMAL,
            )

            result.targets.append(target)

            # Draw bounding box and info
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"Pole: {distance_m:.2f}m",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        result.frame_latency_ms = (time.time() - start_time) * 1000
        return frame, result
