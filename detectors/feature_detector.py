"""
detectors/feature_detector.py
=============================
Custom image matching using ORB features.
"""
import os
from typing import Tuple

import cv2
import numpy as np

from config.config_manager import config_manager
from .base_detector import BaseDetector, DetectionResult, DetectionTarget
from utils.logger import get_logger
from utils.exceptions import DetectorError

log = get_logger(__name__)


class FeatureDetector(BaseDetector):
    def __init__(self, target_image_path: str) -> None:
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.min_matches = config_manager.get("detectors.feature.min_matches", 15)
        self.match_dist_thresh = config_manager.get("detectors.feature.match_distance_threshold", 50)

        if not os.path.exists(target_image_path):
            raise DetectorError(f"Target image {target_image_path} not found.")

        self.target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
        self.target_kp, self.target_des = self.orb.detectAndCompute(self.target_image, None)
        log.info("FeatureDetector initialised with target: %s", target_image_path)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, DetectionResult]:
        result = DetectionResult()

        if self.target_des is None:
            return frame, result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        if des is not None:
            matches = self.bf.match(self.target_des, des)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < self.match_dist_thresh]

            cv2.putText(
                frame, f"Good Matches: {len(good_matches)}/{self.min_matches}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )

            is_turn = len(good_matches) >= self.min_matches
            
            # Estimate center based on matched keypoints
            center_x, center_y = None, None
            if is_turn:
                pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
                center_x = float(np.mean(pts[:, 0]))
                center_y = float(np.mean(pts[:, 1]))
                
                cv2.putText(
                    frame, "TARGET FOUND", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3,
                )

                target = DetectionTarget(
                    id=f"Custom Image (Matches: {len(good_matches)})",
                    center_x=center_x,
                    center_y=center_y,
                    priority=3,
                    is_turn_trigger=is_turn
                )
                result.targets.append(target)

        return frame, result
