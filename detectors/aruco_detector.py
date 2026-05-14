"""
detectors/aruco_detector.py
===========================
Classical CV detector using ArUco fiducial markers.
"""
from typing import Tuple

import cv2
import cv2.aruco as aruco
import numpy as np

from config.config_manager import config_manager
from detectors.base_detector import BaseDetector, DetectionResult, DetectionTarget
from utils.logger import get_logger
from utils.exceptions import CalibrationLoadError
from utils.math_utils import rotation_vector_to_euler, estimate_blur
import time

log = get_logger(__name__)


class ArucoDetector(BaseDetector):
    # Mapping for common ArUco dictionaries
    DICTIONARY_MAP = {
        "DICT_4X4_50": aruco.DICT_4X4_50,
        "DICT_4X4_100": aruco.DICT_4X4_100,
        "DICT_4X4_250": aruco.DICT_4X4_250,
        "DICT_4X4_1000": aruco.DICT_4X4_1000,
        "DICT_5X5_50": aruco.DICT_5X5_50,
        "DICT_5X5_100": aruco.DICT_5X5_100,
        "DICT_5X5_250": aruco.DICT_5X5_250,
        "DICT_5X5_1000": aruco.DICT_5X5_1000,
        "DICT_6X6_50": aruco.DICT_6X6_50,
        "DICT_6X6_100": aruco.DICT_6X6_100,
        "DICT_6X6_250": aruco.DICT_6X6_250,
        "DICT_6X6_1000": aruco.DICT_6X6_1000,
        "DICT_7X7_50": aruco.DICT_7X7_50,
        "DICT_7X7_100": aruco.DICT_7X7_100,
        "DICT_7X7_250": aruco.DICT_7X7_250,
        "DICT_7X7_1000": aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
    }

    def __init__(self, dict_type: int = None) -> None:
        # Load dictionary from config or use provided default
        dict_name = config_manager.get("detectors.aruco.dictionary", "DICT_4X4_50")
        
        if dict_type is None:
            dict_type = self.DICTIONARY_MAP.get(dict_name, aruco.DICT_4X4_50)
            if dict_name not in self.DICTIONARY_MAP:
                log.warning(f"Unknown dictionary '{dict_name}' in config. Defaulting to DICT_4X4_50.")

        self.aruco_dict = aruco.getPredefinedDictionary(dict_type)
        self.parameters = aruco.DetectorParameters()

        cam_mat = config_manager.get("detectors.aruco.camera_matrix")
        dist = config_manager.get("detectors.aruco.dist_coeffs")
        
        if cam_mat is None or dist is None:
            raise CalibrationLoadError("ArUco calibration matrices missing in config.")

        self.camera_matrix = np.array(cam_mat, dtype=np.float32)
        self.dist_coeffs = np.array(dist, dtype=np.float32)
        self.marker_length = config_manager.get("detectors.aruco.marker_length_m", 0.20)
        self.turn_trigger_distance = config_manager.get("detectors.aruco.turn_trigger_distance_m", 1.5)
        self.downsample_width = config_manager.get("detectors.aruco.downsample_width", 0)
        self.show_rejected = config_manager.get("detectors.aruco.show_rejected", False)

        # Tune parameters for maximum robustness
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 10
        self.parameters.adaptiveThreshConstant = 7  # Balanced sensitivity
        self.parameters.minMarkerPerimeterRate = 0.03 # Standard value
        self.parameters.maxMarkerPerimeterRate = 4.0
        self.parameters.polygonalApproxAccuracyRate = 0.05 # Handle perspective better

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, DetectionResult]:
        start_time = time.time()
        
        # Optional downsampling for performance
        h, w = frame.shape[:2]
        if self.downsample_width > 0 and w > self.downsample_width:
            scale = self.downsample_width / w
            target_h = int(h * scale)
            small_frame = cv2.resize(frame, (self.downsample_width, target_h))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        else:
            scale = 1.0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Try primary dictionary
        corners, ids, rejected = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.parameters
        )

        # 2. Robust fallback
        if ids is None:
            for fallback_dict_type in [aruco.DICT_6X6_50, aruco.DICT_6X6_100, aruco.DICT_6X6_1000]:
                if fallback_dict_type == self.aruco_dict: continue
                f_corners, f_ids, f_rejected = aruco.detectMarkers(
                    gray, aruco.getPredefinedDictionary(fallback_dict_type), parameters=self.parameters
                )
                if f_ids is not None:
                    corners, ids, rejected = f_corners, f_ids, f_rejected
                    break

        result = DetectionResult()
        result.frame_latency_ms = (time.time() - start_time) * 1000

        # Debug: Draw rejected
        if self.show_rejected and rejected is not None and len(rejected) > 0:
            for i in range(len(rejected)):
                r_corners = rejected[i] / scale if scale != 1.0 else rejected[i]
                cv2.polylines(frame, [r_corners.astype(np.int32)], True, (100, 100, 100), 1)

        if ids is not None:
            if scale != 1.0:
                for i in range(len(ids)):
                    corners[i] = corners[i] / scale
            
            aruco.drawDetectedMarkers(frame, corners, ids)
            blur_score = estimate_blur(gray)

            for i in range(len(ids)):
                try:
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                        corners[i], self.marker_length, self.camera_matrix, self.dist_coeffs
                    )

                    pitch, yaw, roll = rotation_vector_to_euler(rvec[0])

                    if config_manager.get("debug.draw_pose_axes", True):
                        cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs,
                                          rvec[0], tvec[0], self.marker_length * 0.5)

                    distance_to_marker = tvec[0][0][2]
                    c_xy = corners[i][0]
                    center_x = float(np.mean(c_xy[:, 0]))
                    center_y = float(np.mean(c_xy[:, 1]))

                    # Calculate Confidence
                    # Factors: Size (min 3% perimeter), Blur, Distance (closer is better)
                    size_factor = min(1.0, (cv2.arcLength(corners[i], True) / (w * 0.05)))
                    blur_factor = min(1.0, blur_score / 100.0)
                    confidence = (size_factor * 0.4) + (blur_factor * 0.3) + 0.3

                    target = DetectionTarget(
                        id=f"ID:{ids[i][0]}",
                        distance_m=float(distance_to_marker),
                        center_x=center_x,
                        center_y=center_y,
                        yaw=float(yaw),
                        pitch=float(pitch),
                        roll=float(roll),
                        confidence=float(confidence),
                        blur_score=float(blur_score),
                        latency_ms=result.frame_latency_ms,
                        priority=0 if ids[i][0] == 0 else 1,
                        is_turn_trigger=distance_to_marker <= self.turn_trigger_distance
                    )
                    
                    # Advanced Overlay Info
                    if config_manager.get("debug.advanced_overlay", True):
                        ov_text = f"{target.id} | Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f} | Conf:{confidence:.2f}"
                        cv2.putText(frame, ov_text, (int(c_xy[0][0]), int(c_xy[0][1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                    result.targets.append(target)
                except Exception as e:
                    log.error(f"Pose failed: {e}")
            
        return frame, result
