"""
camera/calibration_validator.py
===============================
Validates calibration quality by checking reprojection errors on markers.
"""
import cv2
import numpy as np
from typing import List, Tuple
from utils.logger import get_logger

log = get_logger(__name__)

class CalibrationValidator:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def validate_marker_pose(self, corners: np.ndarray, marker_length: float, rvec: np.ndarray, tvec: np.ndarray) -> float:
        """
        Calculates the reprojection error for a single marker.
        Returns the RMSE (Root Mean Square Error) in pixels.
        """
        # 3D coordinates of marker corners in its own coordinate system
        obj_points = np.array([
            [-marker_length/2,  marker_length/2, 0],
            [ marker_length/2,  marker_length/2, 0],
            [ marker_length/2, -marker_length/2, 0],
            [-marker_length/2, -marker_length/2, 0]
        ], dtype=np.float32)

        # Project 3D points back to 2D
        projected_points, _ = cv2.projectPoints(
            obj_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        
        projected_points = projected_points.reshape(-1, 2)
        corners = corners.reshape(-1, 2)

        # Calculate Euclidean distance between original corners and reprojected ones
        error = np.sqrt(np.sum((corners - projected_points)**2, axis=1))
        rmse = np.mean(error)
        
        if rmse > 5.0: # Arbitrary threshold for "poor" calibration
            log.warning("High reprojection error detected: %.2f pixels. Check camera calibration!", rmse)
            
        return rmse
