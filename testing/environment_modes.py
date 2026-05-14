"""
testing/environment_modes.py
============================
Handles simulated environmental conditions for robustness testing.
"""
from enum import Enum
import cv2
import numpy as np
from typing import Tuple

class TestMode(Enum):
    NORMAL = "NORMAL"
    LOW_LIGHT = "LOW_LIGHT"
    BRIGHT_LIGHT = "BRIGHT_LIGHT"
    MOTION_BLUR = "MOTION_BLUR"
    NOISY_CAMERA = "NOISY_CAMERA"

class EnvironmentSimulator:
    @staticmethod
    def apply_mode(frame: np.ndarray, mode: TestMode) -> np.ndarray:
        if mode == TestMode.NORMAL:
            return frame
        
        if mode == TestMode.LOW_LIGHT:
            # Reduce brightness and add some shadow noise
            return cv2.convertScaleAbs(frame, alpha=0.3, beta=10)
            
        if mode == TestMode.BRIGHT_LIGHT:
            # Overexpose
            return cv2.convertScaleAbs(frame, alpha=1.5, beta=50)
            
        if mode == TestMode.MOTION_BLUR:
            # Horizontal motion blur kernel
            kernel_size = 15
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel /= kernel_size
            return cv2.filter2D(frame, -1, kernel)
            
        if mode == TestMode.NOISY_CAMERA:
            # Gaussian noise
            gauss = np.random.normal(0, 25, frame.shape).astype(np.uint8)
            return cv2.add(frame, gauss)
            
        return frame

    @staticmethod
    def draw_overlay(frame: np.ndarray, mode: TestMode) -> None:
        text = f"TEST MODE: {mode.value}"
        color = (255, 255, 255)
        if mode != TestMode.NORMAL:
            color = (0, 165, 255) # Orange for active test modes
            
        cv2.putText(frame, text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
