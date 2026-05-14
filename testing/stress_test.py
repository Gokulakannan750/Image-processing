"""
testing/stress_test.py
======================
Simulates system stress (frame drops, lag, noise) to validate robustness.
"""
import time
import random
import numpy as np
import cv2
from typing import Optional

class StressTestSimulator:
    def __init__(self):
        self.drop_rate = 0.0
        self.lag_ms = 0
        self.noise_intensity = 0

    def configure(self, drop_rate: float = 0.0, lag_ms: int = 0, noise_intensity: int = 0):
        self.drop_rate = drop_rate
        self.lag_ms = lag_ms
        self.noise_intensity = noise_intensity

    def process(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns None if frame is dropped, or a modified frame.
        """
        # 1. Frame Drop
        if random.random() < self.drop_rate:
            return None

        # 2. Artificial Lag
        if self.lag_ms > 0:
            time.sleep(self.lag_ms / 1000.0)

        # 3. Heavy Noise
        if self.noise_intensity > 0:
            gauss = np.random.normal(0, self.noise_intensity, frame.shape).astype(np.uint8)
            frame = cv2.add(frame, gauss)

        return frame
