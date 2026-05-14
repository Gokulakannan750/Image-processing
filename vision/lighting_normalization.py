"""
vision/lighting_normalization.py
================================
Applies image enhancement to improve detection in challenging lighting.
"""
import cv2
import numpy as np
from config.config_manager import config_manager

class LightingNormalizer:
    def __init__(self):
        self.method = config_manager.get("normalization.method", "CLAHE")
        self.clip_limit = config_manager.get("normalization.clahe_clip_limit", 2.0)
        self.grid_size = tuple(config_manager.get("normalization.clahe_grid_size", [8, 8]))
        self.gamma = config_manager.get("normalization.gamma_correction", 1.0)
        
        self._clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.grid_size)

    def process(self, frame: np.ndarray) -> np.ndarray:
        if not config_manager.get("normalization.enabled", True):
            return frame

        # 1. Gamma Correction (optional)
        if self.gamma != 1.0:
            invGamma = 1.0 / self.gamma
            table = np.array([((i / 255.0) ** invGamma) * 255
                             for i in np.arange(0, 256)]).astype("uint8")
            frame = cv2.LUT(frame, table)

        # 2. Histogram Equalization
        if self.method == "CLAHE":
            # Convert to LAB to equalize only lightness channel
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = self._clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
        elif self.method == "SIMPLE":
            # Simple global equalization on Y channel (YCrCb)
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        return frame
