"""
vision/pose_filter.py
=====================
Applies low-pass/EMA filtering to raw detection coordinates and distances
to eliminate high-frequency jitter and stabilize navigation.
"""
from typing import Optional
import numpy as np

class PoseFilter:
    """
    Exponential Moving Average (EMA) filter for 1D or 2D values.
    Smooths jittery detection values over time.
    """
    def __init__(self, alpha: float = 0.3):
        """
        alpha: Smoothing factor (0 < alpha <= 1). 
               Higher = more responsive, less smooth.
               Lower = smoother, more latency.
        """
        self.alpha = alpha
        self._current_value: Optional[np.ndarray] = None

    def update(self, new_value: float | np.ndarray) -> float | np.ndarray:
        if self._current_value is None:
            self._current_value = np.array(new_value, dtype=np.float32)
            return self._current_value

        self._current_value = (self.alpha * np.array(new_value, dtype=np.float32) + 
                               (1.0 - self.alpha) * self._current_value)
        
        # If it was a single float, return as float. Else return array.
        if isinstance(new_value, (float, int)):
            return float(self._current_value)
        return self._current_value

    def reset(self):
        self._current_value = None
