"""
simulation/synthetic_environment.py
===================================
Generates synthetic farm-row frames with a real ArUco marker that
approaches the camera using a pinhole projection model.

The marker pixel size is derived from:
    pixel_size = focal_length * marker_length_m / distance_m

This ensures the ArUco detector's solvePnP pose estimate matches the
simulated distance, so turn-trigger logic fires at the correct threshold.
"""
import time

import cv2
import cv2.aruco as aruco
import numpy as np

from config.config_manager import config_manager
from utils.logger import get_logger

log = get_logger(__name__)


class SyntheticEnvironment:
    """Mock camera that renders a physically-consistent ArUco marker."""

    _FOCAL_LENGTH = 800.0  # must match camera_matrix in default.yaml

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self._delay_s = 1.0 / fps
        self.is_open = False

        self._marker_length_m: float = config_manager.get(
            "detectors.aruco.marker_length_m", 0.20
        )
        self._trigger_dist_m: float = config_manager.get(
            "detectors.aruco.turn_trigger_distance_m", 1.5
        )

        # Simulation state
        self._dist_m: float = 5.0
        self._approach_speed: float = 0.4  # m/s
        self._marker_cx: float = width / 2.0
        self._last_time: float = time.time()

        # Pre-generate a real ArUco marker image
        dict_name = config_manager.get("detectors.aruco.dictionary", "DICT_6X6_250")
        aruco_dict = aruco.getPredefinedDictionary(
            getattr(aruco, dict_name, aruco.DICT_6X6_250)
        )
        self._marker_src = aruco.generateImageMarker(aruco_dict, 0, 200)

    def open(self) -> bool:
        self.is_open = True
        self._last_time = time.time()
        log.info("SyntheticEnvironment simulation started.")
        return True

    def release(self) -> None:
        self.is_open = False
        log.info("SyntheticEnvironment stopped.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()

    def read_frame(self):
        if not self.is_open:
            return False, None

        now = time.time()
        dt = min(now - self._last_time, 0.1)
        self._last_time = now

        # Advance simulated distance
        self._dist_m -= self._approach_speed * dt

        # Reset when marker passes well inside the trigger zone
        if self._dist_m < 0.8:
            self._dist_m = 5.0
            self._marker_cx = self.width / 2.0 + float(np.random.randint(-180, 180))

        # Pinhole model: pixel_size = f * L / d
        px = max(int(self._FOCAL_LENGTH * self._marker_length_m / self._dist_m), 6)

        frame = self._build_frame(px)
        time.sleep(self._delay_s)
        return True, frame

    def _build_frame(self, px: int) -> np.ndarray:
        frame = np.full((self.height, self.width, 3), 80, dtype=np.uint8)

        # Crop row lines converging toward horizon
        mid = self.width // 2
        cv2.line(frame, (mid - 300, self.height), (mid - 40, self.height // 2), (50, 80, 35), 4)
        cv2.line(frame, (mid + 300, self.height), (mid + 40, self.height // 2), (50, 80, 35), 4)

        # Resize and place the real ArUco marker
        marker_bgr = cv2.cvtColor(
            cv2.resize(self._marker_src, (px, px), interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_GRAY2BGR,
        )

        x1 = int(self._marker_cx - px / 2)
        y1 = int(self.height // 2 - px // 2)
        x2, y2 = x1 + px, y1 + px

        if x1 >= 0 and y1 >= 0 and x2 <= self.width and y2 <= self.height:
            frame[y1:y2, x1:x2] = marker_bgr

        # Distance HUD
        trigger_color = (0, 80, 255) if self._dist_m <= self._trigger_dist_m else (180, 180, 180)
        cv2.putText(
            frame,
            f"SIM DIST: {self._dist_m:.2f} m  |  TRIGGER: {self._trigger_dist_m:.1f} m",
            (10, self.height - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, trigger_color, 2,
        )

        return frame
