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
        cv2.line(
            frame,
            (mid - 300, self.height),
            (mid - 40, self.height // 2),
            (50, 80, 35),
            4,
        )
        cv2.line(
            frame,
            (mid + 300, self.height),
            (mid + 40, self.height // 2),
            (50, 80, 35),
            4,
        )

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
        trigger_color = (
            (0, 80, 255) if self._dist_m <= self._trigger_dist_m else (180, 180, 180)
        )
        cv2.putText(
            frame,
            f"SIM DIST: {self._dist_m:.2f} m  |  TRIGGER: {self._trigger_dist_m:.1f} m",
            (10, self.height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            trigger_color,
            2,
        )

        return frame


class OrchardSyntheticEnvironment:
    """Mock camera that renders a synthetic ORCHARD alley for the markerless
    stack: two textured tree walls with a grassy alley between them, plus
    scripted events — a generic obstacle in the path (STOP) and a red row-end
    pole that approaches (U-TURN). Lets you demo the new configuration with
    ``python main.py --simulate --config config/orchard.yaml`` (no video file).
    """

    _FOCAL_LENGTH = 800.0

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self._delay_s = 1.0 / fps
        self.is_open = False

        self._pole_w_m: float = config_manager.get("detectors.pole.width_m", 0.10)
        self._trigger_dist_m: float = config_manager.get(
            "detectors.pole.turn_trigger_distance_m", 1.5
        )
        self._base = self._build_base()        # static alley scene
        self._t0 = time.time()
        self._cycle = 18.0                     # seconds per full scripted loop

    # -- scene construction -------------------------------------------------
    def _build_base(self) -> np.ndarray:
        W, H = self.width, self.height
        rng = np.random.default_rng(7)
        vp = (W // 2, int(H * 0.32))           # vanishing point

        sky = np.full((H, W, 3), (150, 130, 110), np.uint8)   # hazy hillside
        # textured tree foliage (green + STRONG noise so it reads as a wall)
        wall = np.full((H, W, 3), (35, 95, 40), np.uint8).astype(np.int16)
        wall += rng.integers(-45, 45, (H, W, 3), dtype=np.int16)
        wall = np.clip(wall, 0, 255).astype(np.uint8)
        # grassy alley (green + gentle noise so it stays SMOOTH, not a wall)
        grass = np.full((H, W, 3), (55, 130, 70), np.uint8).astype(np.int16)
        grass += rng.integers(-8, 8, (H, W, 3), dtype=np.int16)
        grass = np.clip(grass, 0, 255).astype(np.uint8)

        # alley polygon (trapezoid from the bottom up to the vanishing point)
        alley = np.array([[int(W * 0.30), H], [int(W * 0.70), H],
                          [vp[0] + 34, vp[1]], [vp[0] - 34, vp[1]]], np.int32)
        wall_mask = np.zeros((H, W), np.uint8)
        # left & right walls = everything between the horizon and the alley edges
        lwall = np.array([[0, vp[1]], [alley[3][0], vp[1]], [alley[0][0], H],
                          [0, H]], np.int32)
        rwall = np.array([[W, vp[1]], [alley[2][0], vp[1]], [alley[1][0], H],
                          [W, H]], np.int32)
        cv2.fillPoly(wall_mask, [lwall, rwall], 255)
        alley_mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(alley_mask, [alley], 255)

        frame = sky.copy()
        frame[wall_mask > 0] = wall[wall_mask > 0]
        frame[alley_mask > 0] = grass[alley_mask > 0]
        self._alley_left = (alley[0][0], alley[3][0])  # bottom/top left x
        return frame

    # -- camera interface ---------------------------------------------------
    def open(self) -> bool:
        self.is_open = True
        self._t0 = time.time()
        log.info("OrchardSyntheticEnvironment simulation started.")
        return True

    def release(self) -> None:
        self.is_open = False
        log.info("OrchardSyntheticEnvironment stopped.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()

    def read_frame(self):
        if not self.is_open:
            return False, None
        W, H = self.width, self.height
        t = (time.time() - self._t0) % self._cycle
        frame = self._base.copy()
        caption = "DRIVING - markerless alley following"

        if 4.0 <= t < 9.0:
            # Generic obstacle approaching in the path -> STOP
            f = (t - 4.0) / 5.0
            bw = int(W * (0.05 + 0.11 * f))
            bh = int(H * (0.04 + 0.09 * f))
            cy = int(H * (0.55 + 0.18 * f))
            cv2.rectangle(frame, (W // 2 - bw, cy - bh),
                          (W // 2 + bw, cy + bh), (95, 95, 100), -1)
            caption = "OBSTACLE in path -> STOP"
        elif 11.0 <= t < 17.0:
            # Red row-end pole approaching on the alley edge -> U-TURN
            dist = max(0.6, 5.0 - (t - 11.0) * 0.9)
            pw = max(8, int(self._FOCAL_LENGTH * self._pole_w_m / dist))
            ph = pw * 9
            x0 = int(W * 0.20)
            y2 = int(H * 0.88)
            cv2.rectangle(frame, (x0, y2 - ph), (x0 + pw, y2), (40, 40, 210), -1)
            caption = f"RED POLE {dist:.1f} m -> U-TURN"

        cv2.rectangle(frame, (0, H - 30), (W, H), (20, 20, 20), -1)
        cv2.putText(frame, caption, (10, H - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        time.sleep(self._delay_s)
        return True, frame
