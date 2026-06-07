"""
detectors/ground_obstacle.py
============================
GENERIC obstacle detector for the path ahead — the software stand-in for LiDAR.

YOLO can only stop for the ~80 object types it was trained on (person, car,
dog, ...). It will NOT stop for an unknown object — a feed bag, a crate, a
rock, a fallen branch, a bucket — because it has never seen that class. Field
testing confirmed this: a bag lying in the alley produced ZERO YOLO detections.

This detector does not try to recognise *what* the object is. Instead it learns
what the DRIVABLE GROUND looks like immediately in front of the machine
(the grassy alley + any worn soil track) and then flags any sizeable patch
further ahead in the path corridor that does NOT look like that ground.

How it works (per frame):
  1. Downsample for speed and convert to CIELAB colour space.
  2. Sample a "reference" band just in front of the machine (assumed clear
     ground) and build a 2-D colour histogram of its a*/b* chroma. Because the
     band spans the whole near corridor it naturally captures BOTH the grass
     and the bare-soil track, so neither is treated as an obstacle.
  3. For every pixel in the look-ahead corridor, look up how common its colour
     is in that ground model. Rare colours = "not ground" = anomaly.
  4. Clean the anomaly mask, find blobs, and keep ones that are large and solid
     enough to be a real object (not grass speckle). A big blob in the corridor
     => a critical obstacle => the decision engine STOPS the machine.

It is colour/texture based, so it complements (does not replace) a real LiDAR.
Tune the thresholds per field; when in doubt it errs toward stopping.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from config.config_manager import config_manager
from .base_detector import BaseDetector, DetectionResult, ObstacleDetection
from utils.logger import get_logger

log = get_logger(__name__)


class GroundObstacleDetector(BaseDetector):
    """Generic 'anything in the path' obstacle detector (class-agnostic)."""

    def __init__(self) -> None:
        c = "detectors.ground_obstacle."
        self._enabled = config_manager.get(c + "enabled", True)
        self._work_w = int(config_manager.get(c + "work_width", 320))
        # Path-ahead corridor (fractions of frame), same idea as the YOLO ROI.
        self._half_w = float(config_manager.get(c + "roi_half_width", 0.30))
        self._eval_top = float(config_manager.get(c + "eval_top", 0.45))
        self._eval_bottom = float(config_manager.get(c + "eval_bottom", 0.90))
        # Ground-reference band (just in front of the machine = assumed clear).
        self._ref_top = float(config_manager.get(c + "ref_top", 0.80))
        self._ref_bottom = float(config_manager.get(c + "ref_bottom", 0.98))
        # Colour model + anomaly thresholds.
        self._bins = int(config_manager.get(c + "hist_bins", 24))
        self._min_ground_prob = float(
            config_manager.get(c + "min_ground_prob", 0.004)
        )
        # Darkness term: an object much darker than the ground (a dark bag/crate
        # in sun) has near-neutral chroma that blends with the colour model, so
        # we also flag pixels far below the ground's brightness. Set
        # dark_margin very high to disable.
        self._dark_margin = float(config_manager.get(c + "dark_margin", 28.0))
        # "Too grey for ground" term: grass and soil both carry real colour
        # (chroma); a rock, concrete block, cardboard box or weathered plank is
        # nearly grey (very low chroma). So a patch far less colourful than the
        # ground is flagged. Only applied when the ground itself is colourful.
        self._achroma_max = float(config_manager.get(c + "achroma_max", 12.0))
        # Blob size (fraction of the corridor area) to report / to STOP.
        self._min_area_frac = float(config_manager.get(c + "min_area_frac", 0.012))
        self._stop_area_frac = float(config_manager.get(c + "stop_area_frac", 0.020))
        self._solidity_min = float(config_manager.get(c + "solidity_min", 0.45))
        # Need a few consecutive hits before stopping (debounce flicker).
        self._confirm_frames = int(config_manager.get(c + "confirm_frames", 2))
        self._hit_streak = 0

    # ------------------------------------------------------------------
    def _corridor_x(self, w: int) -> Tuple[int, int]:
        lx = int((0.5 - self._half_w) * w)
        rx = int((0.5 + self._half_w) * w)
        return max(0, lx), min(w, rx)

    def _build_ground_model(self, lab: np.ndarray, lx: int, rx: int,
                            h: int) -> Optional[np.ndarray]:
        ry1 = int(self._ref_top * h)
        ry2 = int(self._ref_bottom * h)
        ref = lab[ry1:ry2, lx:rx]
        if ref.size == 0:
            return None
        # Brightness floor of the ground (5th percentile) for the darkness term.
        self._ground_L_lo = float(np.percentile(ref[:, :, 0], 5))
        a = ref[:, :, 1].reshape(-1)
        b = ref[:, :, 2].reshape(-1)
        # Typical colourfulness of the ground (median chroma) for the grey term.
        self._ground_chroma = float(
            np.median(np.hypot(a.astype(np.float32) - 128.0,
                               b.astype(np.float32) - 128.0))
        )
        hist = cv2.calcHist(
            [a.astype(np.float32), b.astype(np.float32)],
            [0, 1], None, [self._bins, self._bins],
            [0, 256, 0, 256],
        )
        # Slight blur so near-miss colours still count as ground, then normalise
        # to a probability (fraction of reference pixels per colour bin).
        hist = cv2.GaussianBlur(hist, (3, 3), 0)
        total = hist.sum()
        if total <= 0:
            return None
        return hist / total

    def _anomaly_mask(self, lab: np.ndarray, model: np.ndarray,
                      lx: int, rx: int, h: int) -> np.ndarray:
        ey1 = int(self._eval_top * h)
        ey2 = int(self._eval_bottom * h)
        sub = lab[ey1:ey2, lx:rx]
        bin_w = 256.0 / self._bins
        ai = np.clip((sub[:, :, 1] / bin_w).astype(np.int32), 0, self._bins - 1)
        bi = np.clip((sub[:, :, 2] / bin_w).astype(np.int32), 0, self._bins - 1)
        prob = model[ai, bi]
        chroma_anom = prob < self._min_ground_prob
        # Darkness anomaly: much darker than the ground reference.
        L = sub[:, :, 0].astype(np.float32)
        dark_anom = L < (getattr(self, "_ground_L_lo", 0.0) - self._dark_margin)
        # "Too grey" anomaly: far less colourful than the ground (rock, box,
        # concrete, weathered wood). Only when the ground is itself colourful,
        # so a naturally grey/gravel path doesn't trip it.
        chroma_px = np.hypot(sub[:, :, 1].astype(np.float32) - 128.0,
                             sub[:, :, 2].astype(np.float32) - 128.0)
        ground_chroma = getattr(self, "_ground_chroma", 0.0)
        if ground_chroma > self._achroma_max * 1.6:
            grey_anom = chroma_px < self._achroma_max
        else:
            grey_anom = np.zeros_like(dark_anom)
        mask = (chroma_anom | dark_anom | grey_anom).astype(np.uint8) * 255
        # Clean: drop grass speckle, then fill object interiors.
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        return mask

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, DetectionResult]:
        result = DetectionResult()
        if not self._enabled or frame is None or frame.size == 0:
            return frame, result

        H, W = frame.shape[:2]
        scale = self._work_w / float(W)
        small = cv2.resize(frame, (self._work_w, max(1, int(H * scale))))
        h, w = small.shape[:2]
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)

        lx, rx = self._corridor_x(w)
        model = self._build_ground_model(lab, lx, rx, h)
        if model is None:
            return frame, result

        mask = self._anomaly_mask(lab, model, lx, rx, h)
        ey1 = int(self._eval_top * h)
        corridor_area = max(mask.shape[0] * mask.shape[1], 1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        inv = 1.0 / scale  # map small-frame coords back to full frame
        biggest_frac = 0.0
        critical_now = False
        obstacles: List[ObstacleDetection] = []

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            frac = area / corridor_area
            if frac < self._min_area_frac:
                continue
            hull = cv2.convexHull(cnt)
            hull_area = max(cv2.contourArea(hull), 1.0)
            solidity = area / hull_area
            if solidity < self._solidity_min:
                continue  # scattered noise, not a solid object
            bx, by, bw, bh = cv2.boundingRect(cnt)
            # Back to small-frame absolute coords (corridor offset + eval offset)
            ax1, ay1 = lx + bx, ey1 + by
            ax2, ay2 = ax1 + bw, ay1 + bh
            is_crit = frac >= self._stop_area_frac
            critical_now = critical_now or is_crit
            biggest_frac = max(biggest_frac, frac)
            obstacles.append(
                ObstacleDetection(
                    label="object",
                    confidence=float(min(1.0, frac / self._stop_area_frac)),
                    bbox=(int(ax1 * inv), int(ay1 * inv),
                          int(ax2 * inv), int(ay2 * inv)),
                    class_id=-1,
                    is_critical=False,  # set after debounce below
                )
            )

        # Debounce: only declare critical after N consecutive critical frames.
        if critical_now:
            self._hit_streak += 1
        else:
            self._hit_streak = 0
        confirmed = self._hit_streak >= self._confirm_frames
        if confirmed:
            for o in obstacles:
                if (o.bbox[2] - o.bbox[0]) * (o.bbox[3] - o.bbox[1]) > 0:
                    o.is_critical = True

        result.obstacles = obstacles
        self._draw(frame, obstacles, confirmed, biggest_frac)
        return frame, result

    # ------------------------------------------------------------------
    def _draw(self, frame: np.ndarray, obstacles: List[ObstacleDetection],
              confirmed: bool, biggest_frac: float) -> None:
        for o in obstacles:
            x1, y1, x2, y2 = o.bbox
            crit = o.is_critical
            col = (0, 0, 255) if crit else (0, 200, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if crit else 2)
            tag = "OBJECT IN PATH" if crit else "object?"
            cv2.putText(frame, tag, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            if crit:
                cv2.putText(frame, "STOP", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def shutdown(self) -> None:  # symmetry with other detectors
        pass
