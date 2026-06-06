"""
detectors/row_follower.py
=========================
Markerless crop-row follower.

Instead of looking for printed markers or poles, this detector uses the CROP
ROWS THEMSELVES. It:

  1. Segments vegetation (green plants) using the Excess-Green (ExG) index,
     which is robust to lighting.
  2. In a near band (just in front of the machine) it finds the soil PATH —
     the low-vegetation gap between two crop rows — and reports its centre.
     The DecisionEngine steers to keep that path centre in the middle of the
     frame, so the machine stays centred between the rows. No markers needed.
  3. In a far band (further ahead) it measures how much crop is still there.
     When the crop ahead runs out (the row ends and it opens into headland),
     it raises a turn trigger so the machine knows the row has ended.

Output: a single DetectionTarget whose center_x is the path centre and whose
is_turn_trigger flags the row end — so it plugs straight into the existing
steering / turn logic with no changes elsewhere.
"""

from typing import Tuple
import time

import cv2
import numpy as np

from config.config_manager import config_manager
from .base_detector import BaseDetector, DetectionResult, DetectionTarget, MarkerType
from utils.logger import get_logger

log = get_logger(__name__)


class RowFollowerDetector(BaseDetector):
    def __init__(self) -> None:
        c = "detectors.row_follower."
        # Vegetation segmentation
        self.exg_threshold = config_manager.get(c + "exg_threshold", 0.10)
        # Near band (steering): fractions of image height
        self.near_top = config_manager.get(c + "near_band_top", 0.55)
        self.near_bottom = config_manager.get(c + "near_band_bottom", 0.95)
        # Far band (row-end check)
        self.far_top = config_manager.get(c + "far_band_top", 0.30)
        self.far_bottom = config_manager.get(c + "far_band_bottom", 0.50)
        # Row-end: crop coverage in the far band below this fraction = ended
        self.rowend_veg_min = config_manager.get(c + "rowend_veg_min", 0.04)
        # Number of consecutive low-crop frames before declaring a row end
        self.rowend_frames = int(config_manager.get(c + "rowend_frames", 5))
        # Path gap detection: a column is "path" if its veg score is below
        # this fraction of the band's peak veg score.
        self.path_veg_frac = config_manager.get(c + "path_veg_frac", 0.35)

        self._rowend_counter = 0
        log.info(
            "RowFollowerDetector ready (ExG>%.2f, near=%.2f-%.2f, far=%.2f-%.2f).",
            self.exg_threshold,
            self.near_top,
            self.near_bottom,
            self.far_top,
            self.far_bottom,
        )

    # ── vegetation index ────────────────────────────────────────────────────
    def _vegetation_mask(self, frame: np.ndarray) -> np.ndarray:
        """Excess-Green mask: True where plants are."""
        b, g, r = cv2.split(frame.astype(np.float32))
        s = b + g + r + 1e-6
        exg = 2.0 * (g / s) - (r / s) - (b / s)
        mask = (exg > self.exg_threshold).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    # ── path centre from the near band ──────────────────────────────────────
    def _path_centre_x(self, mask: np.ndarray, h: int, w: int):
        """
        Find the centre x of the soil path (the low-vegetation gap between two
        crop rows) nearest the image centre. Returns (centre_x or None,
        column-vegetation profile for drawing).
        """
        y0 = int(h * self.near_top)
        y1 = int(h * self.near_bottom)
        band = mask[y0:y1, :]
        if band.size == 0:
            return None, None

        col = band.sum(axis=0).astype(np.float32)  # veg per column
        # Smooth the profile so small gaps in foliage don't fragment the path.
        k = max(3, w // 40) | 1  # odd kernel
        col = cv2.GaussianBlur(col.reshape(1, -1), (k, 1), 0).flatten()

        peak = col.max()
        if peak <= 0:
            # No vegetation at all in the near band -> drive straight.
            return w / 2.0, col

        is_path = col < (peak * self.path_veg_frac)  # low-veg = soil path

        # Find contiguous runs of path columns.
        runs = []
        start = None
        for x in range(w):
            if is_path[x] and start is None:
                start = x
            elif not is_path[x] and start is not None:
                runs.append((start, x - 1))
                start = None
        if start is not None:
            runs.append((start, w - 1))

        if not runs:
            return w / 2.0, col

        # Choose the path run whose centre is closest to the image centre
        # (that's the lane the machine is currently in).
        cx = w / 2.0
        best = min(runs, key=lambda r: abs((r[0] + r[1]) / 2.0 - cx))
        return (best[0] + best[1]) / 2.0, col

    # ── row-end from the far band ────────────────────────────────────────────
    def _row_ended(self, mask: np.ndarray, h: int, w: int) -> bool:
        y0 = int(h * self.far_top)
        y1 = int(h * self.far_bottom)
        band = mask[y0:y1, :]
        if band.size == 0:
            return False
        veg_frac = float(band.sum()) / float(band.size)
        if veg_frac < self.rowend_veg_min:
            self._rowend_counter += 1
        else:
            self._rowend_counter = 0
        return self._rowend_counter >= self.rowend_frames

    # ── main ──────────────────────────────────────────────────────────────--
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, DetectionResult]:
        t0 = time.time()
        result = DetectionResult()
        if frame is None or frame.size == 0:
            return frame, result

        h, w = frame.shape[:2]
        mask = self._vegetation_mask(frame)

        centre_x, col_profile = self._path_centre_x(mask, h, w)
        row_ended = self._row_ended(mask, h, w)

        # Confidence: how much vegetation is visible overall (more = surer).
        veg_frac_total = float(mask.sum()) / float(mask.size)
        confidence = float(min(1.0, veg_frac_total / 0.25))

        if centre_x is not None:
            target = DetectionTarget(
                id="ROW",
                distance_m=None,
                center_x=float(centre_x),
                center_y=float(h * (self.near_top + self.near_bottom) / 2.0),
                confidence=confidence,
                latency_ms=(time.time() - t0) * 1000,
                priority=1,
                is_turn_trigger=row_ended,
                marker_type=MarkerType.NORMAL,
            )
            result.targets.append(target)

        self._draw(frame, mask, centre_x, row_ended, h, w)
        result.frame_latency_ms = (time.time() - t0) * 1000
        return frame, result

    # ── overlay ───────────────────────────────────────────────────────────--
    def _draw(self, frame, mask, centre_x, row_ended, h, w) -> None:
        # Tint vegetation green.
        green = np.zeros_like(frame)
        green[:, :, 1] = mask * 255
        cv2.addWeighted(frame, 1.0, green, 0.25, 0, frame)

        # Near-band guides.
        y0 = int(h * self.near_top)
        y1 = int(h * self.near_bottom)
        cv2.line(frame, (0, y0), (w, y0), (180, 180, 180), 1)
        cv2.line(frame, (0, y1), (w, y1), (180, 180, 180), 1)

        # Image centre (target) vs detected path centre.
        cv2.line(frame, (w // 2, y0), (w // 2, y1), (200, 200, 200), 1)
        if centre_x is not None:
            cxp = int(centre_x)
            col = (0, 0, 255) if row_ended else (255, 80, 0)
            cv2.line(frame, (cxp, y0), (cxp, y1), col, 3)
            err = int(centre_x - w / 2.0)
            cv2.putText(
                frame,
                f"PATH offset: {err:+d}px",
                (10, y0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                col,
                2,
            )

        if row_ended:
            cv2.putText(
                frame,
                "!! ROW END - TURN !!",
                (w // 2 - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
