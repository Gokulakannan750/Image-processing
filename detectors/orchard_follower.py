"""
detectors/orchard_follower.py
=============================
Markerless ORCHARD alley follower (validated on real apple-orchard footage).

In an orchard the path is a grassy alley between two tall tree walls — both the
trees AND the alley are green, so colour alone cannot separate them. What DOES
separate them is TEXTURE: the tree foliage is highly textured/dark, while the
mown alley floor is comparatively smooth.

Algorithm (centred driving view):
  1. Build a "tree-wall" mask = green AND high local texture.
  2. For each scanline in front of the machine, scan left and right from the
     running centre until the tree wall is hit. The midpoint of that gap is the
     alley centre at that row.
  3. The median of the near-band centres is the steering target (center_x); the
     existing DecisionEngine steers to keep it in the middle of the frame.
  4. Row end: when the tree walls stop appearing ahead for several frames (the
     machine exits the corridor into the open headland), raise a turn trigger.

Output: one DetectionTarget (center_x = alley centre, is_turn_trigger = row
end), so it plugs into the existing steering / turn logic unchanged.
"""

from typing import Tuple
import time

import cv2
import numpy as np

from config.config_manager import config_manager
from .base_detector import BaseDetector, DetectionResult, DetectionTarget, MarkerType
from utils.logger import get_logger

log = get_logger(__name__)


class OrchardFollowerDetector(BaseDetector):
    def __init__(self) -> None:
        c = "detectors.orchard."
        self.work_width = int(config_manager.get(c + "work_width", 320))
        self.green_exg_min = config_manager.get(c + "green_exg_min", 0.02)
        self.tree_texture_pct = config_manager.get(c + "tree_texture_pct", 52)
        # Absolute minimum local texture (grayscale std) to count as a tree
        # wall. Prevents smooth scenes (open headland, grass, sky) from being
        # mistaken for walls due to tiny numeric noise.
        self.tree_texture_abs_min = config_manager.get(c + "tree_texture_abs_min", 6.0)
        self.near_top = config_manager.get(c + "near_band_top", 0.60)
        self.near_bottom = config_manager.get(c + "near_band_bottom", 0.92)
        self.look_top = config_manager.get(c + "lookahead_top", 0.45)
        self.look_bottom = config_manager.get(c + "lookahead_bottom", 0.65)
        self.scan_top = config_manager.get(c + "scan_top", 0.45)
        self.scan_bottom = config_manager.get(c + "scan_bottom", 0.92)
        self.ema_alpha = config_manager.get(c + "ema_alpha", 0.4)
        # Row end: if fewer than this fraction of look-ahead rows have walls on
        # BOTH sides, for rowend_frames consecutive frames, the row has ended.
        self.rowend_wall_min = config_manager.get(c + "rowend_wall_min", 0.25)
        self.rowend_frames = int(config_manager.get(c + "rowend_frames", 6))

        self._ema_cx = None  # smoothed centre (in working-image px)
        self._rowend_counter = 0
        log.info(
            "OrchardFollowerDetector ready (work_width=%d, tex_pct=%d).",
            self.work_width,
            self.tree_texture_pct,
        )

    # ── tree-wall mask ───────────────────────────────────────────────────────
    def _tree_mask(self, small: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        kt = max(7, (small.shape[1] // 15) | 1)  # odd texture kernel
        mean = cv2.blur(gray, (kt, kt))
        std = np.sqrt(np.maximum(cv2.blur(gray * gray, (kt, kt)) - mean * mean, 0))

        b, g, r = cv2.split(small.astype(np.float32))
        s = b + g + r + 1e-6
        exg = 2.0 * (g / s) - (r / s) - (b / s)
        green = exg > self.green_exg_min

        thr = max(np.percentile(std, self.tree_texture_pct), self.tree_texture_abs_min)
        tree = ((std > thr) & green).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        tree = cv2.morphologyEx(tree, cv2.MORPH_CLOSE, k)
        return tree

    @staticmethod
    def _gap_runs(row: np.ndarray):
        """Find runs of non-tree (alley) pixels in a scanline.

        Returns a list of (start, end, bounded) where 'bounded' is True if the
        gap has a tree wall on BOTH sides (i.e. doesn't touch either edge).

        Vectorised with numpy (no per-pixel Python loop) for speed.
        """
        w = row.shape[0]
        free = (row == 0).astype(np.int8)
        if not free.any():
            return []
        # Transitions: +1 where a free-run starts, -1 just after it ends.
        diff = np.diff(np.concatenate(([0], free, [0])))
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1) - 1  # inclusive end index
        out = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            if (e - s) >= 2:  # ignore 1-2px slivers
                out.append((s, e, (s > 0 and e < w - 1)))
        return out

    # ── main ──────────────────────────────────────────────────────────────--
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, DetectionResult]:
        t0 = time.time()
        result = DetectionResult()
        if frame is None or frame.size == 0:
            return frame, result

        H, W = frame.shape[:2]
        scale = self.work_width / float(W)
        sw = self.work_width
        sh = max(1, int(H * scale))
        small = cv2.resize(frame, (sw, sh))
        tree = self._tree_mask(small)

        # For each scanline, pick the alley as the gap (run of non-tree pixels)
        # BOUNDED BY tree walls and nearest the running centre. Using runs makes
        # this independent of where the scan happens to start.
        cx = sw // 2
        centres = []  # (x, y) in working coords
        look_rows = 0
        look_walls = 0
        y0s, y1s = int(sh * self.scan_top), int(sh * self.scan_bottom)
        ly0, ly1 = int(sh * self.look_top), int(sh * self.look_bottom)

        for y in range(y1s, y0s, -2):
            row = tree[y, :]
            gaps = self._gap_runs(row)  # list of (start, end, bounded)
            if not gaps:
                continue
            # Prefer gaps bounded by walls on both sides; else any gap.
            bounded = [g for g in gaps if g[2]]
            pool = bounded if bounded else gaps
            start, end, is_bounded = min(
                pool, key=lambda g: abs((g[0] + g[1]) / 2.0 - cx)
            )
            c = (start + end) // 2
            centres.append((c, y))
            cx = c
            if ly0 <= y <= ly1:
                look_rows += 1
                if is_bounded:
                    look_walls += 1

        # Steering centre = median of near-band centres (working coords).
        near = [c for c, y in centres if y > sh * self.near_top]
        if near:
            raw_cx = float(np.median(near))
            self._ema_cx = (
                raw_cx
                if self._ema_cx is None
                else self.ema_alpha * raw_cx + (1 - self.ema_alpha) * self._ema_cx
            )

        # Row end: too few look-ahead rows are bounded by walls on both sides.
        wall_frac = (look_walls / look_rows) if look_rows else 0.0
        if wall_frac < self.rowend_wall_min:
            self._rowend_counter += 1
        else:
            self._rowend_counter = 0
        row_ended = self._rowend_counter >= self.rowend_frames

        # Emit a target if we have a centre, OR if the row has ended (so the
        # turn trigger always reaches the DecisionEngine).
        if self._ema_cx is not None or row_ended:
            cx_work = self._ema_cx if self._ema_cx is not None else (sw / 2.0)
            full_cx = cx_work / scale  # back to full-res px
            target = DetectionTarget(
                id="ORCHARD_ROW",
                distance_m=None,
                center_x=float(full_cx),
                center_y=float(H * (self.near_top + self.near_bottom) / 2.0),
                confidence=float(min(1.0, len(near) / 15.0)),
                latency_ms=(time.time() - t0) * 1000,
                priority=1,
                is_turn_trigger=row_ended,
                marker_type=MarkerType.NORMAL,
            )
            result.targets.append(target)

        self._draw(frame, tree, centres, scale, row_ended, H, W)
        result.frame_latency_ms = (time.time() - t0) * 1000
        return frame, result

    # ── overlay ───────────────────────────────────────────────────────────--
    def _draw(self, frame, tree, centres, scale, row_ended, H, W) -> None:
        # Tint detected tree walls.
        tmask = cv2.resize(tree * 255, (W, H), interpolation=cv2.INTER_NEAREST)
        tint = np.zeros_like(frame)
        tint[:, :, 2] = tmask  # red tint on tree walls
        cv2.addWeighted(frame, 1.0, tint, 0.18, 0, frame)

        for c, y in centres:
            cv2.circle(frame, (int(c / scale), int(y / scale)), 3, (0, 0, 255), -1)

        if self._ema_cx is not None:
            fx = int(self._ema_cx / scale)
            col = (0, 0, 255) if row_ended else (255, 80, 0)
            y0 = int(H * self.near_top)
            cv2.line(frame, (fx, y0), (fx, H), col, 3)
            cv2.line(frame, (W // 2, y0), (W // 2, H), (200, 200, 200), 1)
            cv2.putText(
                frame,
                f"ALLEY offset: {fx - W // 2:+d}px",
                (10, y0 - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                col,
                2,
            )
        if row_ended:
            cv2.putText(
                frame,
                "!! ROW END - TURN !!",
                (W // 2 - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
