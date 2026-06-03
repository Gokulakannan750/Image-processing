#!/usr/bin/env python3
"""
simulation/field_simulator.py
==============================
Top-down 2D visual simulation of the AgriBot field operation.

Field layout matches the real field diagram:
  - Crop rows (green) separated by vertical pole lines
  - Machine travels in the paths BETWEEN the crop rows
  - Markers are mounted on the SIDE POLES at the headland ends of each path
    (NOT at the path centre — on the pole to the right of the machine path)
  - Camera detects the side-pole marker as the machine approaches the headland

Marker types:
  NORMAL   → U-turn, continue to next row
  LAST-ROW → final U-turn, then watch for STOP marker
  STOP     → field complete, machine halts permanently

Controls:
  SPACE / P  — pause / resume
  R          — restart from the beginning
  +  / =     — speed up
  -          — slow down
  Q / ESC    — quit

Run with:
  python simulation/field_simulator.py
"""

import cv2
import numpy as np
import math
import time

# ─────────────────────────────────────────────────────────────────────────────
#  LAYOUT CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
WIN_W, WIN_H = 1280, 720
PANEL_W = 330  # right info panel width
FIELD_W = WIN_W - PANEL_W  # 950 px field canvas

NUM_PATHS = 6  # number of row paths machine traverses
CROP_W = 46  # green crop strip width (px)
PATH_W = 60  # machine-path width (px)
STRIP = CROP_W + PATH_W  # one repeating field unit

MARGIN_L = 50  # left headland margin (px)
MARGIN_T = 95  # top headland height (px)
MARGIN_B = 85  # bottom headland height (px)

# Path centre x positions (machine drives along these x values)
PATH_CX = [MARGIN_L + CROP_W + PATH_W // 2 + i * STRIP for i in range(NUM_PATHS)]

# Pole x positions — right-edge pole of each path (where markers are mounted)
# Pole is between path[i] and the next crop row to its right
POLE_X = [MARGIN_L + CROP_W + (i + 1) * STRIP for i in range(NUM_PATHS)]
# Also the leftmost pole (left edge of path 0 = right edge of first crop row)
LEFT_POLE_X = MARGIN_L + CROP_W

FIELD_TOP_Y = MARGIN_T
FIELD_BOT_Y = WIN_H - MARGIN_B

# Marker geometry
MK_HALF = 13  # half-size of marker square (px)
POLE_LINE_W = 5  # vertical pole line width (px)
HEADLAND_EXT = 22  # how far pole extends into headland (px)

# Detection thresholds (px from machine centre to marker centre)
DETECT_DIST = 145  # camera first sees marker
TRIGGER_DIST = 62  # turn fires (≈ 1.5 m real scale)

# Machine geometry
M_W, M_H = 26, 44  # machine body width / height (px)
CAM_HALF_DEG = 60  # camera half-angle — wide enough to see angled side poles
CAM_RANGE = 150  # camera cone length (px)

# Animation
BASE_SPEED = 2.2  # px/frame when driving

# ─────────────────────────────────────────────────────────────────────────────
#  COLOURS  (BGR)
# ─────────────────────────────────────────────────────────────────────────────
C_HEADLAND = (185, 185, 165)
C_CROP = (25, 90, 25)
C_PATH = (210, 198, 172)
C_POLE_LINE = (40, 55, 70)  # dark pole lines (like black in diagram)
C_POLE_CAP = (60, 80, 100)  # pole cap block at ends

C_MK_NORMAL = (30, 150, 255)  # orange for normal markers
C_MK_LAST = (255, 90, 20)  # blue for LAST-ROW marker
C_MK_STOP = (20, 20, 200)  # red for STOP marker
C_MK_SEEN = (40, 220, 60)  # green flash on detection
C_MK_DONE = (100, 100, 100)  # grey after triggered

C_MACHINE = (230, 230, 255)
C_MACHINE_BD = (80, 80, 200)
C_CAM_IDLE = (180, 180, 40)
C_CAM_DETECT = (40, 230, 60)
C_CAM_TRIG = (30, 30, 220)

C_PANEL = (18, 18, 24)
C_DIV = (45, 45, 55)
C_WHITE = (255, 255, 255)
C_DIM = (120, 120, 130)
C_GREEN = (50, 210, 50)
C_YELLOW = (40, 210, 210)
C_RED = (60, 60, 220)
C_BLUE = (220, 150, 30)


# ─────────────────────────────────────────────────────────────────────────────
#  BEZIER HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def bez(P0, P1, P2, P3, t):
    mt = 1.0 - t
    return (
        mt**3 * P0[0] + 3 * mt**2 * t * P1[0] + 3 * mt * t**2 * P2[0] + t**3 * P3[0],
        mt**3 * P0[1] + 3 * mt**2 * t * P1[1] + 3 * mt * t**2 * P2[1] + t**3 * P3[1],
    )


def bez_angle(P0, P1, P2, P3, t):
    mt = 1.0 - t
    dx = 3 * (
        mt**2 * (P1[0] - P0[0]) + 2 * mt * t * (P2[0] - P1[0]) + t**2 * (P3[0] - P2[0])
    )
    dy = 3 * (
        mt**2 * (P1[1] - P0[1]) + 2 * mt * t * (P2[1] - P1[1]) + t**2 * (P3[1] - P2[1])
    )
    return math.atan2(dy, dx)


# ─────────────────────────────────────────────────────────────────────────────
#  MARKER
# ─────────────────────────────────────────────────────────────────────────────
class Marker:
    def __init__(self, x, y, kind):
        self.x = x  # centre x  — on the POLE (side of path)
        self.y = y  # centre y  — at FIELD_TOP_Y or FIELD_BOT_Y
        self.kind = kind  # 'normal' | 'last_row' | 'stop'
        self.state = "idle"  # 'idle' | 'detected' | 'triggered' | 'done'

    def colour(self):
        if self.state == "done":
            return C_MK_DONE
        if self.state in ("detected", "triggered"):
            return C_MK_SEEN
        return {"normal": C_MK_NORMAL, "last_row": C_MK_LAST, "stop": C_MK_STOP}[
            self.kind
        ]

    def label(self):
        return {"normal": "ID", "last_row": "LAST", "stop": "STOP"}[self.kind]


# ─────────────────────────────────────────────────────────────────────────────
#  BUILD MARKERS  — placed in the MIDDLE of each path at the headland end
# ─────────────────────────────────────────────────────────────────────────────
def build_markers():
    """
    Markers stand on a pole at the END of each machine path (top or bottom
    headland), centred on the MIDDLE of the path — directly ahead of the
    machine as it drives up/down that path.

    Path traversal order:
      Path 0 UP   → NORMAL marker at TOP of path 0
      Path 1 DOWN → NORMAL marker at BOTTOM of path 1
      Path 2 UP   → NORMAL marker at TOP of path 2
      Path 3 DOWN → NORMAL marker at BOTTOM of path 3
      Path 4 UP   → LAST-ROW marker at TOP of path 4  (final U-turn here)
      Path 5 DOWN → STOP marker at BOTTOM of path 5   (field complete)
    """
    markers = []

    # Top-end markers (paths going UP: 0, 2, 4) — centred on the path
    for i, kind in [(0, "normal"), (2, "normal"), (4, "last_row")]:
        markers.append(Marker(PATH_CX[i], FIELD_TOP_Y, kind))

    # Bottom-end markers (paths going DOWN: 1, 3, 5) — centred on the path
    for i, kind in [(1, "normal"), (3, "normal"), (5, "stop")]:
        markers.append(Marker(PATH_CX[i], FIELD_BOT_Y, kind))

    return markers


# ─────────────────────────────────────────────────────────────────────────────
#  SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
class FieldSimulator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.markers = build_markers()
        self.speed = BASE_SPEED
        self.paused = False

        # Machine position & orientation
        self.mx = float(PATH_CX[0])
        self.my = float(FIELD_BOT_Y - M_H // 2)
        self.angle = -math.pi / 2  # pointing UP

        # Navigation state
        self.path_idx = 0
        self.going_up = True
        self.phase = "driving"
        self.turn_t = 0.0
        self.turn_curves = None
        self.last_row_done = False

        # Stats
        self.turns_done = 0
        self.rows_done = 0
        self.state_name = "DRIVING"
        self.current_row = 1
        self.cam_state = "idle"
        self.marker_seen_name = "-"

        # Timing
        self.start_time = time.time()
        self.done_time = None

    # ── geometry ────────────────────────────────────────────────────────────

    def _camera_cone_pts(self):
        half = math.radians(CAM_HALF_DEG)
        lx = math.cos(self.angle - half) * CAM_RANGE
        ly = math.sin(self.angle - half) * CAM_RANGE
        rx = math.cos(self.angle + half) * CAM_RANGE
        ry = math.sin(self.angle + half) * CAM_RANGE
        tip = (int(self.mx), int(self.my))
        lpt = (int(self.mx + lx), int(self.my + ly))
        rpt = (int(self.mx + rx), int(self.my + ry))
        return np.array([tip, lpt, rpt], dtype=np.int32)

    def _dist_to(self, mk):
        return math.hypot(self.mx - mk.x, self.my - mk.y)

    def _in_cone(self, mk):
        dx, dy = mk.x - self.mx, mk.y - self.my
        dist = math.hypot(dx, dy)
        if dist > CAM_RANGE or dist < 1:
            return False
        angle_to = math.atan2(dy, dx)
        diff = (angle_to - self.angle + math.pi) % (2 * math.pi) - math.pi
        return abs(diff) <= math.radians(CAM_HALF_DEG)

    def _target_marker(self):
        """Find the marker the machine is currently heading toward."""
        target_y = FIELD_TOP_Y if self.going_up else FIELD_BOT_Y
        target_x = PATH_CX[self.path_idx]  # centred on the current path
        for mk in self.markers:
            if (
                abs(mk.x - target_x) < 8
                and abs(mk.y - target_y) < 8
                and mk.state != "done"
            ):
                return mk
        return None

    # ── Bezier turn curve ────────────────────────────────────────────────────
    def _build_turn_curve(self, at_top, start_x, start_y):
        """
        Build a Bezier arc that sweeps around the side-pole turning point.
        The arc goes from current path to the next path, curving through
        the headland area (above top or below bottom field line).
        """
        nxt = self.path_idx + 1
        if nxt >= NUM_PATHS:
            return None
        x0 = float(start_x)
        x3 = float(PATH_CX[nxt])
        ctrl = 95  # headland arc depth

        if at_top:
            y0 = float(start_y)
            P0 = (x0, y0)
            P1 = (x0, y0 - ctrl)
            P2 = (x3, float(FIELD_TOP_Y) - ctrl)
            P3 = (x3, float(FIELD_TOP_Y) + 8)
        else:
            y0 = float(start_y)
            P0 = (x0, y0)
            P1 = (x0, y0 + ctrl)
            P2 = (x3, float(FIELD_BOT_Y) + ctrl)
            P3 = (x3, float(FIELD_BOT_Y) - 8)
        return (P0, P1, P2, P3)

    # ── update ───────────────────────────────────────────────────────────────
    def update(self):
        if self.paused or self.phase in ("stopped", "done"):
            return
        if self.phase == "driving":
            self._update_driving()
        elif self.phase == "turning":
            self._update_turning()

    def _update_driving(self):
        mk = self._target_marker()
        eucl = self._dist_to(mk) if mk else 9999

        # Forward distance to the headland (end of the path) in travel direction.
        # This is how the real machine knows it has reached the row end — it
        # reads the SIDE marker and measures how far ahead the headland is.
        if self.going_up:
            fwd = self.my - FIELD_TOP_Y
        else:
            fwd = FIELD_BOT_Y - self.my

        # The marker is "seen" when it is inside the camera cone and within range.
        seen = (
            mk is not None
            and mk.state != "done"
            and self._in_cone(mk)
            and eucl <= DETECT_DIST
        )

        if seen and fwd <= TRIGGER_DIST:
            mk.state = "triggered"
            self.cam_state = "triggered"
            self.marker_seen_name = f"{mk.label()} (end {fwd:.0f}px)"
            # STOP marker after the last-row turn → end the whole field
            if mk.kind == "stop":
                mk.state = "done"
                self._finish_field()
            else:
                self._start_turn(mk)
            return
        elif seen:
            mk.state = "detected"
            self.cam_state = "detected"
            self.marker_seen_name = f"{mk.label()} (end {fwd:.0f}px)"
        else:
            self.cam_state = "idle"
            self.marker_seen_name = "-"

        # Move — decelerate as the machine nears the headland
        spd = self.speed * (max(0.45, fwd / 180) if fwd < 180 else 1.0)
        if self.going_up:
            self.my -= spd
            self.angle = -math.pi / 2
        else:
            self.my += spd
            self.angle = math.pi / 2

    def _start_turn(self, mk):
        curves = self._build_turn_curve(
            at_top=(self.going_up),
            start_x=self.mx,
            start_y=self.my,
        )
        if curves is None:
            self._finish_field()
            return

        mk.state = "done"
        self.turn_curves = curves
        self.turn_t = 0.0
        self.phase = "turning"
        self.state_name = "TURNING"
        self.turns_done += 1
        if self.turns_done % 2 == 0:
            self.rows_done += 1
        if mk.kind == "last_row":
            self.last_row_done = True

    def _update_turning(self):
        P0, P1, P2, P3 = self.turn_curves
        self.turn_t += 0.013 * (self.speed / BASE_SPEED)
        t = min(self.turn_t, 1.0)

        P = bez(P0, P1, P2, P3, t)
        self.mx, self.my = P
        self.angle = bez_angle(P0, P1, P2, P3, max(0.001, t))

        if t >= 1.0:
            self.path_idx += 1
            self.going_up = not self.going_up
            self.current_row = self.path_idx + 1
            self.phase = "driving"
            self.state_name = "DRIVING"
            self.cam_state = "idle"
            # Snap to exact path centre
            self.mx = PATH_CX[self.path_idx]
        # NOTE: the STOP marker after the last-row turn is handled in
        # _update_driving (when mk.kind == 'stop' it calls _finish_field()).

    def _finish_field(self):
        self.phase = "done"
        self.state_name = "FIELD COMPLETE"
        self.done_time = time.time()

    # ── DRAW ─────────────────────────────────────────────────────────────────
    def draw(self):
        canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        self._draw_field(canvas)
        self._draw_panel(canvas)
        return canvas

    # ─── field drawing ───────────────────────────────────────────────────────
    def _draw_field(self, c):
        # 1. Headland background
        c[:, :FIELD_W] = C_HEADLAND

        # 2. Headland tinted zones (top and bottom margins)
        cv2.rectangle(c, (0, 0), (FIELD_W, MARGIN_T), (168, 168, 150), -1)
        cv2.rectangle(c, (0, WIN_H - MARGIN_B), (FIELD_W, WIN_H), (168, 168, 150), -1)

        # 3. Crop strips (green)
        for i in range(NUM_PATHS + 1):
            x = MARGIN_L + i * STRIP
            cv2.rectangle(c, (x, 0), (x + CROP_W, WIN_H), C_CROP, -1)

        # 4. Machine paths (beige)
        for i in range(NUM_PATHS):
            x = MARGIN_L + CROP_W + i * STRIP
            cv2.rectangle(c, (x, 0), (x + PATH_W, WIN_H), C_PATH, -1)

        # 5. Vertical pole lines — the key visual from the diagram
        #    One pole on each side of every crop row
        for i in range(NUM_PATHS + 1):
            # Left edge of crop row i
            px = MARGIN_L + i * STRIP
            cv2.line(c, (px, 0), (px, WIN_H), C_POLE_LINE, POLE_LINE_W)
            # Right edge of crop row i
            px2 = MARGIN_L + i * STRIP + CROP_W
            cv2.line(c, (px2, 0), (px2, WIN_H), C_POLE_LINE, POLE_LINE_W)

        # 6. Headland boundary lines
        cv2.line(c, (0, MARGIN_T), (FIELD_W, MARGIN_T), (120, 115, 100), 1)
        cv2.line(
            c, (0, WIN_H - MARGIN_B), (FIELD_W, WIN_H - MARGIN_B), (120, 115, 100), 1
        )

        # 7. Dashed centre lines on paths
        for cx in PATH_CX:
            for y in range(MARGIN_T + 12, WIN_H - MARGIN_B, 20):
                cv2.line(c, (cx, y), (cx, y + 9), (175, 165, 140), 1)

        # 8. Bezier arc trace during turning
        if self.phase == "turning" and self.turn_curves:
            P0, P1, P2, P3 = self.turn_curves
            pts = [bez(P0, P1, P2, P3, t / 40.0) for t in range(41)]
            for i in range(len(pts) - 1):
                cv2.line(
                    c,
                    (int(pts[i][0]), int(pts[i][1])),
                    (int(pts[i + 1][0]), int(pts[i + 1][1])),
                    (130, 100, 200),
                    2,
                )

        # 9. Markers (on the SIDE POLES)
        for mk in self.markers:
            self._draw_marker(c, mk)

        # 10. Camera cone
        cone = self._camera_cone_pts()
        col = {
            "idle": C_CAM_IDLE,
            "detected": C_CAM_DETECT,
            "triggered": C_CAM_TRIG,
        }.get(self.cam_state, C_CAM_IDLE)
        ov = c.copy()
        cv2.fillPoly(ov, [cone], col)
        cv2.addWeighted(ov, 0.22, c, 0.78, 0, c)
        cv2.polylines(c, [cone], True, col, 1)

        # 11. Machine
        self._draw_machine(c)

        # 12. Labels
        cv2.putText(
            c,
            "TOP OF FIELD (headland)",
            (FIELD_W // 2 - 100, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (80, 75, 65),
            1,
        )
        cv2.putText(
            c,
            "BOTTOM OF FIELD - machine starts here",
            (FIELD_W // 2 - 145, WIN_H - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (80, 75, 65),
            1,
        )

        # 13. Field-complete banner
        if self.phase == "done":
            pulse = int(40 * abs(math.sin(time.time() * 3)))
            col2 = (0, min(255, 170 + pulse), 0)
            cv2.rectangle(
                c,
                (40, WIN_H // 2 - 50),
                (FIELD_W - 40, WIN_H // 2 + 50),
                (15, 50, 15),
                -1,
            )
            cv2.rectangle(
                c, (40, WIN_H // 2 - 50), (FIELD_W - 40, WIN_H // 2 + 50), col2, 3
            )
            cv2.putText(
                c,
                "FIELD COMPLETE!",
                (FIELD_W // 2 - 165, WIN_H // 2 + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                col2,
                3,
            )

        # Panel divider
        cv2.rectangle(c, (FIELD_W, 0), (FIELD_W + 2, WIN_H), (35, 35, 45), -1)

    def _draw_marker(self, c, mk):
        col = mk.colour()
        cx, cy = int(mk.x), int(mk.y)
        is_top = cy <= WIN_H // 2

        # ── Pole cap block (the top/bottom of the pole in headland) ──
        cap_h = HEADLAND_EXT + MK_HALF + 6
        cap_w = POLE_LINE_W + 4
        if is_top:
            cv2.rectangle(
                c,
                (cx - cap_w // 2, cy - cap_h),
                (cx + cap_w // 2, cy + MK_HALF + 2),
                C_POLE_CAP,
                -1,
            )
        else:
            cv2.rectangle(
                c,
                (cx - cap_w // 2, cy - MK_HALF - 2),
                (cx + cap_w // 2, cy + cap_h),
                C_POLE_CAP,
                -1,
            )

        # ── Marker square (like ArUco / QR code) ──
        cv2.rectangle(
            c, (cx - MK_HALF, cy - MK_HALF), (cx + MK_HALF, cy + MK_HALF), col, -1
        )
        cv2.rectangle(
            c, (cx - MK_HALF, cy - MK_HALF), (cx + MK_HALF, cy + MK_HALF), C_WHITE, 1
        )
        # Inner black pattern (ArUco-style)
        inner = MK_HALF // 2
        cv2.rectangle(
            c, (cx - inner, cy - inner), (cx + inner, cy + inner), (20, 20, 20), -1
        )
        # White cells in inner pattern
        q = inner // 2
        for dx, dy in [(-q, -q), (q, q)]:
            cv2.rectangle(
                c,
                (cx + dx - q // 2, cy + dy - q // 2),
                (cx + dx + q // 2, cy + dy + q // 2),
                C_WHITE,
                -1,
            )

        # ── Label ──
        label = mk.label()
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0][0]
        ty = (
            cy - MK_HALF - HEADLAND_EXT - 4
            if is_top
            else cy + MK_HALF + HEADLAND_EXT + 13
        )
        cv2.putText(
            c, label, (cx - tw // 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1
        )

        # ── Detection ring ──
        if mk.state == "detected":
            cv2.circle(c, (cx, cy), DETECT_DIST, (*C_MK_SEEN,), 1)
        if mk.state == "triggered":
            cv2.circle(c, (cx, cy), TRIGGER_DIST, C_CAM_TRIG, 2)

    def _draw_machine(self, c):
        cx, cy = int(self.mx), int(self.my)
        a = self.angle
        hw, hh = M_W / 2, M_H / 2
        cos_a, sin_a = math.cos(a), math.sin(a)

        # Rotate body corners
        corners = [
            (
                int((-hw) * cos_a - (-hh) * sin_a + cx),
                int((-hw) * sin_a + (-hh) * cos_a + cy),
            ),
            (
                int((hw) * cos_a - (-hh) * sin_a + cx),
                int((hw) * sin_a + (-hh) * cos_a + cy),
            ),
            (
                int((hw) * cos_a - (hh) * sin_a + cx),
                int((hw) * sin_a + (hh) * cos_a + cy),
            ),
            (
                int((-hw) * cos_a - (hh) * sin_a + cx),
                int((-hw) * sin_a + (hh) * cos_a + cy),
            ),
        ]
        pts = np.array(corners, dtype=np.int32)
        cv2.fillPoly(c, [pts], C_MACHINE)
        cv2.polylines(c, [pts], True, C_MACHINE_BD, 2)

        # Direction arrow
        fwx = int(cx + cos_a * (hh - 5))
        fwy = int(cy + sin_a * (hh - 5))
        cv2.arrowedLine(c, (cx, cy), (fwx, fwy), C_MACHINE_BD, 2, tipLength=0.5)

        # Camera (yellow dot on front)
        camx = int(cx + cos_a * hh)
        camy = int(cy + sin_a * hh)
        cv2.circle(c, (camx, camy), 5, (0, 230, 230), -1)
        cv2.circle(c, (camx, camy), 5, C_MACHINE_BD, 1)

        # Inline state badge
        badge_col = {
            "driving": C_GREEN,
            "turning": C_YELLOW,
            "stopped": C_RED,
            "done": C_GREEN,
        }.get(self.phase, C_WHITE)
        cv2.putText(
            c,
            self.phase.upper(),
            (cx + M_W + 4, cy - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            badge_col,
            1,
        )

    # ─── info panel ─────────────────────────────────────────────────────────
    def _draw_panel(self, c):
        px = FIELD_W + 2
        c[:, px:] = C_PANEL
        y = 18

        def title(text, col=C_WHITE, sz=0.66):
            nonlocal y
            cv2.putText(c, text, (px + 14, y), cv2.FONT_HERSHEY_SIMPLEX, sz, col, 1)
            y += int(sz * 30) + 6

        def row(lbl, val, vc=C_WHITE):
            nonlocal y
            cv2.putText(c, lbl, (px + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, C_DIM, 1)
            cv2.putText(
                c, str(val), (px + 168, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, vc, 1
            )
            y += 25

        def div():
            nonlocal y
            cv2.line(c, (px + 10, y), (WIN_W - 10, y), C_DIV, 1)
            y += 9

        # Header
        title("AgriBot Simulator", C_WHITE, 0.68)
        title("Field Navigation Demo", C_DIM, 0.46)
        y += 2
        div()

        # State
        scol = {
            "DRIVING": C_GREEN,
            "TURNING": C_YELLOW,
            "STOPPED": C_RED,
            "FIELD COMPLETE": (0, 230, 0),
        }.get(self.state_name, C_WHITE)
        title(self.state_name, scol, 0.78)
        div()

        # Navigation
        title("Navigation", C_DIM, 0.48)
        dir_label = "UP (towards top)" if self.going_up else "DOWN (towards bottom)"
        row("Current path", f"{self.path_idx + 1} / {NUM_PATHS}")
        row("Direction", dir_label)
        row("Turns done", self.turns_done, C_YELLOW)
        row("Rows completed", self.rows_done, C_GREEN)
        row(
            "Last row reached",
            "YES" if self.last_row_done else "No",
            (0, 200, 200) if self.last_row_done else C_DIM,
        )
        div()

        # Camera
        title("Camera", C_DIM, 0.48)
        ccol = {"idle": C_DIM, "detected": C_GREEN, "triggered": C_RED}.get(
            self.cam_state, C_DIM
        )
        row("Camera state", self.cam_state.upper(), ccol)
        row("Marker in view", self.marker_seen_name, ccol)
        div()

        # Marker info — show each marker and its current state
        title("Markers (path ends)", C_DIM, 0.48)
        for mk in self.markers:
            pos_lbl = "TOP" if mk.y < WIN_H // 2 else "BOT"
            state_c = {
                "idle": C_DIM,
                "detected": C_GREEN,
                "triggered": C_RED,
                "done": (80, 80, 80),
            }.get(mk.state, C_DIM)
            row(
                f"  {mk.label()} {pos_lbl} P{self.markers.index(mk)+1}",
                mk.state.upper(),
                state_c,
            )
        div()

        # Legend
        title("Legend", C_DIM, 0.48)

        def legend(col, lbl, desc):
            nonlocal y
            cv2.rectangle(c, (px + 14, y - 9), (px + 30, y + 5), col, -1)
            cv2.rectangle(c, (px + 14, y - 9), (px + 30, y + 5), C_WHITE, 1)
            cv2.putText(
                c,
                f"{lbl} - {desc}",
                (px + 36, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                C_DIM,
                1,
            )
            y += 21

        legend(C_MK_NORMAL, "NORMAL", "U-turn, next row")
        legend(C_MK_LAST, "LAST", "Final U-turn")
        legend(C_MK_STOP, "STOP", "Field complete")
        legend(C_MK_SEEN, "ACTIVE", "Detected / triggered")
        div()

        # Time & speed
        elapsed = time.time() - self.start_time
        mm, ss = divmod(int(elapsed), 60)
        title("Session", C_DIM, 0.48)
        row("Time", f"{mm:02d}:{ss:02d}")
        row("Speed", f"{self.speed:.1f}x")
        div()

        # Controls
        title("Controls", C_DIM, 0.48)
        for key, desc in [
            ("SPACE", "Pause / Resume"),
            ("R", "Restart"),
            ("+ / -", "Speed up/down"),
            ("Q", "Quit"),
        ]:
            cv2.putText(
                c,
                f"[{key}]",
                (px + 14, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (180, 180, 60),
                1,
            )
            cv2.putText(c, desc, (px + 68, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, C_DIM, 1)
            y += 21

        # Field-complete box
        if self.phase == "done":
            pulse = int(30 * abs(math.sin(time.time() * 3)))
            gc = (0, min(255, 190 + pulse), 0)
            cv2.rectangle(
                c, (px + 6, WIN_H - 95), (WIN_W - 6, WIN_H - 8), (12, 40, 12), -1
            )
            cv2.rectangle(c, (px + 6, WIN_H - 95), (WIN_W - 6, WIN_H - 8), gc, 2)
            cv2.putText(
                c,
                "ALL ROWS DONE!",
                (px + 18, WIN_H - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                gc,
                2,
            )
            if self.done_time:
                tot = int(self.done_time - self.start_time)
                mm2, ss2 = divmod(tot, 60)
                cv2.putText(
                    c,
                    f"Time {mm2:02d}:{ss2:02d}  |  Turns {self.turns_done}",
                    (px + 18, WIN_H - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    C_DIM,
                    1,
                )


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 58)
    print("  AgriBot Field Simulator")
    print("  Markers on SIDE POLES — matching real field layout")
    print("  SPACE=pause  R=restart  +/-=speed  Q=quit")
    print("=" * 58)

    sim = FieldSimulator()
    win = "AgriBot Field Simulator — Q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, WIN_W, WIN_H)

    frame_time = 1.0 / 60.0

    while True:
        t0 = time.time()
        sim.update()
        cv2.imshow(win, sim.draw())

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord(" "), ord("p"), ord("P")):
            sim.paused = not sim.paused
            print("⏸ Paused" if sim.paused else "▶ Resumed")
        elif key in (ord("r"), ord("R")):
            sim.reset()
            print("↺ Restarted")
        elif key in (ord("+"), ord("=")):
            sim.speed = min(sim.speed + 0.5, 12.0)
            print(f"Speed: {sim.speed:.1f}x")
        elif key in (ord("-"), ord("_")):
            sim.speed = max(sim.speed - 0.5, 0.5)
            print(f"Speed: {sim.speed:.1f}x")

        sleep = frame_time - (time.time() - t0)
        if sleep > 0:
            time.sleep(sleep)

    cv2.destroyAllWindows()
    print("Simulation ended.")


if __name__ == "__main__":
    main()
