#!/usr/bin/env python3
"""
simulation/field_simulator.py
==============================
Top-down 2D visual simulation of the AgriBot field operation.

Demonstrates the full navigation cycle:
  - Machine drives row by row through crop paths
  - Camera cone detects markers at each row end
  - NORMAL markers  → U-turn, continue to next row
  - LAST-ROW marker → final U-turn, then watch for STOP
  - STOP marker     → field complete, machine halts

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
import sys
import os

# ─────────────────────────────────────────────────────────────────────────────
#  LAYOUT CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
WIN_W, WIN_H   = 1280, 720
PANEL_W        = 340       # right info panel
FIELD_W        = WIN_W - PANEL_W   # 940 px field canvas

NUM_PATHS      = 6         # number of row paths the machine traverses
CROP_W         = 45        # green crop strip width (px)
PATH_W         = 58        # machine-path width (px)
STRIP          = CROP_W + PATH_W   # total per-row unit

MARGIN_L       = 55        # left headland margin
MARGIN_T       = 90        # top headland
MARGIN_B       = 90        # bottom headland

# Derived: path-centre x positions
PATH_CX = [MARGIN_L + CROP_W + PATH_W // 2 + i * STRIP for i in range(NUM_PATHS)]
FIELD_TOP_Y    = MARGIN_T
FIELD_BOT_Y    = WIN_H - MARGIN_B

# Marker geometry
MK_HALF        = 14        # half-size of marker square
POLE_W, POLE_H = 6, 30    # pole drawn above marker

# Detection thresholds (px)
DETECT_DIST    = 130       # camera first sees marker
TRIGGER_DIST   = 60        # turn fires (≈ 1.5 m)

# Machine geometry
M_W, M_H      = 28, 46    # machine width / height
CAM_HALF_DEG  = 35        # camera half-angle
CAM_RANGE     = 120       # camera cone length (px)

# Animation
BASE_SPEED     = 2.2       # pixels/frame when driving

# ─────────────────────────────────────────────────────────────────────────────
#  COLOURS  (BGR)
# ─────────────────────────────────────────────────────────────────────────────
C_HEADLAND   = (185, 185, 165)
C_CROP       = ( 28,  90,  28)
C_PATH       = (205, 195, 170)
C_POLE       = ( 60,  80, 100)
C_MK_NORMAL  = ( 30, 160, 255)   # orange
C_MK_LAST    = (255, 100,  30)   # blue-ish
C_MK_STOP    = ( 30,  30, 210)   # red
C_MK_SEEN    = ( 60, 220,  60)   # green flash when detected
C_MK_DONE    = (120, 120, 120)   # grey after passed
C_MACHINE    = (230, 230, 250)
C_MACHINE_BD = (100, 100, 210)
C_CAM        = (200, 200,  50)   # camera cone (idle)
C_CAM_DETECT = ( 40, 230,  40)   # camera cone (marker detected)
C_CAM_TRIG   = ( 30,  30, 220)   # camera cone (turn triggered)
C_PANEL      = ( 22,  22,  28)
C_PANEL_LINE = ( 50,  50,  60)
C_WHITE      = (255, 255, 255)
C_DIM        = (130, 130, 140)
C_GREEN      = ( 50, 210,  50)
C_YELLOW     = ( 40, 210, 210)
C_RED        = ( 60,  60, 210)
C_BLUE       = (220, 160,  30)
C_ORANGE     = ( 30, 140, 240)

# ─────────────────────────────────────────────────────────────────────────────
#  BEZIER HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def bez(P0, P1, P2, P3, t):
    """Cubic Bezier position."""
    mt = 1.0 - t
    return (
        mt**3*P0[0] + 3*mt**2*t*P1[0] + 3*mt*t**2*P2[0] + t**3*P3[0],
        mt**3*P0[1] + 3*mt**2*t*P1[1] + 3*mt*t**2*P2[1] + t**3*P3[1],
    )

def bez_angle(P0, P1, P2, P3, t):
    """Tangent angle of cubic Bezier at t."""
    mt = 1.0 - t
    dx = 3*(mt**2*(P1[0]-P0[0]) + 2*mt*t*(P2[0]-P1[0]) + t**2*(P3[0]-P2[0]))
    dy = 3*(mt**2*(P1[1]-P0[1]) + 2*mt*t*(P2[1]-P1[1]) + t**2*(P3[1]-P2[1]))
    return math.atan2(dy, dx)


# ─────────────────────────────────────────────────────────────────────────────
#  MARKER  dataclass-like structure
# ─────────────────────────────────────────────────────────────────────────────
class Marker:
    def __init__(self, x, y, kind):
        self.x     = x          # centre x
        self.y     = y          # centre y
        self.kind  = kind       # 'normal' | 'last_row' | 'stop'
        self.state = 'idle'     # 'idle' | 'detected' | 'triggered' | 'done'

    def colour(self):
        if self.state == 'done':     return C_MK_DONE
        if self.state in ('detected', 'triggered'): return C_MK_SEEN
        return {
            'normal':   C_MK_NORMAL,
            'last_row': C_MK_LAST,
            'stop':     C_MK_STOP,
        }[self.kind]

    def label(self):
        return {'normal': 'ID', 'last_row': 'LAST', 'stop': 'STOP'}[self.kind]


# ─────────────────────────────────────────────────────────────────────────────
#  BUILD MARKERS
# ─────────────────────────────────────────────────────────────────────────────
def build_markers():
    """
    Returns list of Marker objects for the field.

    Machine traversal: path 0 UP, path 1 DOWN, ... path 5 DOWN

    Markers:
      Top of paths 0,2   → NORMAL
      Top of path 4       → LAST_ROW  (final turn)
      Bottom of paths 1,3 → NORMAL
      Bottom of path 5    → STOP      (field complete)
    """
    markers = []
    top_paths    = [0, 2]       # normal top markers
    last_top     = 4             # last-row top marker
    bot_paths    = [1, 3]       # normal bottom markers
    stop_bot     = 5             # stop bottom marker

    for i in top_paths:
        markers.append(Marker(PATH_CX[i], FIELD_TOP_Y, 'normal'))
    markers.append(Marker(PATH_CX[last_top], FIELD_TOP_Y, 'last_row'))

    for i in bot_paths:
        markers.append(Marker(PATH_CX[i], FIELD_BOT_Y, 'normal'))
    markers.append(Marker(PATH_CX[stop_bot], FIELD_BOT_Y, 'stop'))

    return markers


# ─────────────────────────────────────────────────────────────────────────────
#  SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
class FieldSimulator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.markers      = build_markers()
        self.speed        = BASE_SPEED
        self.paused       = False

        # Machine state
        self.mx      = float(PATH_CX[0])              # x centre
        self.my      = float(FIELD_BOT_Y - M_H // 2)  # y centre — start at bottom
        self.angle   = -math.pi / 2   # pointing UP (-π/2)

        # Navigation
        self.path_idx      = 0        # which path the machine is on
        self.going_up      = True     # direction of travel
        self.phase         = 'driving'  # 'driving' | 'turning' | 'stopped' | 'done'
        self.turn_t        = 0.0      # Bezier parameter 0→1
        self.turn_curves   = None     # (P0,P1,P2,P3) for current turn
        self.last_row_done = False    # True after LAST_ROW turn completed

        # Stats
        self.turns_done  = 0
        self.rows_done   = 0
        self.state_name  = 'DRIVING'
        self.current_row = 1
        self.cam_state   = 'idle'    # 'idle'|'detected'|'triggered'
        self.marker_seen_name = '—'

        # Timing
        self.start_time  = time.time()
        self.done_time   = None

    # ── geometry helpers ────────────────────────────────────────────────────

    def _camera_cone_pts(self):
        """Return the 3 vertices of the camera cone polygon."""
        fwd_x = math.cos(self.angle)
        fwd_y = math.sin(self.angle)
        half   = math.radians(CAM_HALF_DEG)
        left_x = math.cos(self.angle - half) * CAM_RANGE
        left_y = math.sin(self.angle - half) * CAM_RANGE
        right_x = math.cos(self.angle + half) * CAM_RANGE
        right_y = math.sin(self.angle + half) * CAM_RANGE
        tip = (int(self.mx), int(self.my))
        lpt = (int(self.mx + left_x),  int(self.my + left_y))
        rpt = (int(self.mx + right_x), int(self.my + right_y))
        return np.array([tip, lpt, rpt], dtype=np.int32)

    def _dist_to_marker(self, mk):
        return math.hypot(self.mx - mk.x, self.my - mk.y)

    def _marker_in_cone(self, mk):
        """True if marker is within the camera cone."""
        dx = mk.x - self.mx
        dy = mk.y - self.my
        dist = math.hypot(dx, dy)
        if dist > CAM_RANGE or dist < 1:
            return False
        angle_to = math.atan2(dy, dx)
        diff = (angle_to - self.angle + math.pi) % (2 * math.pi) - math.pi
        return abs(diff) <= math.radians(CAM_HALF_DEG)

    # ── find the marker for the current approach ────────────────────────────
    def _target_marker(self):
        """Return the marker the machine is currently heading toward."""
        if self.going_up:
            target_y = FIELD_TOP_Y
            target_x = PATH_CX[self.path_idx]
        else:
            target_y = FIELD_BOT_Y
            target_x = PATH_CX[self.path_idx]

        for mk in self.markers:
            if (abs(mk.x - target_x) < 5 and abs(mk.y - target_y) < 5
                    and mk.state != 'done'):
                return mk
        return None

    # ── build Bezier control points for a turn ──────────────────────────────
    def _build_turn_curve(self, at_top):
        next_path = self.path_idx + 1
        if next_path >= NUM_PATHS:
            return None
        x0 = PATH_CX[self.path_idx]
        x3 = PATH_CX[next_path]
        ctrl = 85  # control point offset
        if at_top:
            y0 = FIELD_TOP_Y
            y3 = FIELD_TOP_Y
            P0 = (x0, y0); P1 = (x0, y0 - ctrl)
            P2 = (x3, y3 - ctrl); P3 = (x3, y3)
        else:
            y0 = FIELD_BOT_Y
            y3 = FIELD_BOT_Y
            P0 = (x0, y0); P1 = (x0, y0 + ctrl)
            P2 = (x3, y3 + ctrl); P3 = (x3, y3)
        return (P0, P1, P2, P3)

    # ── update one frame ────────────────────────────────────────────────────
    def update(self):
        if self.paused or self.phase in ('stopped', 'done'):
            return

        if self.phase == 'driving':
            self._update_driving()
        elif self.phase == 'turning':
            self._update_turning()

    def _update_driving(self):
        mk = self._target_marker()
        dist = self._dist_to_marker(mk) if mk else 9999

        # Camera cone detection
        if mk and self._marker_in_cone(mk) and mk.state not in ('done',):
            if dist <= TRIGGER_DIST:
                if mk.state != 'triggered':
                    mk.state = 'triggered'
                self.cam_state = 'triggered'
                self.marker_seen_name = f"{mk.label()} at {dist:.0f}px"
                self._start_turn(mk)
                return
            elif dist <= DETECT_DIST:
                mk.state = 'detected'
                self.cam_state = 'detected'
                self.marker_seen_name = f"{mk.label()} at {dist:.0f}px"
            else:
                self.cam_state = 'idle'
                self.marker_seen_name = '—'
        else:
            self.cam_state = 'idle'
            self.marker_seen_name = '—'

        # Move
        spd = self.speed
        if mk and dist < 200:
            spd = self.speed * max(0.5, dist / 200)   # decelerate near marker
        if self.going_up:
            self.my -= spd
            self.angle = -math.pi / 2
        else:
            self.my += spd
            self.angle = math.pi / 2

    def _start_turn(self, mk):
        curves = self._build_turn_curve(at_top=self.going_up)
        if curves is None:
            self.phase = 'done'
            self.state_name = 'FIELD COMPLETE'
            self.done_time = time.time()
            return

        mk.state = 'done'
        self.turn_curves = curves
        self.turn_t = 0.0
        self.phase = 'turning'
        self.state_name = 'TURNING'
        self.turns_done += 1
        if self.turns_done % 2 == 0:
            self.rows_done += 1

        # Check if this was the LAST_ROW turn
        if mk.kind == 'last_row':
            self.last_row_done = True

    def _update_turning(self):
        P0, P1, P2, P3 = self.turn_curves
        self.turn_t += 0.012 * (self.speed / BASE_SPEED)
        if self.turn_t >= 1.0:
            self.turn_t = 1.0
            P = bez(P0, P1, P2, P3, 1.0)
            self.mx, self.my = P
            self.path_idx += 1
            self.going_up = not self.going_up
            self.current_row = self.path_idx + 1
            self.phase = 'driving'
            self.state_name = 'DRIVING'
            self.cam_state = 'idle'
        else:
            P = bez(P0, P1, P2, P3, self.turn_t)
            self.mx, self.my = P
            self.angle = bez_angle(P0, P1, P2, P3, max(0.001, self.turn_t))

        # Check for STOP marker after last-row turn
        if self.last_row_done and self.phase == 'driving':
            mk = self._target_marker()
            if mk and mk.kind == 'stop':
                dist = self._dist_to_marker(mk)
                if self._marker_in_cone(mk) and dist <= TRIGGER_DIST:
                    mk.state = 'done'
                    self.phase = 'done'
                    self.state_name = 'FIELD COMPLETE'
                    self.done_time = time.time()

    # ── DRAW ────────────────────────────────────────────────────────────────
    def draw(self):
        canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        self._draw_field(canvas)
        self._draw_panel(canvas)
        return canvas

    # ─── field canvas ───────────────────────────────────────────────────────
    def _draw_field(self, c):
        # Background (headland)
        c[:, :FIELD_W] = C_HEADLAND

        # Headland top/bottom tinted
        cv2.rectangle(c, (0, 0), (FIELD_W, MARGIN_T), (170, 170, 155), -1)
        cv2.rectangle(c, (0, WIN_H - MARGIN_B), (FIELD_W, WIN_H), (170, 170, 155), -1)

        # Crop strips
        for i in range(NUM_PATHS + 1):
            x = MARGIN_L + i * STRIP
            cv2.rectangle(c, (x, 0), (x + CROP_W, WIN_H), C_CROP, -1)

        # Paths (beige strips between crops)
        for i in range(NUM_PATHS):
            x = MARGIN_L + CROP_W + i * STRIP
            cv2.rectangle(c, (x, 0), (x + PATH_W, WIN_H), C_PATH, -1)

        # Field boundary lines
        cv2.line(c, (0, MARGIN_T), (FIELD_W, MARGIN_T), (130, 125, 110), 1)
        cv2.line(c, (0, WIN_H - MARGIN_B), (FIELD_W, WIN_H - MARGIN_B), (130, 125, 110), 1)

        # Path centre dashed lines
        for cx in PATH_CX:
            for y in range(MARGIN_T + 10, WIN_H - MARGIN_B, 18):
                cv2.line(c, (cx, y), (cx, y + 8), (180, 170, 145), 1)

        # Draw Bezier trace (ghost path) while turning
        if self.phase == 'turning' and self.turn_curves:
            P0, P1, P2, P3 = self.turn_curves
            pts = [bez(P0, P1, P2, P3, t / 40.0) for t in range(41)]
            for i in range(len(pts) - 1):
                cv2.line(c,
                    (int(pts[i][0]),   int(pts[i][1])),
                    (int(pts[i+1][0]), int(pts[i+1][1])),
                    (100, 100, 180), 2)

        # Markers
        for mk in self.markers:
            self._draw_marker(c, mk)

        # Camera cone
        cone_pts = self._camera_cone_pts()
        overlay = c.copy()
        cone_col = {
            'idle':     C_CAM,
            'detected': C_CAM_DETECT,
            'triggered': C_CAM_TRIG,
        }.get(self.cam_state, C_CAM)
        cv2.fillPoly(overlay, [cone_pts], cone_col)
        cv2.addWeighted(overlay, 0.25, c, 0.75, 0, c)
        cv2.polylines(c, [cone_pts], True, cone_col, 1)

        # Machine body
        self._draw_machine(c)

        # Field label
        cv2.putText(c, "TOP OF FIELD", (FIELD_W // 2 - 75, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 70), 1)
        cv2.putText(c, "BOTTOM OF FIELD (START)", (FIELD_W // 2 - 115, WIN_H - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 70), 1)

        # Field-complete banner
        if self.phase == 'done':
            pulse = int(40 * abs(math.sin(time.time() * 3)))
            col   = (0, min(255, 180 + pulse), 0)
            cv2.rectangle(c, (50, WIN_H // 2 - 45), (FIELD_W - 50, WIN_H // 2 + 45),
                           (20, 60, 20), -1)
            cv2.rectangle(c, (50, WIN_H // 2 - 45), (FIELD_W - 50, WIN_H // 2 + 45),
                           col, 3)
            cv2.putText(c, "FIELD COMPLETE!", (FIELD_W // 2 - 155, WIN_H // 2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)

        # Divider
        cv2.rectangle(c, (FIELD_W, 0), (FIELD_W + 2, WIN_H), (40, 40, 50), -1)

    def _draw_marker(self, c, mk):
        col   = mk.colour()
        cx, cy = int(mk.x), int(mk.y)
        is_top = (cy < WIN_H // 2)

        # Pole
        if is_top:
            pole_top = (cx - POLE_W // 2, cy - POLE_H - MK_HALF)
            pole_bot = (cx + POLE_W // 2, cy - MK_HALF)
        else:
            pole_top = (cx - POLE_W // 2, cy + MK_HALF)
            pole_bot = (cx + POLE_W // 2, cy + MK_HALF + POLE_H)
        cv2.rectangle(c, pole_top, pole_bot, C_POLE, -1)

        # Marker square
        cv2.rectangle(c,
            (cx - MK_HALF, cy - MK_HALF),
            (cx + MK_HALF, cy + MK_HALF),
            col, -1)
        cv2.rectangle(c,
            (cx - MK_HALF, cy - MK_HALF),
            (cx + MK_HALF, cy + MK_HALF),
            (255, 255, 255), 1)

        # Inner pattern (like ArUco)
        inner = MK_HALF // 2
        cv2.rectangle(c,
            (cx - inner, cy - inner),
            (cx + inner, cy + inner),
            (30, 30, 30), -1)

        # Label above/below
        label = mk.label()
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
        ty = cy - MK_HALF - POLE_H - 6 if is_top else cy + MK_HALF + POLE_H + 14
        cv2.putText(c, label, (cx - tw // 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

        # Detection ring
        if mk.state == 'detected':
            cv2.circle(c, (cx, cy), DETECT_DIST, C_MK_SEEN, 1)
        if mk.state == 'triggered':
            cv2.circle(c, (cx, cy), TRIGGER_DIST, C_CAM_TRIG, 2)

    def _draw_machine(self, c):
        cx, cy = int(self.mx), int(self.my)
        a = self.angle

        # Rotate corners
        hw, hh = M_W / 2, M_H / 2
        corners_local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        cos_a, sin_a = math.cos(a), math.sin(a)
        corners = []
        for lx, ly in corners_local:
            rx = lx * cos_a - ly * sin_a + cx
            ry = lx * sin_a + ly * cos_a + cy
            corners.append((int(rx), int(ry)))

        pts = np.array(corners, dtype=np.int32)
        cv2.fillPoly(c, [pts], C_MACHINE)
        cv2.polylines(c, [pts], True, C_MACHINE_BD, 2)

        # Arrow indicating direction
        fwd_x = int(cx + math.cos(a) * (hh - 4))
        fwd_y = int(cy + math.sin(a) * (hh - 4))
        cv2.arrowedLine(c, (cx, cy), (fwd_x, fwd_y), C_MACHINE_BD, 2, tipLength=0.5)

        # Camera mount (small yellow dot)
        cam_x = int(cx + math.cos(a) * (hh - 2))
        cam_y = int(cy + math.sin(a) * (hh - 2))
        cv2.circle(c, (cam_x, cam_y), 4, (0, 220, 220), -1)

        # State label near machine
        label = {'driving': 'DRIVING', 'turning': 'TURNING',
                 'stopped': 'STOPPED', 'done': 'DONE'}.get(self.phase, '')
        col   = {'driving': C_GREEN, 'turning': C_YELLOW,
                 'stopped': C_RED, 'done': C_GREEN}.get(self.phase, C_WHITE)
        cv2.putText(c, label, (cx + M_W, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

    # ─── info panel ─────────────────────────────────────────────────────────
    def _draw_panel(self, c):
        px = FIELD_W + 2
        # Background
        c[:, px:] = C_PANEL

        y = 20

        def title(text, colour=C_WHITE, size=0.65):
            nonlocal y
            cv2.putText(c, text, (px + 15, y), cv2.FONT_HERSHEY_SIMPLEX, size, colour, 1)
            y += int(size * 30) + 6

        def row(label, val, val_col=C_WHITE):
            nonlocal y
            cv2.putText(c, label, (px + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_DIM, 1)
            cv2.putText(c, str(val), (px + 175, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, val_col, 1)
            y += 26

        def divider():
            nonlocal y
            cv2.line(c, (px + 10, y), (WIN_W - 10, y), C_PANEL_LINE, 1)
            y += 10

        # ── Header ──────────────────────────────────────────────
        title("AgriBot Simulator", C_WHITE, 0.72)
        title("Field Navigation Demo", C_DIM, 0.48)
        y += 4
        divider()

        # ── Machine state ────────────────────────────────────────
        state_col = {
            'DRIVING':        C_GREEN,
            'TURNING':        C_YELLOW,
            'STOPPED':        C_RED,
            'FIELD COMPLETE': (0, 220, 0),
        }.get(self.state_name, C_WHITE)
        title(self.state_name, state_col, 0.80)
        divider()

        # ── Navigation stats ─────────────────────────────────────
        title("Navigation", C_DIM, 0.50)
        row("Current Path",    f"{self.path_idx + 1} of {NUM_PATHS}", C_WHITE)
        row("Direction",       "UP" if self.going_up else "DOWN", C_WHITE)
        row("Turns completed", self.turns_done, C_YELLOW)
        row("Rows completed",  self.rows_done,  C_GREEN)
        row("Last-row reached", "YES" if self.last_row_done else "No",
            (0, 200, 200) if self.last_row_done else C_DIM)
        divider()

        # ── Camera ───────────────────────────────────────────────
        title("Camera", C_DIM, 0.50)
        cam_col = {'idle': C_DIM, 'detected': C_GREEN, 'triggered': C_RED}.get(self.cam_state, C_DIM)
        row("Camera state",    self.cam_state.upper(), cam_col)
        row("Marker in view",  self.marker_seen_name,  cam_col)
        divider()

        # ── Marker legend ─────────────────────────────────────────
        title("Marker Legend", C_DIM, 0.50)

        def legend(col, label, desc):
            nonlocal y
            cv2.rectangle(c, (px + 15, y - 10), (px + 33, y + 5), col, -1)
            cv2.rectangle(c, (px + 15, y - 10), (px + 33, y + 5), C_WHITE, 1)
            cv2.putText(c, f"{label} — {desc}", (px + 40, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1)
            y += 22

        legend(C_MK_NORMAL, "NORMAL", "U-turn, next row")
        legend(C_MK_LAST,   "LAST",   "Final U-turn")
        legend(C_MK_STOP,   "STOP",   "Field complete")
        legend(C_MK_SEEN,   "ACTIVE", "Detected / triggered")
        divider()

        # ── Uptime ────────────────────────────────────────────────
        elapsed = time.time() - self.start_time
        mm, ss  = divmod(int(elapsed), 60)
        title("Session", C_DIM, 0.50)
        row("Elapsed",  f"{mm:02d}:{ss:02d}", C_WHITE)
        row("Speed",    f"{self.speed:.1f}x",  C_WHITE)
        divider()

        # ── Controls ─────────────────────────────────────────────
        title("Controls", C_DIM, 0.50)

        def ctrl(key, desc):
            nonlocal y
            cv2.putText(c, f"[{key}]", (px + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (180, 180, 60), 1)
            cv2.putText(c, desc, (px + 60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, C_DIM, 1)
            y += 22

        ctrl("SPACE", "Pause / Resume")
        ctrl("R",     "Restart")
        ctrl("+ / -", "Speed up / down")
        ctrl("Q",     "Quit")

        # ── Field-complete celebration ────────────────────────────
        if self.phase == 'done':
            pulse = int(30 * abs(math.sin(time.time() * 3)))
            col = (0, min(255, 200 + pulse), 0)
            cv2.rectangle(c, (px + 8, WIN_H - 90), (WIN_W - 8, WIN_H - 10), (15, 40, 15), -1)
            cv2.rectangle(c, (px + 8, WIN_H - 90), (WIN_W - 8, WIN_H - 10), col, 2)
            cv2.putText(c, "ALL ROWS DONE!", (px + 22, WIN_H - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
            if self.done_time:
                total = int(self.done_time - self.start_time)
                mm2, ss2 = divmod(total, 60)
                cv2.putText(c, f"Time: {mm2:02d}:{ss2:02d}  Turns: {self.turns_done}",
                            (px + 22, WIN_H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_DIM, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  AgriBot Field Simulator")
    print("  Controls: SPACE=pause  R=restart  +/-=speed  Q=quit")
    print("=" * 55)

    sim = FieldSimulator()
    win = "AgriBot Field Simulator — Press SPACE to pause, Q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, WIN_W, WIN_H)

    target_fps = 60
    frame_time = 1.0 / target_fps

    while True:
        t0 = time.time()

        sim.update()
        frame = sim.draw()
        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord(' '), ord('p'), ord('P')):
            sim.paused = not sim.paused
            if sim.paused:
                print("⏸  Paused")
            else:
                print("▶  Resumed")
        elif key in (ord('r'), ord('R')):
            sim.reset()
            print("↺  Restarted")
        elif key in (ord('+'), ord('=')):
            sim.speed = min(sim.speed + 0.5, 10.0)
            print(f"Speed: {sim.speed:.1f}x")
        elif key in (ord('-'), ord('_')):
            sim.speed = max(sim.speed - 0.5, 0.5)
            print(f"Speed: {sim.speed:.1f}x")

        elapsed = time.time() - t0
        sleep   = frame_time - elapsed
        if sleep > 0:
            time.sleep(sleep)

    cv2.destroyAllWindows()
    print("Simulation ended.")


if __name__ == "__main__":
    main()
