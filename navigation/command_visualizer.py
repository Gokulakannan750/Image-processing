"""
navigation/command_visualizer.py
================================
Draws hardware commands and state over the video feed for debugging.
"""
import cv2
import numpy as np
from navigation.vehicle_state import State
from config.config_manager import config_manager

from collections import deque
from typing import List, Tuple, Optional
from detectors.base_detector import ObstacleDetection


class CommandVisualizer:
    def __init__(self):
        self.enabled = config_manager.get("debug.visualize_commands", True)
        self.draw_history = config_manager.get("debug.draw_path_history", True)
        self.history_maxlen = config_manager.get("debug.history_maxlen", 30)
        self.path_history: deque = deque(maxlen=self.history_maxlen)

    def draw(self,
             frame: np.ndarray,
             state: State,
             steering_correction: float,
             target_center: Tuple[Optional[float], Optional[float]] = (None, None),
             obstacles: Optional[List[ObstacleDetection]] = None) -> None:

        if not self.enabled or frame is None:
            return

        h, w = frame.shape[:2]
        center_line_x = w // 2

        # 0. Obstacle warning banner (highest visual priority)
        critical_obstacles = [o for o in (obstacles or []) if o.is_critical]
        if critical_obstacles:
            labels = ", ".join(o.label for o in critical_obstacles)
            banner = f"!! OBSTACLE DETECTED: {labels.upper()} !!"
            (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            bx = (w - bw) // 2
            cv2.rectangle(frame, (bx - 10, 8), (bx + bw + 10, bh + 20), (0, 0, 200), -1)
            cv2.putText(frame, banner, (bx, bh + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # 1. Draw State Label
        color = (0, 255, 0)
        if state == State.RECOVERING:
            color = (0, 165, 255)
        elif state == State.STOPPED:
            color = (0, 0, 255)

        cv2.putText(frame, f"STATE: {state.name}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # 2. Draw Dead-Zone & Alignment Lines
        dead_zone_w = int(w * 0.1)
        cv2.line(frame, (center_line_x - dead_zone_w, 0), (center_line_x - dead_zone_w, h), (100, 100, 100), 1)
        cv2.line(frame, (center_line_x + dead_zone_w, 0), (center_line_x + dead_zone_w, h), (100, 100, 100), 1)
        cv2.line(frame, (center_line_x, 0), (center_line_x, h), (200, 200, 200), 1, cv2.LINE_AA)

        # 3. Path History
        tx, ty = target_center
        if tx is not None and ty is not None:
            self.path_history.append((int(tx), int(ty)))

        if self.draw_history and len(self.path_history) > 1:
            points = list(self.path_history)
            for i in range(len(points) - 1):
                thickness = int(1 + (i / len(points)) * 4)
                cv2.line(frame, points[i], points[i + 1], (0, 255, 255), thickness)

        # 4. Draw Steering Vector
        if state == State.DRIVING:
            center_y = h - 50
            arrow_x = int(center_line_x - (steering_correction * 1.5))
            cv2.arrowedLine(frame, (center_line_x, center_y), (arrow_x, center_y),
                            (255, 0, 0), 4, tipLength=0.2)
            if tx is not None:
                error_color = (0, 255, 0) if abs(tx - center_line_x) < dead_zone_w else (0, 0, 255)
                cv2.line(frame, (int(center_line_x), int(ty)), (int(tx), int(ty)), error_color, 2)
