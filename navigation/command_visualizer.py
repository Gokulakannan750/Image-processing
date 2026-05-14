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
             target_center: Tuple[Optional[float], Optional[float]] = (None, None)) -> None:
             
        if not self.enabled or frame is None:
            return
            
        h, w = frame.shape[:2]
        center_line_x = w // 2
            
        # 1. Draw State Label
        color = (0, 255, 0)
        if state == State.RECOVERING:
            color = (0, 165, 255) # Orange
        elif state == State.STOPPED:
            color = (0, 0, 255)   # Red
            
        cv2.putText(frame, f"STATE: {state.name}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    
        # 2. Draw Dead-Zone & Alignment Lines
        dead_zone_w = int(w * 0.1) # 10% dead zone
        cv2.line(frame, (center_line_x - dead_zone_w, 0), (center_line_x - dead_zone_w, h), (100, 100, 100), 1)
        cv2.line(frame, (center_line_x + dead_zone_w, 0), (center_line_x + dead_zone_w, h), (100, 100, 100), 1)
        cv2.line(frame, (center_line_x, 0), (center_line_x, h), (200, 200, 200), 1, cv2.LINE_AA)

        # 3. Path History
        tx, ty = target_center
        if tx is not None and ty is not None:
            self.path_history.append((int(tx), int(ty)))
            
        if self.draw_history and len(self.path_history) > 1:
            points = list(self.path_history)
            for i in range(len(points)-1):
                thickness = int(1 + (i / len(points)) * 4)
                cv2.line(frame, points[i], points[i+1], (0, 255, 255), thickness)

        # 4. Draw Steering Vector
        if state == State.DRIVING:
            center_y = h - 50
            # Scale the steering correction
            arrow_x = int(center_line_x - (steering_correction * 1.5))
            
            # Draw Vector Line
            cv2.arrowedLine(frame, (center_line_x, center_y), (arrow_x, center_y), 
                            (255, 0, 0), 4, tipLength=0.2)
            
            # Draw Alignment Error Vector
            if tx is not None:
                error_color = (0, 255, 0) if abs(tx - center_line_x) < dead_zone_w else (0, 0, 255)
                cv2.line(frame, (int(center_line_x), int(ty)), (int(tx), int(ty)), error_color, 2)
