"""
vision/performance_monitor.py
=============================
Tracks FPS, pipeline latency, and triggers warnings if performance degrades.
"""
import time
from collections import deque
import cv2
import numpy as np
from typing import Tuple

from config.config_manager import config_manager
from utils.logger import get_logger

log = get_logger(__name__)

class PerformanceMonitor:
    def __init__(self):
        self.frame_times = deque(maxlen=30)
        self.latency_times = deque(maxlen=60)
        self.last_frame_time = time.time()
        self.current_fps = 0.0
        self.min_fps_warning = config_manager.get("safety.min_fps_warning", 10.0)
        self.show_overlay = config_manager.get("debug.show_performance", True)
        self.show_graphs = config_manager.get("debug.show_latency_graph", True)

    def tick(self) -> None:
        """Called at the start of pipeline processing."""
        now = time.time()
        self.frame_times.append(now - self.last_frame_time)
        self.last_frame_time = now
        
        if len(self.frame_times) > 0:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            self.current_fps = 1.0 / avg_time if avg_time > 0 else 0.0

            if self.current_fps < self.min_fps_warning:
                log.warning("Performance degraded! FPS: %.1f (Min: %.1f)", 
                            self.current_fps, self.min_fps_warning)

    def record_latency(self, latency_ms: float) -> None:
        self.latency_times.append(latency_ms)

    def draw_overlay(self, frame: np.ndarray) -> None:
        if not self.show_overlay:
            return
            
        h, w = frame.shape[:2]
        
        # 1. FPS Label
        color = (0, 255, 0) if self.current_fps >= self.min_fps_warning else (0, 0, 255)
        text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(frame, text, (w - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
        # 2. Latency Label
        if self.latency_times:
            avg_lat = sum(self.latency_times) / len(self.latency_times)
            lat_text = f"LAT: {avg_lat:.1f}ms"
            cv2.putText(frame, lat_text, (w - 150, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 3. Mini Graph (Bottom Right)
        if self.show_graphs and len(self.latency_times) > 2:
            self._draw_mini_graph(frame, self.latency_times, (w - 160, h - 60), (150, 50))

    def _draw_mini_graph(self, frame: np.ndarray, data: deque, pos: Tuple[int, int], size: Tuple[int, int]):
        x_off, y_off = pos
        gw, gh = size
        
        # Semi-transparent background
        sub_img = frame[y_off:y_off+gh, x_off:x_off+gw]
        rect = np.zeros(sub_img.shape, dtype=np.uint8)
        cv2.rectangle(rect, (0, 0), (gw, gh), (50, 50, 50), -1)
        frame[y_off:y_off+gh, x_off:x_off+gw] = cv2.addWeighted(sub_img, 0.7, rect, 0.3, 0)
        
        # Draw line
        max_val = max(data) if max(data) > 0 else 1
        points = []
        for i, val in enumerate(data):
            px = int(i * (gw / len(data)))
            py = gh - int((val / max_val) * gh)
            points.append((x_off + px, y_off + py))
            
        for i in range(len(points)-1):
            cv2.line(frame, points[i], points[i+1], (0, 255, 255), 1)
