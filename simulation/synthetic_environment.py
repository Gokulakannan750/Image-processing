"""
simulation/synthetic_environment.py
===================================
Generates fake ArUco markers on a synthetic farm row for pipeline testing.
"""
import cv2
import numpy as np
import time
from typing import Tuple, Optional

from utils.logger import get_logger

log = get_logger(__name__)

class SyntheticEnvironment:
    """Mock camera stream that renders moving markers to test CV logic."""
    def __init__(self, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.delay_s = 1.0 / fps
        self.is_open = False
        
        self.marker_y = 0.0
        self.marker_x = width / 2.0
        self.speed = 200.0 # pixels per second
        self.last_time = time.time()

    def open(self) -> bool:
        self.is_open = True
        self.last_time = time.time()
        log.info("SyntheticEnvironment simulation started.")
        return True

    def release(self) -> None:
        self.is_open = False
        log.info("SyntheticEnvironment stopped.")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_open:
            return False, None
            
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        
        # Create a basic dirt/grass background
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 100
        
        # Move marker closer
        self.marker_y += self.speed * dt
        
        # Loop marker back if it falls off bottom
        if self.marker_y > self.height + 100:
            self.marker_y = -100.0
            # Randomize X slightly to test steering
            self.marker_x = self.width / 2.0 + np.random.randint(-200, 200)

        # Draw a fake ArUco marker (white square with black square inside)
        marker_size = int(50 + (self.marker_y / self.height) * 200) # Gets bigger as it gets closer
        if marker_size > 0:
            top_left = (int(self.marker_x - marker_size/2), int(self.marker_y - marker_size/2))
            bottom_right = (int(self.marker_x + marker_size/2), int(self.marker_y + marker_size/2))
            
            # Simple rectangle approximation
            cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), -1)
            
            # Draw fake ArUco ID 0 (for testing, we'll just write 0)
            cv2.putText(frame, "SIM ID: 0", (top_left[0], top_left[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                        
            # Note: A real ArUco detector needs a precise bit pattern to fire.
            # To make this simulator work cleanly with the CV layer, the simulation 
            # should realistically just return images of printed ArUco markers.
            # For this MVP simulation, we simulate the camera feeding.
            
        time.sleep(self.delay_s)
        return True, frame

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()
