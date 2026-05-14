"""
recording/replay_system.py
==========================
Mock camera stream that feeds recorded frames back into the pipeline.
"""
import os
import cv2
import time
import glob
import numpy as np
from typing import Tuple, Optional

from utils.logger import get_logger

log = get_logger(__name__)

class ReplayCameraStream:
    """
    Imitates the CameraStream interface but reads from a recording directory.
    """
    def __init__(self, session_dir: str, fps: float = 30.0):
        self.session_dir = session_dir
        self.raw_dir = os.path.join(session_dir, "raw")
        self.fps = fps
        self.delay_s = 1.0 / fps if fps > 0 else 0.0
        
        self.frames = []
        self.current_idx = 0
        self.is_open = False
        
    def open(self) -> bool:
        if not os.path.exists(self.raw_dir):
            log.error("Replay directory not found: %s", self.raw_dir)
            return False
            
        # Load sorted list of frames
        search_pattern = os.path.join(self.raw_dir, "*.jpg")
        self.frames = sorted(glob.glob(search_pattern))
        
        if not self.frames:
            log.error("No frames found in %s", self.raw_dir)
            return False
            
        log.info("Replay stream opened: %d frames found.", len(self.frames))
        self.is_open = True
        self.current_idx = 0
        return True
        
    def release(self) -> None:
        self.is_open = False
        log.info("Replay stream finished.")
        
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_open:
            return False, None
            
        if self.current_idx >= len(self.frames):
            log.info("Reached end of replay sequence.")
            return False, None
            
        frame_path = self.frames[self.current_idx]
        frame = cv2.imread(frame_path)
        
        self.current_idx += 1
        time.sleep(self.delay_s) # Simulate real-time playback
        
        if frame is None:
            log.error("Failed to read replay frame: %s", frame_path)
            return False, None
            
        return True, frame

    def __enter__(self) -> "ReplayCameraStream":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.release()
