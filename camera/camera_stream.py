"""
camera/camera_stream.py
========================
Threaded wrapper around cv2.VideoCapture.
Provides a non-blocking stream and auto-recovery health monitoring.
"""
import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple

from config.config_manager import config_manager
from utils.logger import get_logger
from utils.exceptions import CameraInitializationError

log = get_logger(__name__)


class CameraStream:
    def __init__(self) -> None:
        self._index = config_manager.get("camera.index", 0)
        self._width = config_manager.get("camera.width", 1280)
        self._height = config_manager.get("camera.height", 720)
        self._fps = config_manager.get("camera.fps", 30)
        self._timeout_s = config_manager.get("camera.timeout_s", 2.0)
        
        self.max_attempts = config_manager.get("camera.max_reconnect_attempts", 5)
        self.reconnect_delay = config_manager.get("camera.reconnect_delay_s", 1.0)
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._current_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.reconnect_count = 0

    def open(self) -> bool:
        log.info("Opening camera index %d …", self._index)
        self._cap = cv2.VideoCapture(self._index)
        
        if not self._cap.isOpened():
            log.error("Could not open camera index %d.", self._index)
            return False
            
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        
        ret, frame = self._cap.read()
        if not ret:
            log.error("Camera opened but failed to read initial frame.")
            return False
            
        with self._frame_lock:
            self._current_frame = frame
            
        log.info("Camera %d opened successfully.", self._index)
        
        self._running = True
        self._thread = threading.Thread(target=self._update, daemon=True, name="CameraThread")
        self._thread.start()
        
        return True

    def _attempt_reconnect(self) -> bool:
        """Internal method to recover camera without crashing app."""
        log.warning("Attempting camera reconnect (%d/%d)...", self.reconnect_count+1, self.max_attempts)
        if self._cap is not None:
            self._cap.release()
            
        time.sleep(self.reconnect_delay)
        
        self._cap = cv2.VideoCapture(self._index)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)
            ret, _ = self._cap.read()
            if ret:
                log.info("Camera successfully recovered.")
                self.reconnect_count = 0
                return True
                
        self.reconnect_count += 1
        return False

    def _update(self) -> None:
        last_success = time.time()
        
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                if self.reconnect_count < self.max_attempts:
                    self._attempt_reconnect()
                else:
                    log.error("Max reconnect attempts reached. Giving up.")
                    break
                
            ret, frame = self._cap.read()
            now = time.time()
            
            if ret:
                with self._frame_lock:
                    self._current_frame = frame
                last_success = now
            else:
                elapsed = now - last_success
                if elapsed > self._timeout_s:
                    log.error("Camera timeout. Initiating recovery...")
                    if self.reconnect_count < self.max_attempts:
                        self._attempt_reconnect()
                        last_success = time.time() # reset timer
                    else:
                        break
                else:
                    time.sleep(0.1)

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            log.info("Camera %d released.", self._index)
        self._cap = None

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._running:
            return False, None
            
        with self._frame_lock:
            if self._current_frame is not None:
                return True, self._current_frame.copy()
            else:
                return False, None

    def __enter__(self) -> "CameraStream":
        if not self.open():
            raise CameraInitializationError("Failed to open camera via context manager.")
        return self

    def __exit__(self, *_) -> None:
        self.release()

    @property
    def is_open(self) -> bool:
        return self._running and self._cap is not None and self._cap.isOpened()
