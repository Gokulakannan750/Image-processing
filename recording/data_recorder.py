"""
recording/data_recorder.py
==========================
Optionally saves raw frames, annotated frames, and navigation JSON metadata
for later replay, debugging, or AI training.
"""
import os
import cv2
import time
import json
from typing import Optional, Dict, Any
import numpy as np

from detectors.base_detector import DetectionResult
from config.config_manager import config_manager
from utils.logger import get_logger

log = get_logger(__name__)

class DataRecorder:
    def __init__(self):
        self.enabled = config_manager.get("recording.enabled", False)
        self.save_raw = config_manager.get("recording.save_raw_frames", True)
        self.save_annotated = config_manager.get("recording.save_annotated_frames", True)
        
        self.base_dir = config_manager.get("recording.output_dir", "recordings")
        
        if self.enabled:
            # Create a session directory based on timestamp
            session_name = time.strftime("session_%Y%m%d_%H%M%S")
            self.session_dir = os.path.join(self.base_dir, session_name)
            
            self.raw_dir = os.path.join(self.session_dir, "raw")
            self.annotated_dir = os.path.join(self.session_dir, "annotated")
            self.meta_dir = os.path.join(self.session_dir, "metadata")
            
            os.makedirs(self.raw_dir, exist_ok=True)
            os.makedirs(self.annotated_dir, exist_ok=True)
            os.makedirs(self.meta_dir, exist_ok=True)
            
            log.info("DataRecorder initialized. Saving to %s", self.session_dir)
            self._frame_count = 0

    def record(self, 
               raw_frame: np.ndarray, 
               annotated_frame: np.ndarray, 
               result: DetectionResult,
               metadata: Dict[str, Any]) -> None:
        if not self.enabled:
            return
            
        timestamp = time.time()
        frame_id = f"frame_{self._frame_count:06d}"
        
        # Save images
        if self.save_raw and raw_frame is not None:
            cv2.imwrite(os.path.join(self.raw_dir, f"{frame_id}.jpg"), raw_frame)
            
        if self.save_annotated and annotated_frame is not None:
            cv2.imwrite(os.path.join(self.annotated_dir, f"{frame_id}.jpg"), annotated_frame)
            
        # Serialize DetectionResult
        serialized_targets = []
        for t in result.targets:
            serialized_targets.append({
                "id": t.id,
                "dist": t.distance_m,
                "cx": t.center_x,
                "cy": t.center_y,
                "yaw": t.yaw,
                "pitch": t.pitch,
                "roll": t.roll,
                "conf": t.confidence,
                "latency": t.latency_ms
            })

        # Save metadata
        meta_data_full = {
            "timestamp": timestamp,
            "frame_id": frame_id,
            "frame_index": self._frame_count,
            "latency_ms": result.frame_latency_ms,
            "stability": result.tracking_stability_score,
            "targets": serialized_targets,
            **metadata
        }
        
        if config_manager.get("recording.save_metadata_json", True):
            with open(os.path.join(self.meta_dir, f"{frame_id}.json"), "w") as f:
                json.dump(meta_data_full, f, indent=2)
            
        self._frame_count += 1
