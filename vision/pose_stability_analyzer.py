"""
vision/pose_stability_analyzer.py
================================
Analyzes pose jitter, distance fluctuations, and tracking consistency over time.
"""
import time
from collections import deque
from typing import List, Dict, Optional
import numpy as np

from detectors.base_detector import DetectionTarget
from utils.logger import get_logger

log = get_logger(__name__)

class PoseStabilityAnalyzer:
    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        # Store history per marker ID: {id: deque([(timestamp, target), ...])}
        self.history: Dict[str, deque] = {}
        self.stability_scores: Dict[str, float] = {}
        self.last_update = time.time()

    def update(self, targets: List[DetectionTarget]) -> float:
        """
        Updates history with new targets and returns an aggregate stability score (0.0 to 1.0).
        """
        now = time.time()
        current_ids = {t.id for t in targets}
        
        # Cleanup old history for lost markers (optionally keep for a bit)
        to_remove = []
        for marker_id in self.history:
            if marker_id not in current_ids:
                # If marker lost for > 2 seconds, clear history
                if now - self.history[marker_id][-1][0] > 2.0:
                    to_remove.append(marker_id)
        for rid in to_remove:
            del self.history[rid]
            if rid in self.stability_scores:
                del self.stability_scores[rid]

        if not targets:
            return 0.0

        for target in targets:
            if target.id not in self.history:
                self.history[target.id] = deque(maxlen=self.window_size)
            self.history[target.id].append((now, target))
            
            self.stability_scores[target.id] = self._calculate_marker_stability(target.id)

        # Aggregate score: average of active markers
        if not self.stability_scores:
            return 0.0
        return sum(self.stability_scores.values()) / len(self.stability_scores)

    def _calculate_marker_stability(self, marker_id: str) -> float:
        hist = self.history[marker_id]
        if len(hist) < 5:
            return 1.0 # Not enough data to judge jitter yet

        # Extract metrics for variance analysis
        distances = [t.distance_m for _, t in hist if t.distance_m is not None]
        yaws = [t.yaw for _, t in hist]
        
        if not distances:
            return 0.5

        # Calculate Coefficient of Variation (StdDev / Mean) for distance
        dist_mean = np.mean(distances)
        dist_std = np.std(distances)
        dist_cv = (dist_std / dist_mean) if dist_mean > 0 else 1.0
        
        # Yaw jitter (absolute diffs)
        yaw_diffs = np.abs(np.diff(yaws))
        avg_yaw_jitter = np.mean(yaw_diffs) if len(yaw_diffs) > 0 else 0.0

        # Heuristic mapping to 0.0-1.0
        # dist_cv of 0.05 (5%) is quite jittery for a static or slow robot
        # avg_yaw_jitter of 2 degrees per frame is jittery
        dist_score = max(0, 1.0 - (dist_cv / 0.1))
        yaw_score = max(0, 1.0 - (avg_yaw_jitter / 5.0))
        
        return (dist_score * 0.6) + (yaw_score * 0.4)

    def get_stability_report(self, marker_id: str) -> Dict[str, float]:
        hist = self.history.get(marker_id, [])
        if not hist:
            return {}
            
        distances = [t.distance_m for _, t in hist if t.distance_m is not None]
        return {
            "score": self.stability_scores.get(marker_id, 0.0),
            "jitter_m": np.std(distances) if len(distances) > 1 else 0.0,
            "samples": len(hist)
        }
