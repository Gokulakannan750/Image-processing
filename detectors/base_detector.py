"""
detectors/base_detector.py
==========================
Abstract base class and data structures for computer vision detectors.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class DetectionTarget:
    """Represents a single detected marker/object with advanced validation metrics."""
    id: str
    distance_m: Optional[float] = None
    center_x: Optional[float] = None
    center_y: Optional[float] = None

    # Orientation (Euler Angles in Degrees)
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    # Robustness Metrics
    confidence: float = 0.0        # 0.0 to 1.0
    blur_score: float = 0.0        # Laplacian variance
    visibility_score: float = 0.0  # Percentage of marker pixels visible
    latency_ms: float = 0.0        # Detector-specific latency

    priority: int = 1  # Lower number = higher priority
    is_turn_trigger: bool = False


@dataclass
class ObstacleDetection:
    """A single object detected by the YOLO obstacle detector."""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: int
    is_critical: bool = False        # True when bbox area exceeds danger threshold


@dataclass
class DetectionResult:
    """Aggregate result from a detector for a single frame with validation data."""
    targets: List[DetectionTarget] = field(default_factory=list)
    obstacles: List[ObstacleDetection] = field(default_factory=list)

    # Global Frame Metrics
    frame_latency_ms: float = 0.0
    tracking_stability_score: float = 1.0  # 0.0 (unstable) to 1.0 (perfect)
    visibility_quality: float = 1.0        # Aggregate visibility of all targets

    @property
    def has_targets(self) -> bool:
        return len(self.targets) > 0

    @property
    def primary_target(self) -> Optional[DetectionTarget]:
        if not self.targets:
            return None
        sorted_targets = sorted(self.targets, key=lambda t: (t.priority, t.distance_m or float('inf')))
        return sorted_targets[0]

    @property
    def has_critical_obstacle(self) -> bool:
        return any(o.is_critical for o in self.obstacles)

    @property
    def should_turn(self) -> bool:
        if not self.targets:
            return False
        return any(t.is_turn_trigger for t in self.targets)


class BaseDetector(ABC):
    """Common interface for all end-of-row marker detectors."""
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, DetectionResult]:
        """
        Analyse a single camera frame.
        Returns: (annotated_frame, DetectionResult)
        """
        ...
