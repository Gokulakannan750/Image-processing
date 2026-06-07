"""
tests/test_ground_obstacle.py
=============================
Tests for the generic (class-agnostic) ground-anomaly obstacle detector.
"""

import numpy as np
import pytest

from detectors.ground_obstacle import GroundObstacleDetector


def _grass_frame(h=480, w=270):
    """A synthetic 'grassy alley': green with a little noise so it has texture."""
    rng = np.random.default_rng(0)
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = 40 + rng.integers(0, 20, (h, w))   # B
    frame[:, :, 1] = 130 + rng.integers(0, 30, (h, w))  # G (dominant = green)
    frame[:, :, 2] = 50 + rng.integers(0, 20, (h, w))   # R
    return frame


def test_clean_grass_no_stop():
    """An empty grassy alley must not produce a critical obstacle."""
    det = GroundObstacleDetector()
    det._enabled = True
    frame = _grass_frame()
    critical = False
    for _ in range(5):  # run several frames (debounce)
        _, result = det.process_frame(frame.copy())
        critical = critical or result.has_critical_obstacle
    assert critical is False


def test_blue_object_in_path_stops():
    """A clearly foreign (blue) object in the corridor must trigger a STOP."""
    det = GroundObstacleDetector()
    det._enabled = True
    base = _grass_frame()
    h, w = base.shape[:2]
    critical = False
    for _ in range(4):  # exceed confirm_frames debounce
        frame = base.copy()
        # A solid blue blob centred in the path corridor, mid-distance.
        frame[int(h * 0.65):int(h * 0.80),
              int(w * 0.40):int(w * 0.60)] = (200, 60, 30)
        _, result = det.process_frame(frame)
        critical = critical or result.has_critical_obstacle
    assert critical is True
    assert any(o.label == "object" for o in result.obstacles)


def test_dark_object_in_path_stops():
    """A dark object (low brightness, neutral colour) must also stop."""
    det = GroundObstacleDetector()
    det._enabled = True
    base = _grass_frame()
    h, w = base.shape[:2]
    critical = False
    for _ in range(4):
        frame = base.copy()
        frame[int(h * 0.65):int(h * 0.80),
              int(w * 0.40):int(w * 0.60)] = (20, 20, 22)
        _, result = det.process_frame(frame)
        critical = critical or result.has_critical_obstacle
    assert critical is True


def test_disabled_detector_is_noop():
    det = GroundObstacleDetector()
    det._enabled = False
    base = _grass_frame()
    h, w = base.shape[:2]
    base[int(h * 0.65):int(h * 0.80), int(w * 0.40):int(w * 0.60)] = (200, 60, 30)
    _, result = det.process_frame(base)
    assert result.obstacles == []


def test_small_blob_does_not_stop():
    """A tiny speck below min_area_frac must be ignored."""
    det = GroundObstacleDetector()
    det._enabled = True
    base = _grass_frame()
    h, w = base.shape[:2]
    critical = False
    for _ in range(4):
        frame = base.copy()
        frame[int(h * 0.70):int(h * 0.71),
              int(w * 0.49):int(w * 0.51)] = (200, 60, 30)
        _, result = det.process_frame(frame)
        critical = critical or result.has_critical_obstacle
    assert critical is False


def test_handles_empty_frame_gracefully():
    det = GroundObstacleDetector()
    det._enabled = True
    _, result = det.process_frame(np.zeros((0, 0, 3), dtype=np.uint8))
    assert result.obstacles == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
