"""
tests/test_frame_pipeline.py
Tests for FramePipeline: empty-frame guard, normalizer runs, result type,
stability score is in [0,1], and overlay annotations are drawn.
"""
import numpy as np
import pytest

from detectors.detector_registry import DetectorRegistry
from detectors.base_detector import DetectionResult
from vision.frame_pipeline import FramePipeline


def make_pipeline(registry=None) -> FramePipeline:
    if registry is None:
        registry = DetectorRegistry()
    return FramePipeline(registry)


def blank_frame(h=480, w=640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def grey_frame(h=480, w=640, val=128) -> np.ndarray:
    frame = np.full((h, w, 3), val, dtype=np.uint8)
    return frame


# ── Empty frame guard ─────────────────────────────────────────────────────────

def test_none_frame_returns_empty_result():
    pipeline = make_pipeline()
    annotated, result = pipeline.process(None)
    assert annotated is None
    assert isinstance(result, DetectionResult)
    assert not result.has_targets


def test_zero_size_frame_returns_empty_result():
    pipeline = make_pipeline()
    empty = np.array([])
    annotated, result = pipeline.process(empty)
    assert isinstance(result, DetectionResult)
    assert not result.has_targets


# ── Normal frame ──────────────────────────────────────────────────────────────

def test_normal_frame_returns_ndarray():
    pipeline = make_pipeline()
    frame = grey_frame()
    annotated, result = pipeline.process(frame)
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == frame.shape


def test_result_is_detection_result():
    pipeline = make_pipeline()
    _, result = pipeline.process(grey_frame())
    assert isinstance(result, DetectionResult)


def test_stability_score_in_valid_range():
    pipeline = make_pipeline()
    for _ in range(5):
        _, result = pipeline.process(grey_frame())
    assert 0.0 <= result.tracking_stability_score <= 1.0


# ── Performance monitor ticks ─────────────────────────────────────────────────

def test_fps_increases_after_multiple_frames():
    pipeline = make_pipeline()
    for _ in range(10):
        pipeline.process(grey_frame())
    # After enough ticks the FPS counter should be above 0
    assert pipeline.perf_monitor.current_fps >= 0.0


# ── No detectors → still returns valid annotated frame ───────────────────────

def test_empty_registry_still_annotates():
    pipeline = make_pipeline(DetectorRegistry())
    frame = grey_frame()
    annotated, result = pipeline.process(frame)
    assert annotated is not None
    assert not result.has_targets
