"""
tests/test_row_follower.py
Tests for the markerless crop-row follower using synthetic frames.
"""

import numpy as np

from detectors.row_follower import RowFollowerDetector
from detectors.base_detector import DetectionResult


def _two_rows_with_gap(w=640, h=360, gap_centre=None):
    """White soil background with two green crop stripes and a soil gap.

    The gap (path) is centred at gap_centre (default = image centre).
    """
    if gap_centre is None:
        gap_centre = w // 2
    img = np.full((h, w, 3), (180, 180, 180), dtype=np.uint8)  # grey soil
    stripe_w = 90
    gap_half = 70
    # Left crop stripe ends at gap_centre - gap_half
    lx1 = gap_centre - gap_half - stripe_w
    lx2 = gap_centre - gap_half
    rx1 = gap_centre + gap_half
    rx2 = gap_centre + gap_half + stripe_w
    green = (40, 180, 40)
    img[:, max(0, lx1) : max(0, lx2)] = green
    img[:, min(w, rx1) : min(w, rx2)] = green
    return img


def test_detector_initialises():
    det = RowFollowerDetector()
    assert det is not None


def test_returns_detection_result():
    det = RowFollowerDetector()
    _, result = det.process_frame(_two_rows_with_gap())
    assert isinstance(result, DetectionResult)


def test_centred_gap_gives_near_centre_target():
    det = RowFollowerDetector()
    img = _two_rows_with_gap(gap_centre=320)  # centred path
    _, result = det.process_frame(img)
    assert result.has_targets
    tgt = result.primary_target
    # Path centre should be near the image centre (320 +/- 40 px).
    assert abs(tgt.center_x - 320) < 40


def test_offset_gap_detected_to_the_side():
    det = RowFollowerDetector()
    img = _two_rows_with_gap(gap_centre=200)  # path to the left
    _, result = det.process_frame(img)
    assert result.has_targets
    # Detected path centre should be left of image centre.
    assert result.primary_target.center_x < 300


def test_row_end_triggers_after_enough_empty_frames():
    det = RowFollowerDetector()
    # An empty (no vegetation) frame ahead = row ended.
    empty = np.full((360, 640, 3), (180, 180, 180), dtype=np.uint8)
    triggered = False
    for _ in range(det.rowend_frames + 2):
        _, result = det.process_frame(empty)
        if result.has_targets and result.primary_target.is_turn_trigger:
            triggered = True
    assert triggered


def test_full_crop_ahead_does_not_trigger_row_end():
    det = RowFollowerDetector()
    img = _two_rows_with_gap()
    last = None
    for _ in range(det.rowend_frames + 2):
        _, result = det.process_frame(img)
        last = result
    # With crop rows present ahead, no row-end trigger.
    if last.has_targets:
        assert last.primary_target.is_turn_trigger is False


def test_empty_frame_safe():
    det = RowFollowerDetector()
    annotated, result = det.process_frame(np.array([]))
    assert isinstance(result, DetectionResult)
    assert not result.has_targets
