"""
tests/test_orchard_follower.py
Tests for the orchard alley follower using synthetic frames:
textured green 'tree walls' on the sides with a smooth alley between them.
"""

import numpy as np

from detectors.orchard_follower import OrchardFollowerDetector
from detectors.base_detector import DetectionResult


def _orchard_frame(w=480, h=854, alley_centre=None, alley_half=70, walls=True):
    """Two textured green tree walls with a smooth green alley between them."""
    if alley_centre is None:
        alley_centre = w // 2
    rng = np.random.default_rng(1)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Smooth green alley everywhere (low texture).
    img[:, :] = (40, 120, 40)

    if walls:
        lx = alley_centre - alley_half
        rx = alley_centre + alley_half
        # Textured dark-green foliage on both sides (high local variance).
        noise = rng.integers(0, 90, size=(h, max(1, lx), 3), dtype=np.uint8)
        img[:, :lx] = noise + np.array([10, 60, 10], dtype=np.uint8)
        noise2 = rng.integers(0, 90, size=(h, w - rx, 3), dtype=np.uint8)
        img[:, rx:] = noise2 + np.array([10, 60, 10], dtype=np.uint8)
    return img


def test_initialises():
    assert OrchardFollowerDetector() is not None


def test_returns_result():
    det = OrchardFollowerDetector()
    _, res = det.process_frame(_orchard_frame())
    assert isinstance(res, DetectionResult)


def test_centred_alley_detected_near_centre():
    det = OrchardFollowerDetector()
    img = _orchard_frame(alley_centre=240)
    res = None
    for _ in range(6):  # let the output EMA settle
        _, res = det.process_frame(img.copy())
    assert res.has_targets
    assert abs(res.primary_target.center_x - 240) < 60


def test_offset_alley_detected_to_side():
    det = OrchardFollowerDetector()
    img = _orchard_frame(alley_centre=150)  # alley to the left
    res = None
    for _ in range(8):
        _, res = det.process_frame(img.copy())
    assert res.has_targets
    assert res.primary_target.center_x < 240  # left of centre


def test_row_end_when_walls_disappear():
    det = OrchardFollowerDetector()
    # Open headland: no tree walls (all smooth) -> row should end.
    open_field = _orchard_frame(walls=False)
    triggered = False
    for _ in range(det.rowend_frames + 4):
        _, res = det.process_frame(open_field.copy())
        if res.has_targets and res.primary_target.is_turn_trigger:
            triggered = True
    assert triggered


def test_walls_present_no_false_rowend():
    det = OrchardFollowerDetector()
    img = _orchard_frame()
    last = None
    for _ in range(det.rowend_frames + 4):
        _, last = det.process_frame(img.copy())
    if last.has_targets:
        assert last.primary_target.is_turn_trigger is False


def test_empty_frame_safe():
    det = OrchardFollowerDetector()
    _, res = det.process_frame(np.array([]))
    assert isinstance(res, DetectionResult)
    assert not res.has_targets
