"""
tests/test_navigation_filter.py
Tests for NavigationFilter: EMA smoothing, dead-zone elimination, reset, hot-reload.
"""

import pytest
from navigation.navigation_filter import NavigationFilter


def make_filter(alpha: float = 0.5, dead_zone: float = 20.0) -> NavigationFilter:
    f = NavigationFilter()
    f.steering_filter.alpha = alpha
    f.distance_filter.alpha = alpha
    f.dead_zone_x = dead_zone
    return f


# ── Dead-zone ─────────────────────────────────────────────────────────────────


def test_error_within_dead_zone_produces_zero():
    f = make_filter(dead_zone=20.0)
    result = f.process_alignment(10.0)  # within ±20 dead-zone
    # EMA of 0 from rest → 0
    assert result == 0.0


def test_error_outside_dead_zone_is_nonzero():
    f = make_filter(alpha=1.0, dead_zone=20.0)  # alpha=1 → no smoothing, immediate
    result = f.process_alignment(50.0)
    assert result == pytest.approx(50.0)


def test_negative_error_outside_dead_zone():
    f = make_filter(alpha=1.0, dead_zone=20.0)
    result = f.process_alignment(-50.0)
    assert result == pytest.approx(-50.0)


# ── EMA smoothing ─────────────────────────────────────────────────────────────


def test_ema_smoothing_approaches_target():
    # PoseFilter seeds from first call directly (no smoothing on call 1).
    # Pre-seed at 0.0, then drive toward 100.0 and verify values increase.
    f = make_filter(alpha=0.3, dead_zone=0.0)
    f.process_alignment(0.0)  # seed at 0
    values = [float(f.process_alignment(100.0)) for _ in range(5)]
    # Values should be strictly increasing toward 100
    assert all(values[i] < values[i + 1] for i in range(len(values) - 1))
    # After 5 steps with alpha=0.3 from 0, should still be below 100
    assert values[-1] < 100.0


# ── Reset ─────────────────────────────────────────────────────────────────────


def test_reset_clears_ema_state():
    # Seed at 0, drive toward 100, reset, seed at 0 again, then check that
    # the next non-zero call is smoothed (not jumping straight to 100).
    f = make_filter(alpha=0.3, dead_zone=0.0)
    f.process_alignment(0.0)  # seed
    for _ in range(30):
        f.process_alignment(100.0)  # saturate toward 100
    f.reset()
    f.process_alignment(0.0)  # re-seed at 0 after reset
    val = float(f.process_alignment(100.0))
    # With alpha=0.3, first step from 0 → 30.0; well below 100
    assert val < 80.0


# ── hot-reload ────────────────────────────────────────────────────────────────


def test_reload_config_updates_dead_zone(monkeypatch):
    from config.config_manager import config_manager

    monkeypatch.setattr(
        config_manager,
        "get",
        lambda key, default=None: (
            40.0
            if key == "navigation.dead_zone_x"
            else (0.3 if key == "navigation.smoothing_alpha" else default)
        ),
    )
    f = NavigationFilter()
    f.reload_config()
    assert f.dead_zone_x == pytest.approx(40.0)
