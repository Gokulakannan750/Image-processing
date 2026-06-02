"""
tests/test_decision_engine.py
Tests for DecisionEngine: obstacle blocking, turn trigger, steering correction,
recovery transitions, and stop/resume behaviour.
"""
import pytest

from controllers.command_queue import CommandQueue
from detectors.base_detector import DetectionResult, DetectionTarget, ObstacleDetection
from navigation.decision_engine import DecisionEngine
from navigation.vehicle_state import State


def make_engine():
    """Return a fresh DecisionEngine with a clean CommandQueue."""
    q = DecisionEngine(CommandQueue())
    # Bypass state-machine min-duration guard for fast tests
    q.state_machine.min_duration_s = 0.0
    return q


def make_target(
    center_x=640.0,
    distance_m=2.0,
    is_turn_trigger=False,
    priority=1,
) -> DetectionTarget:
    return DetectionTarget(
        id="test-marker",
        center_x=center_x,
        center_y=360.0,
        distance_m=distance_m,
        is_turn_trigger=is_turn_trigger,
        priority=priority,
    )


def make_obstacle(is_critical=True) -> ObstacleDetection:
    return ObstacleDetection(
        label="person",
        confidence=0.9,
        bbox=(0, 0, 100, 100),
        class_id=0,
        is_critical=is_critical,
    )


# ── Obstacle blocking ────────────────────────────────────────────────────────

def test_critical_obstacle_stops_vehicle():
    engine = make_engine()
    result = DetectionResult(obstacles=[make_obstacle(is_critical=True)])
    state, steering = engine.process_detection(result)
    assert state == State.STOPPED
    assert steering == 0.0


def test_non_critical_obstacle_does_not_stop():
    engine = make_engine()
    result = DetectionResult(obstacles=[make_obstacle(is_critical=False)])
    state, _ = engine.process_detection(result)
    # Non-critical obstacle shouldn't trigger a stop (navigation continues)
    assert state != State.STOPPED


def test_obstacle_clear_resumes_driving():
    engine = make_engine()

    # First: obstacle blocks
    blocked_result = DetectionResult(obstacles=[make_obstacle(is_critical=True)])
    engine.process_detection(blocked_result)

    # Then: obstacle gone, target present
    clear_result = DetectionResult(targets=[make_target()])
    state, _ = engine.process_detection(clear_result)
    assert state == State.DRIVING


# ── Turn trigger ─────────────────────────────────────────────────────────────

def test_turn_trigger_queues_u_turn():
    q = CommandQueue()
    engine = DecisionEngine(q)
    engine.state_machine.min_duration_s = 0.0

    result = DetectionResult(targets=[make_target(is_turn_trigger=True, distance_m=1.0)])
    engine.process_detection(result)

    # The state machine transitions inside process_detection before returning;
    # the returned state is captured one step earlier (from recovery_manager),
    # so check the state machine directly.
    assert engine.state_machine.current_state == State.TURNING
    cmd = q.pop(timeout=None)
    assert cmd is not None
    assert cmd.command_type == "U_TURN"


def test_turn_not_triggered_when_obstacle_present():
    q = CommandQueue()
    engine = DecisionEngine(q)
    engine.state_machine.min_duration_s = 0.0

    # Obstacle + turn-trigger simultaneously — obstacle wins
    result = DetectionResult(
        targets=[make_target(is_turn_trigger=True)],
        obstacles=[make_obstacle(is_critical=True)],
    )
    state, _ = engine.process_detection(result)
    assert state == State.STOPPED


# ── Steering correction ──────────────────────────────────────────────────────

def test_centered_target_produces_small_steering():
    engine = make_engine()
    # center_x == camera center (1280/2 = 640) → raw error = 0
    result = DetectionResult(targets=[make_target(center_x=640.0)])
    _, steering = engine.process_detection(result)
    assert abs(steering) < 1.0  # within dead-zone


def test_off_center_target_produces_nonzero_steering():
    engine = make_engine()
    # Marker far to the right
    result = DetectionResult(targets=[make_target(center_x=900.0)])
    _, steering = engine.process_detection(result)
    # Smoothed EMA from 0 → non-zero on first tick
    assert steering != 0.0


# ── Empty result ─────────────────────────────────────────────────────────────

def test_no_target_does_not_crash():
    engine = make_engine()
    result = DetectionResult()
    state, steering = engine.process_detection(result)
    assert isinstance(state, State)
    assert isinstance(steering, float)


# ── confirm_turn_complete ─────────────────────────────────────────────────────

def test_confirm_turn_complete_returns_to_driving():
    engine = make_engine()
    engine.state_machine.transition_to(State.TURNING, "test")
    engine.confirm_turn_complete()
    assert engine.state_machine.current_state == State.DRIVING
