import time
import pytest
from safety.safety_monitor import SafetyMonitor
from controllers.command_queue import CommandQueue, CommandPriority
from utils.exceptions import SafetyViolationError


def _make_monitor(auto_recovery: bool = False) -> tuple:
    queue = CommandQueue()
    monitor = SafetyMonitor(queue)
    monitor.max_stale_ms = 100
    monitor._auto_recovery = auto_recovery
    return queue, monitor


def test_safety_monitor_stale_frame_no_auto_recovery():
    """Without auto-recovery, a stale frame raises SafetyViolationError."""
    queue, monitor = _make_monitor(auto_recovery=False)

    monitor.notify_frame_received()
    monitor.check_health()
    assert not monitor._e_stop_triggered

    time.sleep(0.15)

    with pytest.raises(SafetyViolationError):
        monitor.check_health()

    assert monitor._e_stop_triggered

    cmd = queue.pop()
    assert cmd is not None
    assert cmd.command_type == "E_STOP"
    assert cmd.priority == CommandPriority.CRITICAL


def test_safety_monitor_stale_frame_with_auto_recovery():
    """With auto-recovery, a stale frame queues E-STOP but does NOT raise."""
    queue, monitor = _make_monitor(auto_recovery=True)

    monitor.notify_frame_received()
    time.sleep(0.15)

    # Should not raise
    monitor.check_health()

    assert monitor._e_stop_triggered
    cmd = queue.pop()
    assert cmd is not None
    assert cmd.command_type == "E_STOP"


def test_safety_monitor_auto_recovery_resets_on_frame():
    """When auto-recovery is on, receiving a frame after E-STOP clears it."""
    queue, monitor = _make_monitor(auto_recovery=True)

    monitor.notify_frame_received()
    time.sleep(0.15)
    monitor.check_health()
    assert monitor._e_stop_triggered

    # Simulate camera coming back
    monitor.notify_frame_received()
    assert not monitor._e_stop_triggered
