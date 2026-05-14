import time
import pytest
from safety.safety_monitor import SafetyMonitor
from controllers.command_queue import CommandQueue, CommandPriority
from utils.exceptions import SafetyViolationError

def test_safety_monitor_stale_frame():
    queue = CommandQueue()
    monitor = SafetyMonitor(queue)
    
    # Override for testing
    monitor.max_stale_ms = 100 
    
    # Initial state
    monitor.notify_frame_received()
    monitor.check_health()
    assert not monitor._e_stop_triggered
    
    # Wait for frame to become stale
    time.sleep(0.15)
    
    with pytest.raises(SafetyViolationError):
        monitor.check_health()
        
    assert monitor._e_stop_triggered
    
    # Verify E-STOP command is in queue
    cmd = queue.pop()
    assert cmd is not None
    assert cmd.command_type == "E_STOP"
    assert cmd.priority == CommandPriority.CRITICAL
