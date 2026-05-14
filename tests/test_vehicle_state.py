import time
from navigation.vehicle_state import VehicleStateMachine, State
from config.config_manager import config_manager

def test_vehicle_state_machine_init():
    sm = VehicleStateMachine()
    assert sm.current_state == State.IDLE

def test_state_cooldown():
    sm = VehicleStateMachine()
    
    # Force cooldown to 0.1s for testing
    sm.min_duration_s = 0.1
    
    # Transition to DRIVING
    assert sm.transition_to(State.DRIVING)
    assert sm.current_state == State.DRIVING
    
    # Try transitioning immediately, should fail
    assert not sm.transition_to(State.TURNING)
    assert sm.current_state == State.DRIVING
    
    # Wait for cooldown
    time.sleep(0.15)
    
    # Now it should work
    assert sm.transition_to(State.TURNING)
    assert sm.current_state == State.TURNING

def test_emergency_stop_override():
    sm = VehicleStateMachine()
    sm.min_duration_s = 5.0  # Very long cooldown
    
    assert sm.transition_to(State.DRIVING)
    
    # E-STOP should ignore cooldown
    assert sm.transition_to(State.STOPPED)
    assert sm.current_state == State.STOPPED
