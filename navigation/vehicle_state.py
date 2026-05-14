"""
navigation/vehicle_state.py
===========================
Robust state machine governing vehicle behavior.
Prevents rapid oscillations.
"""
import time
from enum import Enum, auto
from utils.logger import get_logger
from config.config_manager import config_manager

log = get_logger(__name__)

class State(Enum):
    IDLE = auto()
    DRIVING = auto()
    TURNING = auto()
    RECOVERING = auto()
    STOPPED = auto()

class VehicleStateMachine:
    def __init__(self):
        self._state = State.IDLE
        self._last_transition_time = 0.0
        self.min_duration_s = config_manager.get("navigation.min_state_duration_s", 0.5)

    @property
    def current_state(self) -> State:
        return self._state

    def can_transition(self, new_state: State) -> bool:
        if new_state == self._state:
            return False
            
        # Emergency stop overrides duration lock
        if new_state == State.STOPPED:
            return True
            
        elapsed = time.time() - self._last_transition_time
        if elapsed < self.min_duration_s:
            return False
            
        return True

    def transition_to(self, new_state: State, reason: str = "") -> bool:
        if not self.can_transition(new_state):
            return False
            
        log.info(f"STATE TRANSITION: {self._state.name} -> {new_state.name} | Reason: {reason}")
        self._state = new_state
        self._last_transition_time = time.time()
        return True
