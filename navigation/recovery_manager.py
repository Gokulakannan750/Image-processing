"""
navigation/recovery_manager.py
==============================
Handles lost-marker situations and transitions vehicle state safely.

Recovery is ONLY triggered when:
  - A turn-trigger marker was recently being tracked, AND
  - It has now disappeared from frame.

Normal driving without any markers in view is perfectly valid and
never triggers recovery.
"""
import time
from navigation.vehicle_state import VehicleStateMachine, State
from config.config_manager import config_manager
from utils.logger import get_logger

log = get_logger(__name__)

class RecoveryManager:
    def __init__(self, state_machine: VehicleStateMachine):
        self.state_machine = state_machine
        self.recovery_timeout_s = config_manager.get("navigation.recovery_timeout_s", 2.0)
        self.lost_time = None
        # Only track recovery when we were recently homing in on a turn-trigger
        self._was_tracking_trigger = False

    def notify_turn_trigger_visible(self) -> None:
        """Called by DecisionEngine when a close turn-trigger marker is in frame."""
        self._was_tracking_trigger = True
        self.lost_time = None

    def update(self, has_target: bool) -> State:
        current_state = self.state_machine.current_state

        if has_target:
            self.lost_time = None
            # Come back to DRIVING from RECOVERING or STOPPED
            if current_state in [State.RECOVERING, State.STOPPED]:
                self.state_machine.transition_to(State.DRIVING, "Target re-acquired")
            return self.state_machine.current_state

        # ── No target in frame ─────────────────────────────────────────────────
        # If we are in TURNING, don't interfere — let the controller signal
        # turn completion via confirm_turn_complete().
        if current_state == State.TURNING:
            return current_state

        # Only enter RECOVERING if we were actively tracking a close marker
        # that has now vanished. Normal forward travel without any marker is fine.
        if not self._was_tracking_trigger:
            # Nothing to recover from — just keep DRIVING silently.
            return current_state

        # We were tracking a turn-trigger marker that just disappeared.
        if self.lost_time is None:
            self.lost_time = time.time()

        elapsed = time.time() - self.lost_time

        if current_state == State.DRIVING:
            self.state_machine.transition_to(State.RECOVERING, "Turn-trigger marker lost")

        elif current_state == State.RECOVERING:
            if elapsed > self.recovery_timeout_s:
                self.state_machine.transition_to(State.STOPPED, "Recovery timeout exceeded")

        return self.state_machine.current_state

    def reset(self) -> None:
        """Reset after a successful turn or after STOPPED → resume scanning."""
        self.lost_time = None
        self._was_tracking_trigger = False
        # Return to DRIVING so the robot resumes forward travel immediately.
        if self.state_machine.current_state != State.DRIVING:
            self.state_machine.transition_to(State.DRIVING, "RecoveryManager reset")
