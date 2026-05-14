import time
from typing import Tuple
from config.config_manager import config_manager
from controllers.command_queue import CommandQueue, HardwareCommand, CommandPriority
from detectors.base_detector import DetectionResult
from navigation.vehicle_state import VehicleStateMachine, State
from navigation.recovery_manager import RecoveryManager
from navigation.navigation_filter import NavigationFilter
from utils.logger import get_logger

log = get_logger(__name__)

class DecisionEngine:
    def __init__(self, command_queue: CommandQueue):
        self.command_queue = command_queue
        
        self.state_machine = VehicleStateMachine()
        self.state_machine.transition_to(State.DRIVING, "System Start")
        
        self.recovery_manager = RecoveryManager(self.state_machine)
        self.nav_filter = NavigationFilter()

    def process_detection(self, result: DetectionResult) -> Tuple[State, float]:
        """
        Takes the DetectionResult from the Vision Pipeline and decides on navigation.
        Returns the current vehicle state and the smoothed steering correction.
        """
        # Notify recovery manager if we see a turn trigger
        if result.should_turn:
            self.recovery_manager.notify_turn_trigger_visible()

        # 1. Check for valid targets and manage recovery state
        has_target = result.has_targets
        current_state = self.recovery_manager.update(has_target)
        
        steering_correction = 0.0
        primary = result.primary_target
        
        if primary is not None and current_state == State.DRIVING:
            if primary.center_x is not None:
                cam_center_x = config_manager.get("camera.width", 1280) / 2.0 
                raw_error = primary.center_x - cam_center_x
                steering_correction = self.nav_filter.process_alignment(raw_error)
            
            if primary.is_turn_trigger:
                if self.state_machine.transition_to(State.TURNING, f"Marker {primary.id} reached"):
                    self._issue_u_turn_command(primary.id)
                    
        elif current_state == State.STOPPED:
            self._issue_stop_command()

        return current_state, steering_correction

    def _issue_u_turn_command(self, row_info: str) -> None:
        cmd = HardwareCommand(
            priority=CommandPriority.NORMAL,
            timestamp=0.0,
            command_type="U_TURN",
            payload={"row_info": row_info}
        )
        if self.command_queue.push(cmd):
            log.info("Queued U_TURN command for %s", row_info)

    def _issue_stop_command(self) -> None:
        cmd = HardwareCommand(
            priority=CommandPriority.CRITICAL,
            timestamp=0.0,
            command_type="E_STOP",
            payload={"reason": "Recovery timeout exceeded"}
        )
        self.command_queue.push(cmd)

    def confirm_turn_complete(self) -> None:
        self.state_machine.transition_to(State.DRIVING, "Turn complete")
        self.nav_filter.reset()
