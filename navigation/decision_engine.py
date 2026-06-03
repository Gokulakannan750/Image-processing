"""
navigation/decision_engine.py
==============================
Processes detection results and issues navigation commands.

Field-end logic
---------------
The DecisionEngine handles three marker types:
  NORMAL   → U-turn and continue to next row (normal operation)
  LAST_ROW → Final U-turn. After this, watches for the STOP marker.
  STOP     → After a LAST_ROW turn has been completed, halt the entire
              field operation. No more turns. Field is done.
"""

from typing import Tuple
from config.config_manager import config_manager
from controllers.command_queue import CommandQueue, HardwareCommand, CommandPriority
from detectors.base_detector import DetectionResult, MarkerType
from navigation.vehicle_state import VehicleStateMachine, State
from navigation.recovery_manager import RecoveryManager
from navigation.navigation_filter import NavigationFilter
from navigation.row_tracker import RowTracker
from utils.logger import get_logger

log = get_logger(__name__)


class DecisionEngine:
    def __init__(self, command_queue: CommandQueue):
        self.command_queue = command_queue

        self.state_machine = VehicleStateMachine()
        self.state_machine.transition_to(State.DRIVING, "System Start")

        self.recovery_manager = RecoveryManager(self.state_machine)
        self.nav_filter = NavigationFilter()
        self.row_tracker = RowTracker()
        self._obstacle_blocked: bool = False

    def process_detection(self, result: DetectionResult) -> Tuple[State, float]:
        """
        Takes the DetectionResult from the Vision Pipeline and decides on navigation.
        Returns the current vehicle state and the smoothed steering correction.
        """
        # If the field is already finished, stay stopped — nothing more to do.
        if self.row_tracker.is_finished():
            return self.state_machine.current_state, 0.0

        # Obstacle check — highest priority; halts navigation until path is clear
        if result.has_critical_obstacle:
            if not self._obstacle_blocked:
                self._obstacle_blocked = True
                labels = ", ".join(o.label for o in result.obstacles if o.is_critical)
                self.state_machine.transition_to(State.STOPPED, f"Obstacle: {labels}")
                self._issue_stop_command()
            return self.state_machine.current_state, 0.0

        if self._obstacle_blocked:
            self._obstacle_blocked = False
            self.state_machine.transition_to(State.DRIVING, "Obstacle cleared")
            log.info("Obstacle cleared — resuming navigation.")

        # Notify recovery manager and row tracker whenever a turn-trigger is visible
        if result.should_turn and result.primary_target is not None:
            self.recovery_manager.notify_turn_trigger_visible()
            self.row_tracker.notify_marker_seen(result.primary_target.id)

        # ── STOP marker after last-row turn → end the field operation ──────
        if self.row_tracker.should_stop_field():
            log.critical("FIELD COMPLETE — STOP marker reached after final U-turn.")
            self.row_tracker.notify_field_finished()
            self.state_machine.transition_to(State.STOPPED, "Field operation complete")
            self._issue_field_complete_stop()
            return self.state_machine.current_state, 0.0

        # Check for valid targets and manage recovery state
        has_target = result.has_targets
        current_state = self.recovery_manager.update(has_target)

        steering_correction = 0.0
        primary = result.primary_target

        if primary is not None and current_state == State.DRIVING:
            if primary.center_x is not None:
                cam_center_x = config_manager.get("camera.width", 1280) / 2.0
                raw_error = primary.center_x - cam_center_x
                steering_correction = self.nav_filter.process_alignment(raw_error)

            if primary.is_turn_trigger and self.row_tracker.should_turn():
                marker_type = self.row_tracker.classify_marker(primary.id)
                reason = (
                    f"LAST-ROW marker {primary.id} reached — final turn"
                    if marker_type == MarkerType.LAST_ROW
                    else f"Marker {primary.id} reached"
                )
                if self.state_machine.transition_to(State.TURNING, reason):
                    self._issue_u_turn_command(primary.id)
                    self.row_tracker.notify_turn_completed()

        elif current_state == State.STOPPED:
            self._issue_stop_command()

        return current_state, steering_correction

    # ── Command helpers ────────────────────────────────────────────────────

    def _issue_u_turn_command(self, row_info: str) -> None:
        robot_id = config_manager.get("robot_id", "robot-0")
        cmd = HardwareCommand(
            priority=CommandPriority.NORMAL,
            timestamp=0.0,
            command_type="U_TURN",
            payload={"row_info": row_info, "robot_id": robot_id},
        )
        if self.command_queue.push(cmd):
            log.info("Queued U_TURN command for %s (robot=%s)", row_info, robot_id)

    def _issue_stop_command(self) -> None:
        robot_id = config_manager.get("robot_id", "robot-0")
        cmd = HardwareCommand(
            priority=CommandPriority.CRITICAL,
            timestamp=0.0,
            command_type="E_STOP",
            payload={"reason": "Recovery timeout exceeded", "robot_id": robot_id},
        )
        self.command_queue.push(cmd)

    def _issue_field_complete_stop(self) -> None:
        """Sends a final E-STOP with a 'field complete' reason — no resume."""
        robot_id = config_manager.get("robot_id", "robot-0")
        cmd = HardwareCommand(
            priority=CommandPriority.CRITICAL,
            timestamp=0.0,
            command_type="E_STOP",
            payload={
                "reason": "Field operation complete — all rows finished",
                "robot_id": robot_id,
                "field_complete": True,
            },
        )
        self.command_queue.push(cmd)

    def confirm_turn_complete(self) -> None:
        self.state_machine.transition_to(State.DRIVING, "Turn complete")
        self.nav_filter.reset()
