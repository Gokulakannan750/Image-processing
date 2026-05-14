"""
controllers/machine_controller.py
===================================
Hardware abstraction for the agricultural machine's CAN-bus controller.
Consumes commands from the CommandQueue.
"""
import time
from typing import Optional

from config.config_manager import config_manager
from controllers.command_queue import CommandQueue, HardwareCommand, CommandPriority
from utils.logger import get_logger
from utils.exceptions import ControllerError

log = get_logger(__name__)

try:
    import can
    _CAN_AVAILABLE = True
except ImportError:
    log.warning("python-can not installed — running in MOCK mode (console output only).")
    _CAN_AVAILABLE = False


class MachineController:
    """
    Communicates with the machine's steering controller via CAN bus.
    Pulls commands from a CommandQueue.
    """

    def __init__(self, command_queue: CommandQueue) -> None:
        self.command_queue = command_queue
        
        # Load from config manager
        interface = config_manager.get("controller.can_interface", "virtual")
        channel = config_manager.get("controller.can_channel", "can0")
        bitrate = config_manager.get("controller.can_bitrate", 500000)
        
        self.is_turning: bool = False
        self.last_turn_time: float = 0.0
        self.cooldown: float = config_manager.get("navigation.turn_cooldown_s", 10.0)

        self._bus = None
        if _CAN_AVAILABLE:
            try:
                self._bus = can.interface.Bus(
                    bustype=interface, channel=channel, bitrate=bitrate
                )
                log.info("CAN bus initialized on %s (%s) @ %d bps.", channel, interface, bitrate)
            except Exception as exc:
                log.error("Failed to initialize CAN bus: %s", exc)
                log.warning("Falling back to MOCK mode.")

    def process_queue(self) -> None:
        """
        Polls the command queue for the next command and executes it.
        Should be called frequently in the main loop or a dedicated controller thread.
        """
        # Pop command without blocking
        cmd = self.command_queue.pop(timeout=None)
        if not cmd:
            return

        log.debug("Processing command: %s (Priority: %s)", cmd.command_type, cmd.priority.name)

        if cmd.command_type == "E_STOP":
            self._execute_e_stop()
        elif cmd.command_type == "U_TURN":
            self._execute_u_turn(cmd.payload.get("row_info", "Unknown"))
        else:
            log.warning("Unknown command type: %s", cmd.command_type)

    def _execute_e_stop(self) -> None:
        """Executes an emergency stop command immediately."""
        log.critical("!!! EXECUTING EMERGENCY STOP !!!")
        self.command_queue.clear()  # Clear pending commands
        # TODO: Send actual E-STOP CAN message
        if self._bus is not None:
             pass # Implement CAN E-STOP message here

    def _execute_u_turn(self, row_info: str) -> bool:
        now = time.time()
        if now - self.last_turn_time < self.cooldown:
            log.debug("U-turn request ignored — still in cooldown.")
            return False

        log.info("=" * 50)
        log.info("!!! INITIATING U-TURN MANOEUVRE !!!")
        log.info("Reason: %s", row_info)

        cmd_id = config_manager.get("controller.turn_command_id", 0x123)
        cmd_data = config_manager.get("controller.turn_command_data", [1,0,0,0,0,0,0,0])

        if self._bus is not None:
            try:
                msg = can.Message(
                    arbitration_id=cmd_id,
                    data=cmd_data,
                    is_extended_id=False,
                )
                self._bus.send(msg)
                log.info("CAN sent — ID=%s  data=%s", hex(cmd_id), cmd_data)
            except Exception as exc:
                log.error("CAN send failed: %s", exc)
                raise ControllerError(f"Failed to send CAN message: {exc}")
        else:
            log.info("MOCK — would send CAN ID=%s  data=%s", hex(cmd_id), cmd_data)

        log.info("=" * 50)
        self.is_turning = True
        self.last_turn_time = now
        return True

    def shutdown(self) -> None:
        if self._bus is not None:
            try:
                self._bus.shutdown()
                log.info("CAN bus shut down cleanly.")
            except Exception as exc:
                log.error("Error during CAN shutdown: %s", exc)
            finally:
                self._bus = None

    def __del__(self) -> None:
        self.shutdown()
