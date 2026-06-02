"""
controllers/machine_controller.py
===================================
Hardware abstraction for the agricultural machine's CAN-bus controller.
Consumes commands from the CommandQueue.
"""
import time
from typing import Optional  # noqa: F401 — used in type hints throughout

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

    # CAN arbitration IDs for safety messages (configurable)
    _ESTOP_CMD_ID: int = 0x000   # Highest priority on standard CAN

    def _execute_e_stop(self) -> None:
        """Executes an emergency stop command immediately."""
        log.critical("!!! EXECUTING EMERGENCY STOP !!!")
        self.command_queue.clear()

        if self._bus is None:
            log.warning("MOCK E-STOP — no CAN bus connected.")
            return

        estop_id = config_manager.get("controller.estop_command_id", self._ESTOP_CMD_ID)
        estop_data = config_manager.get(
            "controller.estop_command_data", [0xFF, 0xFF, 0, 0, 0, 0, 0, 0]
        )

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                msg = can.Message(
                    arbitration_id=estop_id,
                    data=estop_data,
                    is_extended_id=False,
                )
                self._bus.send(msg, timeout=0.05)
                log.critical(
                    "E-STOP CAN sent (attempt %d/%d) — ID=%s data=%s",
                    attempt, max_attempts, hex(estop_id), estop_data,
                )
                return  # Success — exit after first confirmed send
            except Exception as exc:
                log.error("E-STOP CAN send failed (attempt %d/%d): %s", attempt, max_attempts, exc)
                time.sleep(0.01 * attempt)  # brief exponential backoff

        log.critical("E-STOP CAN message could not be delivered after %d attempts.", max_attempts)

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
            max_attempts = 3
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    msg = can.Message(
                        arbitration_id=cmd_id,
                        data=cmd_data,
                        is_extended_id=False,
                    )
                    self._bus.send(msg, timeout=0.05)
                    log.info(
                        "CAN sent (attempt %d/%d) — ID=%s  data=%s",
                        attempt, max_attempts, hex(cmd_id), cmd_data,
                    )
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    log.warning("CAN send failed (attempt %d/%d): %s", attempt, max_attempts, exc)
                    time.sleep(0.01 * attempt)
            if last_exc is not None:
                raise ControllerError(f"Failed to send U-TURN CAN message after {max_attempts} attempts: {last_exc}")
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
