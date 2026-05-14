"""
utils/communication.py
========================
DEPRECATED: Hardware abstraction for the agricultural machine's CAN-bus controller.

This module is maintained for backwards compatibility.
Please use `controllers.machine_controller` instead.
"""
from controllers.machine_controller import MachineController
from utils.logger import get_logger

log = get_logger(__name__)

log.warning(
    "Importing from `utils.communication` is deprecated. "
    "Please update imports to use `controllers.machine_controller`."
)

__all__ = ["MachineController"]
