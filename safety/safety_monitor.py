"""
safety/safety_monitor.py
========================
Monitors the robotics pipeline for critical failures (e.g. stale frames).
Triggers Emergency Stops via the CommandQueue.
"""

import time
from typing import Optional

from config.config_manager import config_manager
from controllers.command_queue import CommandQueue, HardwareCommand, CommandPriority
from utils.logger import get_logger
from utils.exceptions import SafetyViolationError

log = get_logger(__name__)


class SafetyMonitor:
    """
    Independent monitor that ensures the system is running safely.
    Checks frame timestamps and triggers E-STOP if the camera dies.
    """

    def __init__(self, command_queue: CommandQueue):
        self.command_queue = command_queue
        self.max_stale_ms = config_manager.get("safety.max_stale_frame_ms", 1500)
        self._auto_recovery: bool = config_manager.get(
            "safety.enable_auto_recovery", True
        )
        self.last_frame_time: Optional[float] = None
        self._e_stop_triggered = False

    def notify_frame_received(self) -> None:
        """Called by the camera or vision pipeline when a new frame is acquired."""
        now = time.time()
        self.last_frame_time = now

        # Auto-recover from a stale-frame E-STOP once frames resume.
        if self._e_stop_triggered and self._auto_recovery:
            log.info("Camera feed resumed — auto-resetting stale-frame E-STOP.")
            self.reset()

    def check_health(self) -> None:
        """
        Evaluates system health. Should be called periodically in the main loop.
        """
        if self._e_stop_triggered:
            return  # Already stopped

        now = time.time()

        # Check camera staleness
        if self.last_frame_time is not None:
            stale_ms = (now - self.last_frame_time) * 1000
            if stale_ms > self.max_stale_ms:
                self.trigger_e_stop(
                    f"Stale frame detected: {stale_ms:.1f}ms without a new frame."
                )

    def trigger_e_stop(self, reason: str) -> None:
        """Injects an E_STOP command with CRITICAL priority.

        When auto-recovery is enabled the exception is suppressed so the main
        loop keeps running and can recover once the camera feed resumes.
        When auto-recovery is disabled the exception propagates and the process
        exits immediately (fail-safe behaviour for unattended deployments).
        """
        if not self._e_stop_triggered:
            log.critical("SAFETY VIOLATION: %s", reason)
            cmd = HardwareCommand(
                priority=CommandPriority.CRITICAL,
                timestamp=time.time(),
                command_type="E_STOP",
                payload={"reason": reason},
            )
            self.command_queue.push(cmd)
            self._e_stop_triggered = True

            if not self._auto_recovery:
                raise SafetyViolationError(f"Emergency Stop Triggered: {reason}")
            log.warning(
                "Auto-recovery is ON — holding E-STOP, waiting for camera to resume."
            )

    def reload_config(self) -> None:
        """Re-read tunable parameters from config. Called by ConfigManager on hot-reload."""
        self.max_stale_ms = config_manager.get("safety.max_stale_frame_ms", 1500)
        self._auto_recovery = config_manager.get("safety.enable_auto_recovery", True)

    def reset(self) -> None:
        """Resets the safety monitor (clears E-STOP state)."""
        self._e_stop_triggered = False
        self.last_frame_time = time.time()
        log.info("SafetyMonitor reset.")
