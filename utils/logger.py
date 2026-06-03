"""
utils/logger.py
===============
Structured logging architecture with rotating files.
Outputs separate logs for debug trace, runtime events, navigation, and safety.
Also feeds the web dashboard log panel via DashboardLogHandler.
"""

import logging
import logging.handlers
import os
import sys
from typing import Callable, Optional as _Optional

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Module-level robot_id — set once by main() via set_robot_id().
_robot_id: str = "robot-0"


def set_robot_id(robot_id: str) -> None:
    """Call once at startup to stamp every log line with the robot identity."""
    global _robot_id
    _robot_id = robot_id
    # Re-stamp existing handlers by rebuilding the formatter on the root logger
    for handler in logging.root.handlers:
        handler.setFormatter(_make_formatter())


def _make_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt=f"%(asctime)s.%(msecs)03d | {_robot_id} | %(levelname)-8s | [%(name)s] | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = _make_formatter()

    # Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    def add_file_handler(filename: str, handler_level: int, max_mb: int, backup: int):
        fpath = os.path.join(LOG_DIR, filename)
        handler = logging.handlers.RotatingFileHandler(
            fpath, maxBytes=max_mb * 1024 * 1024, backupCount=backup
        )
        handler.setLevel(handler_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Runtime & Debug logs (all modules)
    add_file_handler("runtime.log", logging.INFO, 5, 5)
    add_file_handler("debug.log", logging.DEBUG, 10, 3)

    # Specific specialized logs
    if "navigation" in name or "controller" in name:
        add_file_handler("navigation.log", logging.INFO, 5, 5)

    if "safety" in name or "camera" in name:
        add_file_handler("safety.log", logging.INFO, 5, 5)

    return logger


def get_logger(name: str) -> logging.Logger:
    return setup_logger(name, level=logging.INFO)


# ── Dashboard log handler ──────────────────────────────────────────────────

class DashboardLogHandler(logging.Handler):
    """Forwards log records to the web dashboard's log buffer.

    Call attach_to_dashboard(push_fn) once after dashboard_state is created.
    push_fn should be dashboard_state.push_log.
    """

    _push_fn: _Optional[Callable[[str], None]] = None

    def emit(self, record: logging.LogRecord) -> None:
        if self._push_fn is None:
            return
        try:
            # Short format for the UI: level + logger name + message
            msg = f"[{record.levelname}] {record.name}: {self.format(record).split(' | ')[-1]}"
            self._push_fn(msg)
        except Exception:
            pass  # never let a logging handler crash the app


_dashboard_handler: _Optional[DashboardLogHandler] = None


def attach_dashboard_log_handler(push_fn: Callable[[str], None]) -> None:
    """Wire all loggers into the dashboard log panel.

    Call this once in main.py after dashboard_state is initialised:
        from utils.logger import attach_dashboard_log_handler
        attach_dashboard_log_handler(dashboard_state.push_log)
    """
    global _dashboard_handler
    if _dashboard_handler is not None:
        return  # already attached

    handler = DashboardLogHandler(level=logging.INFO)
    handler._push_fn = push_fn
    handler.setFormatter(logging.Formatter("%(message)s"))  # raw message only
    _dashboard_handler = handler

    # Attach to the root logger so every module's log goes to the dashboard
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)  # ensure root passes records down
