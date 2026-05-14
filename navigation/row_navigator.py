"""
navigation/row_navigator.py
============================
Tracks row-level state for the autonomous field traversal mission.
"""
from enum import Enum, auto
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)


class NavigationCommand(Enum):
    """Commands returned to the DecisionEngine by :class:`RowNavigator`."""
    CONTINUE = auto()
    """Keep moving forward — no action needed."""

    U_TURN = auto()
    """End of row reached — execute a U-turn and advance to the next row."""


class RowNavigator:
    """
    Finite-state machine for row-by-row field navigation.
    """

    def __init__(self, total_rows: Optional[int] = None) -> None:
        self.total_rows = total_rows
        self.current_row: int = 1
        self.rows_completed: int = 0
        self._pending_turn: bool = False

        log.info(
            "RowNavigator initialised — total rows: %s",
            total_rows if total_rows else "unlimited",
        )

    def on_detection(self, row_info: Optional[str] = None) -> NavigationCommand:
        if self._pending_turn:
            return NavigationCommand.CONTINUE

        if self.is_mission_complete:
            log.info("Mission complete — all %d rows processed.", self.total_rows)
            return NavigationCommand.CONTINUE

        log.info(
            "End-of-row detected on row %d — %s",
            self.current_row,
            row_info or "no detail",
        )
        self._pending_turn = True
        return NavigationCommand.U_TURN

    def confirm_turn_complete(self) -> None:
        if self._pending_turn:
            self.rows_completed += 1
            self.current_row += 1
            self._pending_turn = False
            log.info(
                "U-turn complete — now on row %d  (%d rows done).",
                self.current_row,
                self.rows_completed,
            )

    @property
    def is_mission_complete(self) -> bool:
        if self.total_rows is None:
            return False
        return self.rows_completed >= self.total_rows

    @property
    def progress(self) -> str:
        total = str(self.total_rows) if self.total_rows else "?"
        return f"Row {self.current_row} / {total}"
