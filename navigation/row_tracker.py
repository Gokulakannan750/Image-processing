"""
navigation/row_tracker.py
=========================
Tracks which crop row the machine is currently in and detects special
end-of-field markers so the robot stops automatically when all rows are done.

Field-size-independent operation
---------------------------------
The farmer does NOT need to configure how many rows the field has.
They simply print two special markers:

  LAST-ROW marker  — placed on the pole at the far end of the LAST row.
                     When the machine reads this, it makes one final U-turn
                     and then watches for the STOP marker.

  STOP marker      — placed on the pole at the near end (starting end) of
                     the last row.  When the machine reads this after the
                     final turn, it halts completely. Field is done.

Marker identification
----------------------
ArUco:
  - last_row_marker_id (default 249) → LAST_ROW type
  - stop_marker_id     (default 248) → STOP type
  - all other IDs                    → NORMAL type

Barcode / QR text:
  - Text contains "LAST", "LAST-ROW", "LAST ROW" → LAST_ROW
  - Text contains "STOP", "END", "FINISH"         → STOP
  - Anything else                                  → NORMAL

Row numbering schemes (same as before):
  "simple"     — ArUco ID = row number
  "dual"       — even ID = top end, odd ID = bottom end
  "barcode"    — text encodes row (e.g. "ROW-3", "ROW4-TOP")
  "custom_map" — explicit {id: row} mapping in config
"""

import re
from typing import Dict, Optional

from config.config_manager import config_manager
from detectors.base_detector import MarkerType
from utils.logger import get_logger

log = get_logger(__name__)


class RowTracker:
    """
    Parses a marker ID or barcode text into a row number and MarkerType,
    and tracks turns/rows completed across the session.
    """

    def __init__(self) -> None:
        self._reload_settings()

        self.current_row: Optional[int] = None
        self.current_end: Optional[str] = None
        self.current_marker_id: Optional[str] = None
        self.current_marker_type: MarkerType = MarkerType.NORMAL
        self.turns_completed: int = 0
        self.rows_completed: int = 0
        self._last_row_reached: bool = False  # True after LAST_ROW marker turn
        self.field_finished: bool = False

        log.info(
            "RowTracker initialised — scheme=%s, last_row_id=%s, stop_id=%s",
            self._scheme,
            self._last_row_id if self._last_row_id >= 0 else "barcode-only",
            self._stop_id if self._stop_id >= 0 else "barcode-only",
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def classify_marker(self, marker_id: str) -> MarkerType:
        """
        Returns the MarkerType for a given marker string.
        Called by the ArUco / Barcode detectors (and by DecisionEngine).
        """
        raw_int = self._extract_int(marker_id)

        # ArUco special IDs take highest priority
        if raw_int is not None:
            if self._last_row_id >= 0 and raw_int == self._last_row_id:
                return MarkerType.LAST_ROW
            if self._stop_id >= 0 and raw_int == self._stop_id:
                return MarkerType.STOP

        # Barcode / QR text keywords
        text = marker_id.replace("Barcode:", "").replace("ID:", "").strip().upper()
        if re.search(r"\bLAST[-\s]?ROW\b|\bLAST\b", text):
            return MarkerType.LAST_ROW
        if re.search(r"\bSTOP\b|\bEND\b|\bFINISH\b", text):
            return MarkerType.STOP

        return MarkerType.NORMAL

    def notify_marker_seen(self, marker_id: str) -> None:
        """
        Called by DecisionEngine each time a turn-trigger marker is detected.
        Updates current row and marker type.
        """
        self.current_marker_id = marker_id
        self.current_marker_type = self.classify_marker(marker_id)
        row, end = self._parse_row(marker_id)

        if self.current_marker_type == MarkerType.LAST_ROW:
            log.info("ROW TRACKER: LAST-ROW marker detected — '%s'", marker_id)
        elif self.current_marker_type == MarkerType.STOP:
            log.info("ROW TRACKER: STOP marker detected — '%s'", marker_id)

        if row is not None and row != self.current_row:
            log.info(
                "ROW TRACKER: now at Row %d (%s end) — marker '%s'",
                row,
                end or "?",
                marker_id,
            )
        if row is not None:
            self.current_row = row
            self.current_end = end

    def should_stop_field(self) -> bool:
        """
        Returns True when the machine should halt the entire field operation
        (i.e. when a STOP marker is seen after the LAST_ROW turn was completed).
        """
        return self.current_marker_type == MarkerType.STOP and self._last_row_reached

    def should_turn(self) -> bool:
        """
        Returns True when the machine should still make a U-turn
        (NORMAL or LAST_ROW markers, not STOP).
        """
        return self.current_marker_type in (MarkerType.NORMAL, MarkerType.LAST_ROW)

    def notify_turn_completed(self) -> None:
        """Called by DecisionEngine after each U-turn fires."""
        self.turns_completed += 1

        # If the turn we just made was triggered by a LAST_ROW marker,
        # set the flag so the next STOP marker halts the field.
        if self.current_marker_type == MarkerType.LAST_ROW:
            self._last_row_reached = True
            log.info(
                "ROW TRACKER: final U-turn complete — "
                "watching for STOP marker to end field operation."
            )

        # Every two turns = one row completed (for informational counting)
        if self.turns_completed % 2 == 0:
            self.rows_completed += 1
            log.info("ROW TRACKER: completed row %d", self.rows_completed)

    def notify_field_finished(self) -> None:
        """Called by DecisionEngine when the STOP marker halts the field."""
        self.field_finished = True
        log.info(
            "ROW TRACKER: FIELD COMPLETE — %d rows, %d turns total.",
            self.rows_completed,
            self.turns_completed,
        )

    def is_finished(self) -> bool:
        return self.field_finished

    def status(self) -> dict:
        return {
            "current_row": self.current_row,
            "current_end": self.current_end,
            "current_marker_id": self.current_marker_id,
            "current_marker_type": self.current_marker_type.name,
            "last_row_reached": self._last_row_reached,
            "turns_completed": self.turns_completed,
            "rows_completed": self.rows_completed,
            "total_rows": None,  # not needed — marker-driven
            "is_finished": self.field_finished,
        }

    # ── Row number parsing ─────────────────────────────────────────────────

    def _parse_row(self, marker_id: str) -> tuple:
        """Returns (row_number, end_label) or (None, None)."""
        scheme = self._scheme
        raw_int = self._extract_int(marker_id)

        if scheme == "simple" and raw_int is not None:
            row = max(1, raw_int) if raw_int > 0 else 1
            return row, None

        if scheme == "dual" and raw_int is not None:
            row = (raw_int // 2) + 1
            end = "top" if raw_int % 2 == 0 else "bottom"
            return row, end

        if scheme == "custom_map" and raw_int is not None:
            row = self._id_map.get(raw_int)
            if row is None:
                log.warning("RowTracker custom_map: no entry for ID %d", raw_int)
            return row, None

        if scheme == "barcode":
            text = marker_id.replace("Barcode:", "").strip()
            m = re.search(r"(\d+)", text, re.IGNORECASE)
            if m:
                row = int(m.group(1))
                end_m = re.search(
                    r"(top|bottom|north|south|start|end)", text, re.IGNORECASE
                )
                end = end_m.group(1).lower() if end_m else None
                return row, end
            return None, None

        return None, None

    def _extract_int(self, marker_id: str) -> Optional[int]:
        m = re.search(r"(\d+)", marker_id)
        return int(m.group(1)) if m else None

    def _reload_settings(self) -> None:
        self._scheme = config_manager.get("row_tracker.scheme", "simple")
        self._last_row_id = int(
            config_manager.get("row_tracker.last_row_marker_id", 249)
        )
        self._stop_id = int(config_manager.get("row_tracker.stop_marker_id", 248))
        raw_map = config_manager.get("row_tracker.id_map", {})
        self._id_map: Dict[int, int] = {int(k): int(v) for k, v in raw_map.items()}

    def reload_config(self) -> None:
        """Re-read config on hot-reload."""
        self._reload_settings()
