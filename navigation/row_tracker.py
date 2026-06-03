"""
navigation/row_tracker.py
=========================
Tracks which crop row the machine is currently in by reading the marker ID
at each row end.

How it works
------------
Each ArUco marker printed on the row-end poles has a unique ID number (0–249).
You assign specific IDs to specific rows when you print and place the markers.

Two supported naming schemes (configured in default.yaml):

  Scheme A — "simple"  (one marker per row end, same ID used at both ends)
    Marker ID 1  →  Row 1
    Marker ID 2  →  Row 2
    ...
    Marker ID N  →  Row N

  Scheme B — "dual"  (separate IDs for top and bottom of each row)
    Even ID (0, 2, 4…) →  top end of row  (ID/2 + 1)
    Odd  ID (1, 3, 5…) →  bottom end of row  ((ID-1)/2 + 1)
    e.g.  ID 0 → Row 1 top, ID 1 → Row 1 bottom,
          ID 2 → Row 2 top, ID 3 → Row 2 bottom

  Scheme C — "barcode"  (barcode/QR text encodes the row directly)
    Barcode text "ROW-3" or "3" or "ROW3"  →  Row 3
    Anything not parseable  →  unknown

  Scheme D — "custom_map"  (explicit ID→row mapping in config)
    config: row_tracker.id_map: {0: 1, 5: 2, 10: 3, ...}

The current row, total rows completed, and turn count are exposed via
RowTracker.status() for the dashboard and logs.
"""
import re
from typing import Dict, Optional

from config.config_manager import config_manager
from utils.logger import get_logger

log = get_logger(__name__)


class RowTracker:
    """
    Parses a marker ID or barcode text into a row number and tracks
    how many rows have been completed across the session.
    """

    def __init__(self) -> None:
        self._scheme: str = config_manager.get("row_tracker.scheme", "simple")
        self._total_rows: int = config_manager.get("row_tracker.total_rows", 0)
        self._id_map: Dict[int, int] = config_manager.get("row_tracker.id_map", {})
        # Convert string keys (YAML loads int keys as ints, but be safe)
        self._id_map = {int(k): int(v) for k, v in self._id_map.items()}

        self.current_row: Optional[int] = None   # row number (1-based)
        self.current_end: Optional[str] = None   # "top" | "bottom" | None
        self.current_marker_id: Optional[str] = None
        self.turns_completed: int = 0
        self.rows_completed: int = 0

        log.info(
            "RowTracker initialised — scheme=%s, total_rows=%d",
            self._scheme, self._total_rows,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def notify_marker_seen(self, marker_id: str) -> None:
        """
        Called by the DecisionEngine each time a turn-trigger marker is detected.
        marker_id is the string from DetectionTarget.id, e.g.:
          ArUco  →  "ID:3"
          Barcode → "Barcode: ROW-3"
        """
        self.current_marker_id = marker_id
        row, end = self._parse(marker_id)
        if row is not None:
            if row != self.current_row:
                log.info(
                    "ROW TRACKER: now at Row %d (%s end) — marker '%s'",
                    row, end or "?", marker_id,
                )
            self.current_row = row
            self.current_end = end

    def notify_turn_completed(self) -> None:
        """Called by the DecisionEngine after each U-turn fires."""
        self.turns_completed += 1
        # Every two turns = one row completed
        if self.turns_completed % 2 == 0:
            self.rows_completed += 1
            log.info(
                "ROW TRACKER: completed row %d of %s",
                self.rows_completed,
                str(self._total_rows) if self._total_rows else "?",
            )
        if self._total_rows and self.rows_completed >= self._total_rows:
            log.info("ROW TRACKER: all %d rows completed!", self._total_rows)

    def is_finished(self) -> bool:
        """Returns True when all configured rows have been completed."""
        if not self._total_rows:
            return False
        return self.rows_completed >= self._total_rows

    def status(self) -> dict:
        """Return a dict suitable for dashboard /status and logs."""
        return {
            "current_row": self.current_row,
            "current_end": self.current_end,
            "current_marker_id": self.current_marker_id,
            "turns_completed": self.turns_completed,
            "rows_completed": self.rows_completed,
            "total_rows": self._total_rows or None,
            "is_finished": self.is_finished(),
        }

    # ── Parsing ────────────────────────────────────────────────────────────

    def _parse(self, marker_id: str) -> tuple:
        """Returns (row_number, end_label) or (None, None) if unparseable."""
        scheme = self._scheme

        # ── ArUco ID-based schemes ─────────────────────────────────────
        raw_int = self._extract_int(marker_id)

        if scheme == "simple" and raw_int is not None:
            # ID 1 → Row 1, ID 2 → Row 2, …  (ID 0 treated as Row 1)
            row = max(1, raw_int) if raw_int > 0 else 1
            return row, None

        if scheme == "dual" and raw_int is not None:
            # Even → top end, Odd → bottom end
            row = (raw_int // 2) + 1
            end = "top" if raw_int % 2 == 0 else "bottom"
            return row, end

        if scheme == "custom_map" and raw_int is not None:
            row = self._id_map.get(raw_int)
            if row is not None:
                return row, None
            log.warning(
                "RowTracker custom_map: no entry for ID %d — row unknown", raw_int
            )
            return None, None

        # ── Barcode / QR text scheme ──────────────────────────────────
        if scheme == "barcode":
            # Strip "Barcode: " prefix added by barcode_detector.py
            text = marker_id.replace("Barcode:", "").strip()
            # Try patterns: "ROW-3", "ROW3", "row 3", "3"
            m = re.search(r"(\d+)", text, re.IGNORECASE)
            if m:
                row = int(m.group(1))
                end_m = re.search(r"(top|bottom|north|south|start|end)", text, re.IGNORECASE)
                end = end_m.group(1).lower() if end_m else None
                return row, end
            log.warning("RowTracker barcode: could not parse row from '%s'", text)
            return None, None

        log.debug("RowTracker: scheme '%s' could not parse '%s'", scheme, marker_id)
        return None, None

    def _extract_int(self, marker_id: str) -> Optional[int]:
        """Pulls the integer out of strings like 'ID:3' or 'ID:12'."""
        m = re.search(r"(\d+)", marker_id)
        return int(m.group(1)) if m else None

    def reload_config(self) -> None:
        """Re-read config on hot-reload."""
        self._scheme = config_manager.get("row_tracker.scheme", "simple")
        self._total_rows = config_manager.get("row_tracker.total_rows", 0)
        raw_map = config_manager.get("row_tracker.id_map", {})
        self._id_map = {int(k): int(v) for k, v in raw_map.items()}
