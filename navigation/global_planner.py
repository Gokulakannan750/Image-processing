"""
navigation/global_planner.py
============================
Global map-based path planning with vacuum-style row coverage.
"""

import json
from typing import List, Optional
from dataclasses import dataclass
import math

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Waypoint:
    x: float
    y: float
    is_row_entry: bool = False
    is_row_exit: bool = False
    row_index: int = -1


class GlobalPlanner:
    def __init__(self):
        self.waypoints: List[Waypoint] = []
        self.current_waypoint_idx = 0
        self.farm_map = None

    def load_farm_map(self, file_path: str):
        """
        Loads a map (e.g. JSON format with field boundaries and rows).
        For now, we simulate loading by generating a generic map.
        """
        try:
            with open(file_path, "r") as f:
                self.farm_map = json.load(f)
            log.info(f"Loaded farm map from {file_path}")
        except FileNotFoundError:
            log.warning(f"Map {file_path} not found. Using default generated map.")
            self._generate_mock_map()

    def _generate_mock_map(self):
        """Generates a mock map with 6 rows of 100 meters length."""
        self.farm_map = {"origin_lat": 0.0, "origin_lon": 0.0, "rows": []}

        row_width = 2.0  # distance between rows in meters
        num_rows = 6
        row_length = 100.0

        # Rows go from y=0 to y=100
        for i in range(num_rows):
            x_pos = i * row_width
            self.farm_map["rows"].append(
                {
                    "index": i,
                    "start": {"x": x_pos, "y": 0.0},
                    "end": {"x": x_pos, "y": row_length},
                }
            )

    def generate_vacuum_path(self):
        """
        Generates a vacuum-style coverage path:
        Start in the middle, cover right side, return to centre, cover left side.
        """
        if not self.farm_map or "rows" not in self.farm_map:
            log.error("No valid map loaded to generate path.")
            return

        rows = self.farm_map["rows"]
        if not rows:
            return

        num_rows = len(rows)
        mid_idx = num_rows // 2

        # Split rows into right and left relative to the middle
        right_rows = rows[mid_idx:]  # Middle to rightmost
        left_rows = rows[:mid_idx][
            ::-1
        ]  # Middle-1 to leftmost, reversed so we start from middle-1

        self.waypoints = []

        # Helper to add a row to the path
        def add_row_to_path(row_data, traverse_up: bool):
            idx = row_data["index"]
            sx, sy = row_data["start"]["x"], row_data["start"]["y"]
            ex, ey = row_data["end"]["x"], row_data["end"]["y"]

            if traverse_up:
                self.waypoints.append(
                    Waypoint(sx, sy, is_row_entry=True, row_index=idx)
                )
                self.waypoints.append(Waypoint(ex, ey, is_row_exit=True, row_index=idx))
            else:
                self.waypoints.append(
                    Waypoint(ex, ey, is_row_entry=True, row_index=idx)
                )
                self.waypoints.append(Waypoint(sx, sy, is_row_exit=True, row_index=idx))

        # 1. Start at the middle and cover the right side
        traverse_up = True
        for row in right_rows:
            add_row_to_path(row, traverse_up)
            traverse_up = not traverse_up  # Alternate direction for next row (U-turn)

        # 2. Transit back to the start of the left rows
        # The last row on the right might leave us at the top or bottom.
        # We need to navigate to the entry point of the first left row.
        # The add_row_to_path will handle the entry/exit points, but in a real planner
        # we would add intermediate transit waypoints to avoid crossing rows diagonally.

        # 3. Cover the left side
        for row in left_rows:
            add_row_to_path(row, traverse_up)
            traverse_up = not traverse_up

        self.current_waypoint_idx = 0
        log.info(f"Generated vacuum path with {len(self.waypoints)} waypoints.")

    def get_current_waypoint(self) -> Optional[Waypoint]:
        if self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
        return None

    def advance_waypoint(self):
        self.current_waypoint_idx += 1
        if self.current_waypoint_idx >= len(self.waypoints):
            log.info("Reached end of global path.")

    def is_path_complete(self) -> bool:
        return self.current_waypoint_idx >= len(self.waypoints)

    def distance_to_waypoint(self, current_x: float, current_y: float) -> float:
        wp = self.get_current_waypoint()
        if not wp:
            return float("inf")
        return math.hypot(wp.x - current_x, wp.y - current_y)
