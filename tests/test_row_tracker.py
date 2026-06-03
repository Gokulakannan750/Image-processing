"""
tests/test_row_tracker.py
Tests for RowTracker: all four schemes, turn counting, finish detection.
"""
import pytest
from navigation.row_tracker import RowTracker


def make_tracker(scheme="simple", total_rows=0, id_map=None):
    from config.config_manager import config_manager
    config_manager._config.setdefault("row_tracker", {})
    config_manager._config["row_tracker"]["scheme"]     = scheme
    config_manager._config["row_tracker"]["total_rows"] = total_rows
    config_manager._config["row_tracker"]["id_map"]     = id_map or {}
    return RowTracker()


# ── Scheme: simple ───────────────────────────────────────────────────────────

def test_simple_aruco_id1_is_row1():
    t = make_tracker(scheme="simple")
    t.notify_marker_seen("ID:1")
    assert t.current_row == 1

def test_simple_aruco_id5_is_row5():
    t = make_tracker(scheme="simple")
    t.notify_marker_seen("ID:5")
    assert t.current_row == 5

def test_simple_id0_treated_as_row1():
    t = make_tracker(scheme="simple")
    t.notify_marker_seen("ID:0")
    assert t.current_row == 1


# ── Scheme: dual ─────────────────────────────────────────────────────────────

def test_dual_even_id_is_top():
    t = make_tracker(scheme="dual")
    t.notify_marker_seen("ID:4")   # 4 // 2 + 1 = row 3, top
    assert t.current_row == 3
    assert t.current_end == "top"

def test_dual_odd_id_is_bottom():
    t = make_tracker(scheme="dual")
    t.notify_marker_seen("ID:5")   # (5-1)//2 + 1 = row 3, bottom
    assert t.current_row == 3
    assert t.current_end == "bottom"

def test_dual_id0_is_row1_top():
    t = make_tracker(scheme="dual")
    t.notify_marker_seen("ID:0")
    assert t.current_row == 1
    assert t.current_end == "top"


# ── Scheme: barcode ───────────────────────────────────────────────────────────

def test_barcode_row3():
    t = make_tracker(scheme="barcode")
    t.notify_marker_seen("Barcode: ROW-3")
    assert t.current_row == 3

def test_barcode_plain_number():
    t = make_tracker(scheme="barcode")
    t.notify_marker_seen("Barcode: 7")
    assert t.current_row == 7

def test_barcode_with_end_label():
    t = make_tracker(scheme="barcode")
    t.notify_marker_seen("Barcode: ROW4-TOP")
    assert t.current_row == 4
    assert t.current_end == "top"

def test_barcode_unparseable_returns_none():
    t = make_tracker(scheme="barcode")
    t.notify_marker_seen("Barcode: HELLO")
    assert t.current_row is None


# ── Scheme: custom_map ────────────────────────────────────────────────────────

def test_custom_map_lookup():
    t = make_tracker(scheme="custom_map", id_map={0: 1, 5: 2, 10: 3})
    t.notify_marker_seen("ID:5")
    assert t.current_row == 2

def test_custom_map_missing_id_returns_none():
    t = make_tracker(scheme="custom_map", id_map={0: 1})
    t.notify_marker_seen("ID:99")
    assert t.current_row is None


# ── Turn counting ─────────────────────────────────────────────────────────────

def test_turns_counted_correctly():
    t = make_tracker()
    t.notify_turn_completed()
    t.notify_turn_completed()
    assert t.turns_completed == 2

def test_rows_completed_every_two_turns():
    t = make_tracker()
    t.notify_turn_completed()
    assert t.rows_completed == 0
    t.notify_turn_completed()
    assert t.rows_completed == 1
    t.notify_turn_completed()
    t.notify_turn_completed()
    assert t.rows_completed == 2


# ── Field finished ────────────────────────────────────────────────────────────

def test_field_finished_when_all_rows_done():
    t = make_tracker(total_rows=2)
    for _ in range(4):   # 4 turns = 2 rows
        t.notify_turn_completed()
    assert t.is_finished() is True

def test_field_not_finished_when_total_rows_zero():
    t = make_tracker(total_rows=0)
    for _ in range(100):
        t.notify_turn_completed()
    assert t.is_finished() is False


# ── Status dict ───────────────────────────────────────────────────────────────

def test_status_dict_keys():
    t = make_tracker()
    s = t.status()
    for key in ("current_row", "current_end", "turns_completed",
                "rows_completed", "total_rows", "is_finished"):
        assert key in s
