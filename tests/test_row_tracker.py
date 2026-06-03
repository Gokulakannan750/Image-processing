"""
tests/test_row_tracker.py
Tests for RowTracker: all four schemes, turn counting,
LAST_ROW / STOP marker classification, and field-finish logic.
"""

from navigation.row_tracker import RowTracker
from detectors.base_detector import MarkerType


def make_tracker(
    scheme="simple", total_rows=0, id_map=None, last_row_id=249, stop_id=248
):
    from config.config_manager import config_manager

    config_manager._config.setdefault("row_tracker", {})
    config_manager._config["row_tracker"]["scheme"] = scheme
    config_manager._config["row_tracker"]["total_rows"] = total_rows
    config_manager._config["row_tracker"]["id_map"] = id_map or {}
    config_manager._config["row_tracker"]["last_row_marker_id"] = last_row_id
    config_manager._config["row_tracker"]["stop_marker_id"] = stop_id
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
    t.notify_marker_seen("ID:4")
    assert t.current_row == 3
    assert t.current_end == "top"


def test_dual_odd_id_is_bottom():
    t = make_tracker(scheme="dual")
    t.notify_marker_seen("ID:5")
    assert t.current_row == 3
    assert t.current_end == "bottom"


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


# ── MarkerType classification — ArUco IDs ────────────────────────────────────


def test_normal_aruco_id_is_normal_type():
    t = make_tracker(last_row_id=249, stop_id=248)
    assert t.classify_marker("ID:5") == MarkerType.NORMAL


def test_last_row_aruco_id_is_last_row_type():
    t = make_tracker(last_row_id=249, stop_id=248)
    assert t.classify_marker("ID:249") == MarkerType.LAST_ROW


def test_stop_aruco_id_is_stop_type():
    t = make_tracker(last_row_id=249, stop_id=248)
    assert t.classify_marker("ID:248") == MarkerType.STOP


# ── MarkerType classification — Barcode text ──────────────────────────────────


def test_barcode_last_row_text():
    t = make_tracker()
    assert t.classify_marker("Barcode: LAST-ROW") == MarkerType.LAST_ROW


def test_barcode_last_text():
    t = make_tracker()
    assert t.classify_marker("Barcode: LAST") == MarkerType.LAST_ROW


def test_barcode_stop_text():
    t = make_tracker()
    assert t.classify_marker("Barcode: STOP") == MarkerType.STOP


def test_barcode_end_text():
    t = make_tracker()
    assert t.classify_marker("Barcode: END") == MarkerType.STOP


def test_barcode_finish_text():
    t = make_tracker()
    assert t.classify_marker("Barcode: FINISH") == MarkerType.STOP


def test_barcode_normal_row_text_is_normal():
    t = make_tracker()
    assert t.classify_marker("Barcode: ROW-3") == MarkerType.NORMAL


# ── Field-end logic ───────────────────────────────────────────────────────────


def test_stop_before_last_row_does_not_end_field():
    t = make_tracker(last_row_id=249, stop_id=248)
    # See STOP marker but last row NOT yet reached
    t.notify_marker_seen("ID:248")
    assert not t.should_stop_field()


def test_last_row_turn_sets_flag():
    t = make_tracker(last_row_id=249, stop_id=248)
    t.notify_marker_seen("ID:249")  # LAST_ROW seen
    t.notify_turn_completed()  # turn fires
    assert t._last_row_reached is True


def test_stop_after_last_row_ends_field():
    t = make_tracker(last_row_id=249, stop_id=248)
    t.notify_marker_seen("ID:249")  # LAST_ROW
    t.notify_turn_completed()
    t.notify_marker_seen("ID:248")  # STOP after last row turn
    assert t.should_stop_field() is True


def test_notify_field_finished_sets_flag():
    t = make_tracker(last_row_id=249, stop_id=248)
    t.notify_marker_seen("ID:249")
    t.notify_turn_completed()
    t.notify_marker_seen("ID:248")
    t.notify_field_finished()
    assert t.is_finished() is True


def test_should_turn_false_for_stop_after_last_row():
    t = make_tracker(last_row_id=249, stop_id=248)
    t.notify_marker_seen("ID:249")
    t.notify_turn_completed()
    t.notify_marker_seen("ID:248")
    # Should NOT turn at the STOP marker
    assert t.should_turn() is False


def test_should_turn_true_for_last_row_marker():
    t = make_tracker(last_row_id=249, stop_id=248)
    t.notify_marker_seen("ID:249")
    assert t.should_turn() is True  # still needs to make the final U-turn


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


# ── Status dict ───────────────────────────────────────────────────────────────


def test_status_dict_keys():
    t = make_tracker()
    s = t.status()
    for key in (
        "current_row",
        "current_end",
        "turns_completed",
        "rows_completed",
        "total_rows",
        "is_finished",
        "last_row_reached",
        "current_marker_type",
    ):
        assert key in s
