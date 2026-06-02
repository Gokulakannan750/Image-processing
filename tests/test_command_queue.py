"""
tests/test_command_queue.py
Tests for CommandQueue: priority ordering, push/pop, full-queue handling, clear.
"""
import pytest
from controllers.command_queue import CommandQueue, HardwareCommand, CommandPriority


def make_cmd(priority: CommandPriority, cmd_type: str = "TEST") -> HardwareCommand:
    return HardwareCommand(priority=priority, timestamp=0.0, command_type=cmd_type)


# ── Basic push / pop ─────────────────────────────────────────────────────────

def test_push_and_pop_single():
    q = CommandQueue()
    cmd = make_cmd(CommandPriority.NORMAL)
    assert q.push(cmd)
    result = q.pop(timeout=None)
    assert result is not None
    assert result.command_type == "TEST"


def test_pop_empty_returns_none():
    q = CommandQueue()
    assert q.pop(timeout=None) is None


# ── Priority ordering ─────────────────────────────────────────────────────────

def test_critical_before_normal():
    q = CommandQueue()
    q.push(make_cmd(CommandPriority.NORMAL,   "NORMAL"))
    q.push(make_cmd(CommandPriority.CRITICAL, "CRITICAL"))
    q.push(make_cmd(CommandPriority.LOW,      "LOW"))
    q.push(make_cmd(CommandPriority.HIGH,     "HIGH"))

    order = [q.pop(timeout=None).command_type for _ in range(4)]
    assert order == ["CRITICAL", "HIGH", "NORMAL", "LOW"]


def test_two_criticals_both_delivered():
    q = CommandQueue()
    q.push(make_cmd(CommandPriority.CRITICAL, "E_STOP_1"))
    q.push(make_cmd(CommandPriority.CRITICAL, "E_STOP_2"))
    types = {q.pop(timeout=None).command_type, q.pop(timeout=None).command_type}
    assert "E_STOP_1" in types and "E_STOP_2" in types


# ── Queue full ────────────────────────────────────────────────────────────────

def test_push_fails_when_full():
    q = CommandQueue(maxsize=2)
    assert q.push(make_cmd(CommandPriority.NORMAL))
    assert q.push(make_cmd(CommandPriority.NORMAL))
    assert not q.push(make_cmd(CommandPriority.NORMAL))  # should fail, queue full


# ── Clear ─────────────────────────────────────────────────────────────────────

def test_clear_empties_queue():
    q = CommandQueue()
    for _ in range(5):
        q.push(make_cmd(CommandPriority.NORMAL))
    q.clear()
    assert q.pop(timeout=None) is None


# ── Timestamp auto-fill ───────────────────────────────────────────────────────

def test_timestamp_auto_filled():
    cmd = HardwareCommand(priority=CommandPriority.NORMAL, timestamp=0.0, command_type="X")
    assert cmd.timestamp > 0.0
