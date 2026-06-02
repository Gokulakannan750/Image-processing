"""
tests/test_dashboard_api.py
Tests for dashboard Flask API routes: /status, /logs, /metrics/history, /metrics.
Uses Flask test client — no real server needed.
"""

import json
import pytest
from dashboard.server import app, dashboard_state


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def reset_state():
    """Reset dashboard state before each test."""
    dashboard_state.vehicle_state = "IDLE"
    dashboard_state.fps = 15.0
    dashboard_state.latency_ms = 30.0
    dashboard_state.steering = 0.5
    dashboard_state.obstacles = []
    dashboard_state.has_critical_obstacle = False
    dashboard_state.has_target = False
    dashboard_state.target_distance_m = None
    dashboard_state.robot_id = "robot-test"
    dashboard_state.yolo_faulted = False
    dashboard_state._fps_history.clear()
    dashboard_state._latency_history.clear()
    dashboard_state._steering_history.clear()
    dashboard_state._log_buffer.clear()


# ── /status ───────────────────────────────────────────────────────────────────


def test_status_returns_200(client):
    r = client.get("/status")
    assert r.status_code == 200


def test_status_contains_expected_keys(client):
    r = client.get("/status")
    data = json.loads(r.data)
    for key in (
        "state",
        "fps",
        "latency_ms",
        "steering",
        "has_target",
        "obstacle_count",
        "has_critical_obstacle",
        "obstacles",
        "uptime_s",
        "robot_id",
        "yolo_faulted",
    ):
        assert key in data, f"Missing key: {key}"


def test_status_reflects_state(client):
    dashboard_state.vehicle_state = "DRIVING"
    dashboard_state.fps = 28.5
    r = client.get("/status")
    data = json.loads(r.data)
    assert data["state"] == "DRIVING"
    assert data["fps"] == 28.5


def test_status_robot_id(client):
    dashboard_state.robot_id = "robot-42"
    r = client.get("/status")
    data = json.loads(r.data)
    assert data["robot_id"] == "robot-42"


def test_status_yolo_faulted_flag(client):
    dashboard_state.yolo_faulted = True
    r = client.get("/status")
    data = json.loads(r.data)
    assert data["yolo_faulted"] is True


# ── /logs ─────────────────────────────────────────────────────────────────────


def test_logs_returns_list(client):
    r = client.get("/logs")
    data = json.loads(r.data)
    assert isinstance(data, list)


def test_logs_contain_pushed_entries(client):
    dashboard_state.push_log("Test message")
    r = client.get("/logs")
    data = json.loads(r.data)
    assert any("Test message" in entry["msg"] for entry in data)


# ── /metrics/history ──────────────────────────────────────────────────────────


def test_history_returns_three_series(client):
    r = client.get("/metrics/history")
    data = json.loads(r.data)
    assert "fps" in data and "latency_ms" in data and "steering" in data


def test_history_grows_after_update(client):
    dashboard_state.update(
        frame=None,
        vehicle_state="DRIVING",
        fps=25.0,
        latency_ms=20.0,
        steering=1.2,
        obstacles=[],
        has_target=True,
        target_distance_m=1.5,
    )
    r = client.get("/metrics/history")
    data = json.loads(r.data)
    assert len(data["fps"]) >= 1
    assert 25.0 in data["fps"]


# ── /metrics (Prometheus) ─────────────────────────────────────────────────────


def test_prometheus_endpoint_returns_200(client):
    r = client.get("/metrics")
    assert r.status_code == 200


def test_prometheus_content_type(client):
    r = client.get("/metrics")
    assert "text/plain" in r.content_type


def test_prometheus_contains_key_metrics(client):
    r = client.get("/metrics")
    body = r.data.decode()
    for metric in (
        "agribot_fps",
        "agribot_latency_ms",
        "agribot_vehicle_state",
        "agribot_estop_total",
        "agribot_yolo_faulted",
    ):
        assert metric in body, f"Missing metric: {metric}"
