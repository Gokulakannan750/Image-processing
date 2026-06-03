"""
dashboard/server.py
===================
Flask web dashboard for the robotics vision system.

Exposes:
  GET /           — browser dashboard (HTML)
  GET /video      — live MJPEG stream of the annotated camera feed
  GET /status     — JSON snapshot of system state (polled by the UI)
  GET /logs       — last N log lines as JSON

Runs in a daemon thread; does not block the main vision loop.
"""

import threading
import time
from collections import deque
from typing import Generator, List, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

from utils.logger import get_logger

log = get_logger(__name__)

app = Flask(__name__)

# ── Shared state (written by main loop, read by Flask) ─────────────────────

_HISTORY_LEN = 120  # retain 120 data points (~60 s at 0.5 s poll rate)


class DashboardState:
    def __init__(self) -> None:
        self._lock = threading.Lock()

        self._frame: Optional[np.ndarray] = None
        self.vehicle_state: str = "IDLE"
        self.fps: float = 0.0
        self.latency_ms: float = 0.0
        self.steering: float = 0.0
        self.obstacles: list = []
        self.has_critical_obstacle: bool = False
        self.has_target: bool = False
        self.target_distance_m: Optional[float] = None
        self.uptime_s: float = 0.0
        self._start_time: float = time.time()
        self._log_buffer: deque = deque(maxlen=200)

        # Ring buffers for historical trend data
        self._fps_history: deque = deque(maxlen=_HISTORY_LEN)
        self._latency_history: deque = deque(maxlen=_HISTORY_LEN)
        self._steering_history: deque = deque(maxlen=_HISTORY_LEN)

        # Prometheus counters (monotonically increasing)
        self.estop_count: int = 0
        self.frames_processed: int = 0
        self.robot_id: str = "robot-0"
        self.yolo_faulted: bool = False

    def update(
        self,
        frame: Optional[np.ndarray],
        vehicle_state: str,
        fps: float,
        latency_ms: float,
        steering: float,
        obstacles: list,
        has_target: bool,
        target_distance_m: Optional[float],
    ) -> None:
        with self._lock:
            self._frame = frame.copy() if frame is not None else None
            self.vehicle_state = vehicle_state
            self.fps = fps
            self.latency_ms = latency_ms
            self.steering = steering
            self.obstacles = obstacles
            self.has_critical_obstacle = any(o.is_critical for o in obstacles)
            self.has_target = has_target
            self.target_distance_m = target_distance_m
            self.uptime_s = time.time() - self._start_time
            self._fps_history.append(round(fps, 1))
            self._latency_history.append(round(latency_ms, 1))
            self._steering_history.append(round(steering, 2))
            self.frames_processed += 1
            if vehicle_state == "STOPPED" and any(o.is_critical for o in obstacles):
                self.estop_count += 1

    def push_log(self, message: str) -> None:
        with self._lock:
            self._log_buffer.append({"t": time.strftime("%H:%M:%S"), "msg": message})

    def get_status(self) -> dict:
        with self._lock:
            return {
                "robot_id": self.robot_id,
                "state": self.vehicle_state,
                "fps": round(self.fps, 1),
                "latency_ms": round(self.latency_ms, 1),
                "steering": round(self.steering, 2),
                "has_target": self.has_target,
                "target_distance_m": (
                    round(self.target_distance_m, 2)
                    if self.target_distance_m is not None
                    else None
                ),
                "obstacle_count": len(self.obstacles),
                "has_critical_obstacle": self.has_critical_obstacle,
                "obstacles": [
                    {
                        "label": o.label,
                        "confidence": round(o.confidence * 100),
                        "critical": o.is_critical,
                    }
                    for o in self.obstacles
                ],
                "uptime_s": round(self.uptime_s),
                "yolo_faulted": self.yolo_faulted,
            }

    def get_prometheus_metrics(self) -> str:
        """Return a Prometheus text-format metrics payload."""
        with self._lock:
            state_val = {
                "IDLE": 0,
                "DRIVING": 1,
                "TURNING": 2,
                "RECOVERING": 3,
                "STOPPED": 4,
            }.get(self.vehicle_state, -1)
            lines = [
                "# HELP agribot_fps Current camera processing FPS",
                "# TYPE agribot_fps gauge",
                f"agribot_fps {self.fps:.2f}",
                "",
                "# HELP agribot_latency_ms Frame processing latency in milliseconds",
                "# TYPE agribot_latency_ms gauge",
                f"agribot_latency_ms {self.latency_ms:.2f}",
                "",
                "# HELP agribot_steering_correction Current steering correction value",
                "# TYPE agribot_steering_correction gauge",
                f"agribot_steering_correction {self.steering:.4f}",
                "",
                "# HELP agribot_vehicle_state Encoded vehicle state (0=IDLE 1=DRIVING 2=TURNING 3=RECOVERING 4=STOPPED)",
                "# TYPE agribot_vehicle_state gauge",
                f"agribot_vehicle_state {state_val}",
                "",
                "# HELP agribot_obstacle_count Number of currently detected obstacles",
                "# TYPE agribot_obstacle_count gauge",
                f"agribot_obstacle_count {len(self.obstacles)}",
                "",
                "# HELP agribot_critical_obstacle Whether a critical obstacle is blocking the path (0/1)",
                "# TYPE agribot_critical_obstacle gauge",
                f"agribot_critical_obstacle {int(self.has_critical_obstacle)}",
                "",
                "# HELP agribot_has_target Whether a navigation target is currently detected (0/1)",
                "# TYPE agribot_has_target gauge",
                f"agribot_has_target {int(self.has_target)}",
                "",
                "# HELP agribot_uptime_seconds Seconds since the system started",
                "# TYPE agribot_uptime_seconds counter",
                f"agribot_uptime_seconds {self.uptime_s:.1f}",
                "",
                "# HELP agribot_estop_total Total number of emergency stop events triggered",
                "# TYPE agribot_estop_total counter",
                f"agribot_estop_total {self.estop_count}",
                "",
                "# HELP agribot_frames_total Total number of frames processed",
                "# TYPE agribot_frames_total counter",
                f"agribot_frames_total {self.frames_processed}",
                "",
                "# HELP agribot_yolo_faulted Whether YOLO obstacle detection is disabled due to errors (0/1)",
                "# TYPE agribot_yolo_faulted gauge",
                f"agribot_yolo_faulted {int(self.yolo_faulted)}",
                "",
            ]
        return "\n".join(lines)

    def get_history(self) -> dict:
        with self._lock:
            return {
                "fps": list(self._fps_history),
                "latency_ms": list(self._latency_history),
                "steering": list(self._steering_history),
            }

    def get_logs(self) -> List[dict]:
        with self._lock:
            return list(self._log_buffer)

    def get_frame_jpeg(self) -> Optional[bytes]:
        with self._lock:
            if self._frame is None:
                return None
            ok, buf = cv2.imencode(".jpg", self._frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return buf.tobytes() if ok else None


# Module-level singleton — imported by main.py
dashboard_state = DashboardState()


# ── Flask routes ───────────────────────────────────────────────────────────


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/video")
def video() -> Response:
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status() -> Response:
    return jsonify(dashboard_state.get_status())


@app.route("/logs")
def logs() -> Response:
    return jsonify(dashboard_state.get_logs())


@app.route("/metrics/history")
def metrics_history() -> Response:
    return jsonify(dashboard_state.get_history())


@app.route("/metrics")
def prometheus_metrics() -> Response:
    """Prometheus-compatible scrape endpoint."""
    return Response(
        dashboard_state.get_prometheus_metrics(),
        mimetype="text/plain; version=0.0.4; charset=utf-8",
    )


def _mjpeg_generator() -> Generator[bytes, None, None]:
    while True:
        jpeg = dashboard_state.get_frame_jpeg()
        if jpeg:
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(0.033)  # ~30 fps cap on the stream


# ── Server lifecycle ───────────────────────────────────────────────────────


def start(host: str = "0.0.0.0", port: int = 5000) -> None:
    """Start the Flask server in a background daemon thread."""

    def _run() -> None:
        import logging as _logging

        _logging.getLogger("werkzeug").setLevel(
            _logging.ERROR
        )  # suppress Flask access logs
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)

    t = threading.Thread(target=_run, name="dashboard-server", daemon=True)
    t.start()
    log.info("Web dashboard started -> http://localhost:%d", port)
