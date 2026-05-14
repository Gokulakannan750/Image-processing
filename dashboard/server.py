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
from typing import List, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

from utils.logger import get_logger

log = get_logger(__name__)

app = Flask(__name__)

# ── Shared state (written by main loop, read by Flask) ─────────────────────

class DashboardState:
    def __init__(self):
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
        self._log_buffer: deque = deque(maxlen=60)

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

    def push_log(self, message: str) -> None:
        with self._lock:
            self._log_buffer.append({"t": time.strftime("%H:%M:%S"), "msg": message})

    def get_status(self) -> dict:
        with self._lock:
            return {
                "state": self.vehicle_state,
                "fps": round(self.fps, 1),
                "latency_ms": round(self.latency_ms, 1),
                "steering": round(self.steering, 2),
                "has_target": self.has_target,
                "target_distance_m": round(self.target_distance_m, 2)
                    if self.target_distance_m is not None else None,
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
            }

    def get_logs(self) -> List[dict]:
        with self._lock:
            return list(self._log_buffer)

    def get_frame_jpeg(self) -> Optional[bytes]:
        with self._lock:
            if self._frame is None:
                return None
            ok, buf = cv2.imencode(
                ".jpg", self._frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            return buf.tobytes() if ok else None


# Module-level singleton — imported by main.py
dashboard_state = DashboardState()


# ── Flask routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    return jsonify(dashboard_state.get_status())


@app.route("/logs")
def logs():
    return jsonify(dashboard_state.get_logs())


def _mjpeg_generator():
    while True:
        jpeg = dashboard_state.get_frame_jpeg()
        if jpeg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
        time.sleep(0.033)  # ~30 fps cap on the stream


# ── Server lifecycle ───────────────────────────────────────────────────────

def start(host: str = "0.0.0.0", port: int = 5000) -> None:
    """Start the Flask server in a background daemon thread."""
    def _run():
        import logging as _logging
        _logging.getLogger("werkzeug").setLevel(_logging.ERROR)  # suppress Flask access logs
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)

    t = threading.Thread(target=_run, name="dashboard-server", daemon=True)
    t.start()
    log.info("Web dashboard started -> http://localhost:%d", port)
