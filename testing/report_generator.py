"""
testing/report_generator.py
==========================
Generates automated test reports from a recording session.
"""
import os
import json
import numpy as np
from typing import Dict, Any, List

class ReportGenerator:
    @staticmethod
    def generate_from_session(session_dir: str) -> str:
        meta_dir = os.path.join(session_dir, "metadata")
        if not os.path.exists(meta_dir):
            return "No metadata found for report."

        frame_data = []
        for filename in sorted(os.listdir(meta_dir)):
            if filename.endswith(".json"):
                with open(os.path.join(meta_dir, filename), "r") as f:
                    frame_data.append(json.load(f))

        if not frame_data:
            return "Empty session."

        # Aggregate Stats
        latencies = [f["latency_ms"] for f in frame_data]
        stabilities = [f["stability"] for f in frame_data]
        detection_counts = [len(f["targets"]) for f in frame_data]
        
        # Calculate marker loss events (target count drops to 0)
        marker_losses = 0
        for i in range(1, len(detection_counts)):
            if detection_counts[i-1] > 0 and detection_counts[i] == 0:
                marker_losses += 1

        report = {
            "session": os.path.basename(session_dir),
            "total_frames": len(frame_data),
            "avg_latency_ms": float(np.mean(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "avg_stability": float(np.mean(stabilities)),
            "marker_loss_events": marker_losses,
            "avg_targets_per_frame": float(np.mean(detection_counts))
        }

        # Save report
        report_path = os.path.join(session_dir, "test_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
            
        # Create text summary
        summary_path = os.path.join(session_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("=== ROBOTICS VISION TEST REPORT ===\n")
            f.write(f"Session: {report['session']}\n")
            f.write(f"Total Frames: {report['total_frames']}\n")
            f.write(f"Avg Latency: {report['avg_latency_ms']:.2f} ms\n")
            f.write(f"Avg Stability: {report['avg_stability']:.1%}\n")
            f.write(f"Marker Losses: {report['marker_loss_events']}\n")
            f.write("===================================\n")

        return report_path
