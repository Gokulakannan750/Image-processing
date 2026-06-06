"""
tools/analyze_video.py
======================
Run the FULL pipeline (vision detectors + decision engine + state machine)
over a video file, headless, and:
  - write an annotated output video with the STATE banner burned in, and
  - print every state transition (DRIVING / TURNING / STOPPED) with timestamp.

This lets you watch the software's quality on a recorded clip and verify that
it changes state to STOPPED when an obstacle is in the path ahead.

Usage:
    python tools/analyze_video.py V3.mp4
    python tools/analyze_video.py V3.mp4 --config config/orchard.yaml
    python tools/analyze_video.py V3.mp4 --out out.mp4 --max 600 --every 2
"""

import argparse
import os
import sys
import time

import cv2

# Allow running as a script (python tools/analyze_video.py) by putting the
# project root on the path so the project packages import correctly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_manager import config_manager  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger("analyze_video")

STATE_COLOURS = {
    "DRIVING": (60, 200, 60),
    "TURNING": (40, 210, 210),
    "RECOVERING": (40, 210, 210),
    "STOPPED": (40, 40, 230),
    "IDLE": (160, 160, 160),
}


def _banner(frame, state: str, extra: str = "") -> None:
    h, w = frame.shape[:2]
    col = STATE_COLOURS.get(state, (255, 255, 255))
    cv2.rectangle(frame, (0, 0), (w, 46), (20, 20, 20), -1)
    cv2.putText(frame, f"STATE: {state}", (12, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
    if extra:
        cv2.putText(frame, extra, (w - 360, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze a video through the full pipeline")
    ap.add_argument("video", help="Path to the video file")
    ap.add_argument("--config", default="config/orchard.yaml", help="Config file")
    ap.add_argument("--out", default=None, help="Annotated output mp4 (default: <video>_analyzed.mp4)")
    ap.add_argument("--max", type=int, default=0, help="Max frames to process (0=all)")
    ap.add_argument("--every", type=int, default=1, help="Process 1 of every N frames")
    ap.add_argument("--pace", type=float, default=0.05,
                    help="Seconds per frame so the YOLO thread keeps up (0=fast)")
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        print(f"Video not found: {args.video}")
        return

    config_manager.config_path = args.config
    config_manager.load()

    # Build the real pipeline.
    from detectors.detector_registry import build_detectors_from_config
    from vision.frame_pipeline import FramePipeline
    from navigation.decision_engine import DecisionEngine
    from controllers.command_queue import CommandQueue

    registry = build_detectors_from_config()
    pipeline = FramePipeline(registry)
    engine = DecisionEngine(CommandQueue())
    time.sleep(1.5)  # let the YOLO background thread warm up

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_path = args.out or (os.path.splitext(args.video)[0] + "_analyzed.mp4")
    writer = None

    prev_state = None
    transitions = []
    stop_frames = 0
    idx = 0
    processed = 0
    print(f"Analyzing {args.video} (config={args.config}) ...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.every == 0:
            annotated, result = pipeline.process(frame)
            state, steering = engine.process_detection(result)
            state_name = state.name

            crit = [o.label for o in result.obstacles if o.is_critical]
            extra = f"obstacles: {len(result.obstacles)}  critical: {len(crit)}"
            _banner(annotated, state_name, extra)

            if state_name != prev_state:
                t = idx / fps
                transitions.append((t, prev_state, state_name, crit))
                print(f"  t={t:6.1f}s  {prev_state} -> {state_name}"
                      + (f"  (obstacle: {', '.join(crit)})" if crit else ""))
                prev_state = state_name
            if state_name == "STOPPED":
                stop_frames += 1

            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                    max(1.0, fps / args.every), (w, h),
                )
            writer.write(annotated)
            processed += 1
            if args.pace > 0:
                time.sleep(args.pace)
            if args.max and processed >= args.max:
                break
        idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    registry.shutdown()

    print("\n=== Summary ===")
    print(f"  frames processed : {processed}")
    print(f"  state transitions: {len(transitions)}")
    print(f"  frames STOPPED   : {stop_frames}")
    print(f"  annotated video  : {os.path.abspath(out_path)}")
    if not any(t[2] == "STOPPED" for t in transitions):
        print("  NOTE: never entered STOPPED — no obstacle was in the path ahead.")


if __name__ == "__main__":
    main()
