"""
tools/collect_training_data.py
==============================
Captures frames from the live camera and saves them as training images.

Usage:
    python tools/collect_training_data.py
    python tools/collect_training_data.py --output training/dataset/images/raw

Controls (in the camera window):
    SPACE  — save current frame
    A      — auto-capture mode (saves every N seconds)
    Q      — quit

Aim to collect 50-100 images per obstacle class, from different:
    - distances (close / mid / far)
    - angles (straight, left, right)
    - lighting conditions (morning, midday, overcast)
"""
import argparse
import os
import time
import cv2
from config.config_manager import config_manager

def main():
    parser = argparse.ArgumentParser(description="Training data collector")
    parser.add_argument(
        "--output", default="training/dataset/images/raw",
        help="Folder to save captured frames"
    )
    parser.add_argument(
        "--interval", type=float, default=2.0,
        help="Seconds between auto-captures (default 2.0)"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    source = config_manager.get("camera.source", None)
    if source is None:
        source = config_manager.get("camera.index", 0)
    source = int(source) if str(source).isdigit() else source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera source: {source}")
        return

    saved = 0
    auto_mode = False
    last_auto = time.time()

    print(f"\nSaving images to: {os.path.abspath(args.output)}")
    print("Controls: SPACE=save  A=toggle auto-capture  Q=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        h, w = display.shape[:2]

        # Status overlay
        mode_text = f"AUTO ({args.interval}s)" if auto_mode else "MANUAL"
        cv2.putText(display, f"Mode: {mode_text}  |  Saved: {saved}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "SPACE=save  A=auto  Q=quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Auto-capture countdown
        if auto_mode:
            remaining = args.interval - (time.time() - last_auto)
            cv2.putText(display, f"Next: {remaining:.1f}s",
                        (w - 140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        cv2.imshow("Training Data Collector", display)
        key = cv2.waitKey(30) & 0xFF

        save_now = False

        if key == ord("q"):
            break
        elif key == ord(" "):
            save_now = True
        elif key == ord("a"):
            auto_mode = not auto_mode
            last_auto = time.time()
            print(f"Auto-capture {'ON' if auto_mode else 'OFF'}")

        if auto_mode and (time.time() - last_auto) >= args.interval:
            save_now = True
            last_auto = time.time()

        if save_now:
            fname = os.path.join(args.output, f"frame_{int(time.time()*1000)}.jpg")
            cv2.imwrite(fname, frame)
            saved += 1
            print(f"  Saved [{saved}] {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {saved} images saved to: {os.path.abspath(args.output)}")
    print("Next step: upload these images to Roboflow for labeling.")

if __name__ == "__main__":
    main()
