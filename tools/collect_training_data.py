"""
tools/collect_training_data.py
==============================
Captures frames from the live camera and saves them as training images,
organised into per-class sub-folders so they are ready for Roboflow upload.

Usage:
    # Basic — saves to training/dataset/images/raw/
    python tools/collect_training_data.py

    # Tag images with a class name (creates a sub-folder)
    python tools/collect_training_data.py --class rock
    python tools/collect_training_data.py --class tractor
    python tools/collect_training_data.py --class tree
    python tools/collect_training_data.py --class irrigation_pipe
    python tools/collect_training_data.py --class debris
    python tools/collect_training_data.py --class animal
    python tools/collect_training_data.py --class person

Controls (in the camera window):
    SPACE       — save current frame
    A           — toggle auto-capture mode (saves every N seconds)
    1-7         — switch active class on the fly (see class list on screen)
    Q           — quit

Aim for 50-100 images per class from different:
    - distances  (close / mid / far)
    - angles     (straight, left, right)
    - lighting   (morning sun, overcast, midday shadow)
"""
import argparse
import os
import time
import cv2
from config.config_manager import config_manager

# Default classes — must match training/farm_obstacles.yaml
DEFAULT_CLASSES = [
    "person",
    "rock",
    "tractor",
    "tree",
    "irrigation_pipe",
    "animal",
    "debris",
]


def main():
    parser = argparse.ArgumentParser(description="Training data collector")
    parser.add_argument(
        "--output", default="training/dataset/images/raw",
        help="Root folder to save captured frames (default: training/dataset/images/raw)"
    )
    parser.add_argument(
        "--class", dest="cls", default=None,
        help="Starting obstacle class (e.g. rock, tractor). Creates a sub-folder."
    )
    parser.add_argument(
        "--interval", type=float, default=2.0,
        help="Seconds between auto-captures (default 2.0)"
    )
    args = parser.parse_args()

    # Open camera
    source = config_manager.get("camera.source", None)
    if source is None:
        source = config_manager.get("camera.index", 0)
    source = int(source) if str(source).isdigit() else source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera source: {source}")
        print("Tip: Try running  python tools/find_camera.py  to find your camera index.")
        return

    # Active class state
    classes = DEFAULT_CLASSES
    active_idx = classes.index(args.cls) if args.cls and args.cls in classes else 0

    saved_counts = {c: 0 for c in classes}
    auto_mode = False
    last_auto = time.time()

    print("\n=== Training Data Collector ===")
    print(f"Root output : {os.path.abspath(args.output)}")
    print(f"Classes     : {', '.join(classes)}")
    print("Controls    : SPACE=save | A=auto | 1-7=switch class | Q=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera lost. Exiting.")
            break

        display = frame.copy()
        h, w = display.shape[:2]
        active_class = classes[active_idx]

        # ── HUD ────────────────────────────────────────────────────────────
        # Top bar background
        cv2.rectangle(display, (0, 0), (w, 50), (30, 30, 30), -1)

        mode_text = f"AUTO ({args.interval}s)" if auto_mode else "MANUAL"
        cv2.putText(display, f"Class: [{active_idx+1}] {active_class.upper()}",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 120), 2)
        cv2.putText(display, f"Mode: {mode_text}  |  Saved: {saved_counts[active_class]}",
                    (w // 2 - 20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

        # Class selector list on the right
        for i, cls in enumerate(classes):
            color = (0, 255, 120) if i == active_idx else (140, 140, 140)
            cv2.putText(display, f"[{i+1}] {cls}  ({saved_counts[cls]})",
                        (w - 220, 75 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        # Bottom hint
        cv2.putText(display, "SPACE=save  A=auto  1-7=class  Q=quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Auto-capture countdown
        if auto_mode:
            remaining = max(0.0, args.interval - (time.time() - last_auto))
            cv2.putText(display, f"Next: {remaining:.1f}s",
                        (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

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
            print(f"Auto-capture {'ON' if auto_mode else 'OFF'} for class: {active_class}")
        elif ord("1") <= key <= ord(str(min(len(classes), 9))):
            active_idx = key - ord("1")
            active_class = classes[active_idx]
            print(f"Switched to class: {active_class}")

        if auto_mode and (time.time() - last_auto) >= args.interval:
            save_now = True
            last_auto = time.time()

        if save_now:
            # Save into a per-class sub-folder
            class_dir = os.path.join(args.output, active_class)
            os.makedirs(class_dir, exist_ok=True)
            fname = os.path.join(class_dir, f"{active_class}_{int(time.time()*1000)}.jpg")
            cv2.imwrite(fname, frame)
            saved_counts[active_class] += 1
            total = sum(saved_counts.values())
            print(f"  [{active_class}] saved {saved_counts[active_class]}  "
                  f"(total: {total})  →  {fname}")

    cap.release()
    cv2.destroyAllWindows()

    print("\n=== Session Summary ===")
    total = 0
    for cls, count in saved_counts.items():
        if count > 0:
            status = "✓ good" if count >= 50 else f"⚠ only {count} — aim for 50+"
            print(f"  {cls:<18} {count:>4} images   {status}")
            total += count
    print(f"\n  Total saved: {total} images")
    print(f"  Location   : {os.path.abspath(args.output)}")
    print("\nNext step: upload these images to Roboflow for labeling.")
    print("  See TRAINING_GUIDE.md for the full workflow.")


if __name__ == "__main__":
    main()
