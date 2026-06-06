"""
tools/extract_frames.py
========================
Pull still frames out of the farm videos so they can be:
  - inspected / tuned against, and
  - labelled to train a drivable-alley segmentation model.

Usage:
    python tools/extract_frames.py path/to/V1.mp4
    python tools/extract_frames.py path/to/V1.mp4 --every 30 --out dataset/raw
    python tools/extract_frames.py path/to/V1.mp4 --max 200 --resize 1280

Options:
    --every N     save 1 frame every N frames (default 30 ~= 1/sec at 30fps)
    --max N       stop after saving N frames (default: all)
    --out DIR     output folder (default: dataset/raw)
    --resize W    resize so the width is W px (keeps aspect; 0 = no resize)
    --prefix STR  filename prefix (default: video file stem)
"""

import argparse
import os
import cv2


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract frames from a video")
    ap.add_argument("video", help="Path to the video file (mp4/avi/mov)")
    ap.add_argument("--every", type=int, default=30, help="Save 1 of every N frames")
    ap.add_argument("--max", type=int, default=0, help="Max frames to save (0=all)")
    ap.add_argument("--out", default="dataset/raw", help="Output folder")
    ap.add_argument("--resize", type=int, default=0, help="Target width px (0=off)")
    ap.add_argument("--prefix", default=None, help="Filename prefix")
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        print(f"Video not found: {args.video}")
        return

    prefix = args.prefix or os.path.splitext(os.path.basename(args.video))[0]
    os.makedirs(args.out, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    print(f"{args.video}: {total} frames @ {fps:.1f} fps. "
          f"Saving 1 every {args.every} frame(s) to '{args.out}'.")

    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.every == 0:
            if args.resize and frame.shape[1] != args.resize:
                scale = args.resize / frame.shape[1]
                frame = cv2.resize(
                    frame, (args.resize, int(frame.shape[0] * scale))
                )
            path = os.path.join(args.out, f"{prefix}_{idx:06d}.jpg")
            cv2.imwrite(path, frame)
            saved += 1
            if saved % 25 == 0:
                print(f"  saved {saved} frames...")
            if args.max and saved >= args.max:
                break
        idx += 1

    cap.release()
    print(f"\nDone: {saved} frames in '{os.path.abspath(args.out)}'.")
    print("Next: label these (alley vs trees vs background) to train a model.")


if __name__ == "__main__":
    main()
