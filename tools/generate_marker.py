"""
tools/generate_marker.py
========================
Generates and displays an ArUco marker that the camera can detect.

Usage:
    python tools/generate_marker.py              # display on screen
    python tools/generate_marker.py --save       # save as aruco_marker.png
    python tools/generate_marker.py --id 5       # use marker ID 5
"""
import argparse
import sys
import cv2
import cv2.aruco as aruco
import numpy as np


def generate(marker_id: int = 0, size_px: int = 600) -> np.ndarray:
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    marker = aruco.generateImageMarker(aruco_dict, marker_id, size_px)

    # Add a white border so the detector has contrast to lock onto
    border = size_px // 8
    canvas = np.ones((size_px + border * 2, size_px + border * 2), dtype=np.uint8) * 255
    canvas[border:border + size_px, border:border + size_px] = marker
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="ArUco marker generator")
    parser.add_argument("--id",   type=int, default=0,    help="Marker ID (0-249)")
    parser.add_argument("--size", type=int, default=600,  help="Marker size in pixels")
    parser.add_argument("--save", action="store_true",    help="Save to aruco_marker.png")
    args = parser.parse_args()

    canvas = generate(args.id, args.size)

    if args.save:
        path = f"aruco_marker_id{args.id}.png"
        cv2.imwrite(path, canvas)
        print(f"Saved: {path}")
        print("Print it out or open it on your phone and point the camera at it.")
        return

    # Display fullscreen so you can point a phone camera at the monitor
    bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    cv2.putText(bgr, f"ArUco ID: {args.id}  |  DICT_6X6_250  |  Press Q to quit",
                (10, bgr.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2)

    win = "ArUco Marker — point your camera here"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 800, 800)
    cv2.imshow(win, bgr)

    print(f"\nShowing ArUco marker ID {args.id} (DICT_6X6_250)")
    print("Point your camera at this window, then run:  python main.py")
    print("Press Q in this window to close.\n")

    while True:
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
