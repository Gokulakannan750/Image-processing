"""
tools/calibrate_camera.py
=========================
Calibrate your camera so the app measures marker distances accurately.

You need a printed chessboard. The default is a 9x6 INNER-corner board
(i.e. a 10x7 grid of squares). Print it on A4, glue it to something flat
and rigid, and keep it flat.

LIVE mode (capture from the camera, the normal way):
    python tools/calibrate_camera.py
    - Hold the chessboard in view; green corners appear when it's found.
    - Press SPACE to capture a sample (take 15-20 from different angles
      and distances, tilting the board around).
    - Press C to calibrate, Q to quit.

OFFLINE mode (calibrate from a folder of photos you already took):
    python tools/calibrate_camera.py --images path/to/photos

Options:
    --cols 9 --rows 6      inner corners (width x height) of the board
    --square 0.025         square size in metres (does not affect the
                           camera_matrix; only used for the board pose)
    --camera 0             camera index or video/stream URL to use
    --min 12               minimum samples before you can calibrate
    --out camera_calibration.yaml   where to write the result

When done it prints a ready-to-paste YAML block for config/default.yaml
(detectors.aruco.camera_matrix and dist_coeffs) and the reprojection error
(lower is better; aim for < 1.0 pixel).
"""

import argparse
import glob
import os
import cv2
import numpy as np


def _object_points(cols, rows, square):
    """3D coordinates of the chessboard's inner corners (z=0 plane)."""
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return objp * square


def _format_matrix(mtx):
    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]
    return (
        f"[[{fx:.2f}, 0, {cx:.2f}], "
        f"[0, {fy:.2f}, {cy:.2f}], "
        f"[0, 0, 1]]"
    )


def _format_dist(dist):
    vals = ", ".join(f"{v:.6f}" for v in dist.flatten()[:5])
    return f"[[{vals}]]"


def calibrate(obj_points, img_points, image_size):
    """Run OpenCV calibration; return (matrix, dist, reproj_error)."""
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    # Mean reprojection error in pixels.
    total = 0.0
    for i in range(len(obj_points)):
        proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        total += cv2.norm(img_points[i], proj, cv2.NORM_L2) / len(proj)
    return mtx, dist, total / max(len(obj_points), 1)


def _report(mtx, dist, err, out_path):
    print("\n" + "=" * 60)
    print("  CALIBRATION COMPLETE")
    print("=" * 60)
    print(f"  Reprojection error: {err:.4f} px  "
          f"({'GOOD' if err < 1.0 else 'consider re-capturing'})")
    print("\n  Paste these into config/default.yaml under detectors.aruco:\n")
    print("    camera_matrix: " + _format_matrix(mtx))
    print("    dist_coeffs: " + _format_dist(dist))
    print()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Camera calibration result\n")
        f.write(f"# Reprojection error: {err:.4f} px\n")
        f.write("# Paste under detectors.aruco in config/default.yaml\n")
        f.write("camera_matrix: " + _format_matrix(mtx) + "\n")
        f.write("dist_coeffs: " + _format_dist(dist) + "\n")
    print(f"  Saved to: {os.path.abspath(out_path)}")
    print("=" * 60 + "\n")


def run_offline(args):
    objp = _object_points(args.cols, args.rows, args.square)
    obj_points, img_points = [], []
    image_size = None
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(args.images, p)))

    if not files:
        print(f"No images found in {args.images}")
        return

    print(f"Found {len(files)} images. Detecting chessboard corners...")
    found = 0
    for fp in sorted(files):
        img = cv2.imread(fp)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows))
        if ok:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            obj_points.append(objp)
            img_points.append(corners)
            found += 1
            print(f"  [OK]   {os.path.basename(fp)}")
        else:
            print(f"  [miss] {os.path.basename(fp)} (board not found)")

    if found < args.min:
        print(f"\nOnly {found} usable images (need >= {args.min}). "
              "Take more photos of the board from varied angles.")
        return

    print(f"\nCalibrating from {found} images...")
    mtx, dist, err = calibrate(obj_points, img_points, image_size)
    _report(mtx, dist, err, args.out)


def run_live(args):
    objp = _object_points(args.cols, args.rows, args.square)
    obj_points, img_points = [], []
    image_size = None

    source = args.camera
    if source is None:
        source = 0
    # Allow numeric camera indices passed as strings.
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open camera source: {source}")
        print("Tip: run  python tools/find_camera.py  to find the right index.")
        return

    win = "Camera Calibration - SPACE=capture  C=calibrate  Q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("\nHold the chessboard in view. SPACE to capture, C to calibrate, "
          "Q to quit.\n")

    while True:
        ok_read, frame = cap.read()
        if not ok_read:
            print("Camera read failed.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows))

        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, (args.cols, args.rows), corners, found)

        status = (
            f"Samples: {len(obj_points)}/{args.min}+   "
            f"Board: {'FOUND - press SPACE' if found else 'not found'}"
        )
        cv2.rectangle(view, (0, 0), (view.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(view, status, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if found else (0, 165, 255), 2)
        cv2.imshow(win, view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key == ord(" ") and found:
            refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            obj_points.append(objp)
            img_points.append(refined)
            print(f"Captured sample {len(obj_points)}.")
        elif key in (ord("c"), ord("C")):
            if len(obj_points) < args.min:
                print(f"Need at least {args.min} samples "
                      f"(have {len(obj_points)}).")
                continue
            print(f"Calibrating from {len(obj_points)} samples...")
            mtx, dist, err = calibrate(obj_points, img_points, image_size)
            _report(mtx, dist, err, args.out)
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser(description="Camera calibration (chessboard)")
    p.add_argument("--cols", type=int, default=9, help="Inner corners across")
    p.add_argument("--rows", type=int, default=6, help="Inner corners down")
    p.add_argument("--square", type=float, default=0.025, help="Square size (m)")
    p.add_argument("--camera", default=None, help="Camera index or video/URL")
    p.add_argument("--images", default=None, help="Folder of photos (offline)")
    p.add_argument("--min", type=int, default=12, help="Min usable samples")
    p.add_argument("--out", default="camera_calibration.yaml", help="Output file")
    args = p.parse_args()

    if args.images:
        run_offline(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
