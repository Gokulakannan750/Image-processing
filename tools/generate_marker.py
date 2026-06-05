"""
tools/generate_marker.py
========================
Generate ArUco markers (DICT_6X6_250) for the field — print-ready PNGs with a
white quiet-zone border and a caption underneath so you know which is which.

Single marker (display on screen):
    python tools/generate_marker.py --id 0

Save one marker to a PNG:
    python tools/generate_marker.py --id 0 --save

Save a batch (great for a whole field):
    python tools/generate_marker.py --range 1 8        # IDs 1..8
    python tools/generate_marker.py --ids 1,2,3,248,249

Save the two special end-of-field markers:
    python tools/generate_marker.py --ids 249,248      # LAST-ROW + STOP

Options:
    --out markers/        # output folder (created if needed)
    --size 800            # marker pixel size (bigger = sharper print)
    --no-caption          # omit the text caption under the marker

Default special IDs (match config/default.yaml):
    249 = LAST-ROW   (final U-turn, then watch for STOP)
    248 = STOP       (field complete — machine halts)
"""

import argparse
import os
import cv2
import cv2.aruco as aruco
import numpy as np

# Keep these in sync with config/default.yaml -> row_tracker.*
LAST_ROW_ID = 249
STOP_ID = 248


def caption_for(marker_id: int) -> str:
    if marker_id == LAST_ROW_ID:
        return f"LAST-ROW  (ID {marker_id})"
    if marker_id == STOP_ID:
        return f"STOP  (ID {marker_id})"
    return f"ID {marker_id}"


def generate(marker_id: int, size_px: int = 600, caption: bool = True) -> np.ndarray:
    """Return a print-ready marker image (white border + optional caption)."""
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    marker = aruco.generateImageMarker(aruco_dict, marker_id, size_px)

    border = size_px // 8  # quiet zone — required for reliable detection
    cap_h = (size_px // 6) if caption else 0

    canvas = np.ones(
        (size_px + border * 2 + cap_h, size_px + border * 2), dtype=np.uint8
    ) * 255
    canvas[border : border + size_px, border : border + size_px] = marker

    img = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    if caption:
        text = caption_for(marker_id)
        scale = size_px / 600.0
        thick = max(1, int(2 * scale))
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, thick
        )
        tx = (canvas.shape[1] - tw) // 2
        ty = border + size_px + cap_h // 2 + th // 2
        cv2.putText(
            img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
            0.9 * scale, (0, 0, 0), thick, cv2.LINE_AA,
        )
    return img


def _parse_ids(args) -> list:
    if args.ids:
        return [int(x) for x in args.ids.split(",") if x.strip() != ""]
    if args.range:
        a, b = args.range
        return list(range(a, b + 1))
    return [args.id]


def main() -> None:
    p = argparse.ArgumentParser(description="ArUco marker generator (DICT_6X6_250)")
    p.add_argument("--id", type=int, default=0, help="Single marker ID (0-249)")
    p.add_argument("--ids", type=str, help="Comma list, e.g. 1,2,3,248,249")
    p.add_argument(
        "--range", type=int, nargs=2, metavar=("START", "END"),
        help="Inclusive ID range, e.g. --range 1 8",
    )
    p.add_argument("--size", type=int, default=800, help="Marker size in px")
    p.add_argument("--out", type=str, default=".", help="Output folder")
    p.add_argument("--save", action="store_true", help="Save single --id to PNG")
    p.add_argument("--no-caption", action="store_true", help="Omit text caption")
    args = p.parse_args()

    caption = not args.no_caption
    batch = bool(args.ids or args.range)
    ids = _parse_ids(args)

    # Validate range
    bad = [i for i in ids if i < 0 or i > 249]
    if bad:
        p.error(f"Marker IDs must be 0-249 for DICT_6X6_250. Invalid: {bad}")

    # ── Batch or explicit save → write PNG files ──────────────────────────
    if batch or args.save:
        os.makedirs(args.out, exist_ok=True)
        for mid in ids:
            img = generate(mid, args.size, caption)
            path = os.path.join(args.out, f"aruco_marker_id{mid}.png")
            cv2.imwrite(path, img)
            print(f"Saved {path}   ({caption_for(mid)})")
        print(f"\nDone — {len(ids)} marker(s) in '{os.path.abspath(args.out)}'.")
        print("Print each at >= 20 cm x 20 cm, laminate, and mount at the row ends.")
        return

    # ── Single marker → display on screen ─────────────────────────────────
    mid = args.id
    img = generate(mid, args.size, caption)
    win = "ArUco Marker — point your camera here (Q to close)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 800, 800)
    cv2.imshow(win, img)
    print(f"\nShowing ArUco {caption_for(mid)} (DICT_6X6_250)")
    print("Point your camera at this window, then run:  python main.py")
    print("To save instead, add --save.  Press Q to close.\n")
    while True:
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
