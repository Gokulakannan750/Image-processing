"""
tools/find_camera.py
====================
Scans all camera indices and backends to find working cameras.
Run this first to diagnose camera issues.
"""
import cv2

BACKENDS = [
    (cv2.CAP_DSHOW,  "DirectShow (DSHOW)"),
    (cv2.CAP_MSMF,   "Media Foundation (MSMF)"),
    (cv2.CAP_ANY,    "Auto"),
]

print("\n=== Camera Scanner ===\n")
found = []

for index in range(5):
    for backend, name in BACKENDS:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"  FOUND — index={index}  backend={name}  resolution={w}x{h}")
                found.append((index, backend, name))
            cap.release()

if not found:
    print("  No cameras found.\n")
    print("  Possible causes:")
    print("  1. Camera app is open and blocking access — close it")
    print("  2. Windows camera privacy blocked — Settings > Privacy > Camera > ON")
    print("  3. Driver not installed — check Device Manager")
else:
    print(f"\n  Best choice for config/default.yaml:")
    index, backend, name = found[0]
    print(f"    camera:")
    print(f"      index: {index}")
    if backend == cv2.CAP_DSHOW:
        print(f"      backend: DSHOW   # add this to camera_stream.py if needed")
    print(f"\n  Total cameras found: {len(found)}")
