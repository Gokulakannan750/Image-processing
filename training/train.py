"""
training/train.py
=================
Trains a custom YOLOv8n model on your farm obstacle dataset.

Prerequisites:
    1. Images collected  →  tools/collect_training_data.py
    2. Images labeled    →  Roboflow (exported as YOLOv8 format)
    3. Dataset placed at →  training/dataset/
                               images/train/   *.jpg
                               images/val/     *.jpg
                               labels/train/   *.txt
                               labels/val/     *.txt

Usage:
    python training/train.py
    python training/train.py --epochs 100 --batch 8 --device cuda
"""
import argparse
import os
import sys
from pathlib import Path


def validate_dataset(dataset_path: str) -> bool:
    required = [
        "images/train", "images/val",
        "labels/train", "labels/val",
    ]
    missing = [r for r in required if not Path(dataset_path, r).exists()]
    if missing:
        print("\nERROR: Missing dataset folders:")
        for m in missing:
            print(f"  {dataset_path}/{m}")
        print("\nRun the full workflow first:")
        print("  1. python tools/collect_training_data.py")
        print("  2. Label images in Roboflow → export as YOLOv8")
        print("  3. Place exported dataset in training/dataset/")
        return False
    train_imgs = list(Path(dataset_path, "images/train").glob("*.jpg")) + \
                 list(Path(dataset_path, "images/train").glob("*.png"))
    if len(train_imgs) < 10:
        print(f"\nWARNING: Only {len(train_imgs)} training images found.")
        print("Aim for at least 50 images per class for reliable detection.")
    else:
        print(f"  Training images : {len(train_imgs)}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Train custom farm obstacle YOLOv8 model")
    parser.add_argument("--epochs",  type=int,   default=50,                       help="Training epochs (default 50)")
    parser.add_argument("--batch",   type=int,   default=16,                       help="Batch size (default 16; reduce to 8 if OOM)")
    parser.add_argument("--device",  type=str,   default="cpu",                    help="'cpu', 'cuda', or '0' for first GPU")
    parser.add_argument("--model",   type=str,   default="yolov8n.pt",             help="Base model to fine-tune")
    parser.add_argument("--data",    type=str,   default="training/farm_obstacles.yaml", help="Dataset YAML config")
    parser.add_argument("--name",    type=str,   default="farm_obstacles",         help="Run name (saved in runs/detect/)")
    parser.add_argument("--imgsz",   type=int,   default=640,                      help="Input image size")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    dataset_path = str(Path(args.data).parent / "dataset")
    print("\n=== Farm Obstacle Training ===")
    print(f"  Base model : {args.model}")
    print(f"  Dataset    : {args.data}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch}")
    print(f"  Device     : {args.device}")
    print(f"  Image size : {args.imgsz}")

    if not validate_dataset(dataset_path):
        sys.exit(1)

    print("\nStarting training...\n")
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        imgsz=args.imgsz,
        name=args.name,
        patience=15,           # early stopping
        augment=True,          # random flips, crops, colour jitter
        degrees=10.0,          # rotation augmentation for tilted camera
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        project="runs/detect",
        exist_ok=True,
    )

    best_weights = Path("runs/detect") / args.name / "weights/best.pt"
    print(f"\n=== Training complete ===")
    print(f"Best weights saved to: {best_weights}")
    print(f"\nTo use your custom model, update config/default.yaml:")
    print(f"  detectors:")
    print(f"    yolo:")
    print(f"      model: \"{best_weights}\"")
    print(f"      classes:  # leave empty to use all classes in the model")


if __name__ == "__main__":
    main()
