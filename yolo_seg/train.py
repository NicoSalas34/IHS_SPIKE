"""
Train a YOLO-seg model on the wheat-kernel dataset.

Usage:
  python yolo_seg/train.py --data datasets/grains/data.yaml \
      [--model yolo26s-seg.pt] [--epochs 100] [--imgsz 1280] [--batch 8]

Note: if 'yolo26s-seg.pt' is not yet published on the Ultralytics hub,
fall back to 'yolo11s-seg.pt' (or whatever is current) with --model.
"""

import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to data.yaml")
    ap.add_argument("--model", default="yolo26s-seg.pt",
                   help="Base weights (downloaded automatically by ultralytics)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280,
                   help="Training image size (grain images are wide; 1280 keeps detail)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0", help="GPU id, 'cpu', or comma list")
    ap.add_argument("--project", default="runs/seg")
    ap.add_argument("--name", default="grains")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
