"""
Train a YOLO model on grain datasets (single-stage or two-stage pipeline).

Single-stage seg (original):
  python yolo_seg/train.py --data datasets/grains/data.yaml --model yolo26s-seg.pt

Two-stage — stage 1 (detection, full images, single class):
  python yolo_seg/train.py --data datasets/grains_det/data.yaml \
      --model yolo26s.pt --task detect --imgsz 1280 --name grains_det

Two-stage — stage 2 (seg on grain crops, grain/broken):
  python yolo_seg/train.py --data datasets/grains_crops/data.yaml \
      --model yolo26s-seg.pt --task segment --imgsz 640 --name grains_crops
"""

import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to data.yaml")
    ap.add_argument("--model", default="yolo26s-seg.pt",
                    help="Base weights (downloaded automatically by ultralytics)")
    ap.add_argument("--task", default=None,
                    help="Override task: 'detect' or 'segment' "
                         "(inferred from model name if omitted)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0", help="GPU id, 'cpu', or comma list")
    ap.add_argument("--project", default="seg")
    ap.add_argument("--name", default="grains")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--mask-ratio", type=int, default=4,
                    help="Mask downsampling ratio — segment task only. "
                         "Lower = finer masks, more VRAM.")
    args = ap.parse_args()

    model = YOLO(args.model)

    train_kwargs = dict(
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

    # Segment-specific params
    task = args.task or model.task
    if task == "segment":
        train_kwargs["mask_ratio"] = args.mask_ratio
        train_kwargs["retina_masks"] = True

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
