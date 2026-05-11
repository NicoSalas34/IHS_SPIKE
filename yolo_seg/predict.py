"""
Run a trained YOLO-seg model on hyperspectral acquisitions.

For each .hdr file:
  - rebuild RGB via SpectrumCamera (spectralon-normalized bands 82/54/14)
  - crop x to grain area
  - run YOLO-seg
  - save annotated overlay JPG + per-grain CSV (class, confidence, bbox)

Usage:
  python yolo_seg/predict.py --weights runs/seg/grains/weights/best.pt \
      --input <dir|.hdr> --output <dir> [--conf 0.25]
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spectrum import SpectrumCamera


def load_default_config():
    cfg_path = Path(__file__).resolve().parent.parent / "config.yml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return {}


def collect_hdr_files(p: Path) -> list[Path]:
    if p.is_file() and p.suffix == ".hdr":
        return [p]
    return sorted(p.rglob("*.hdr"))


def main():
    cfg = load_default_config()
    seg_cfg = cfg.get("segment_kernels", {})

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    ap.add_argument("--crop-x-left", type=int, default=seg_cfg.get("crop_x_left", 400))
    ap.add_argument("--crop-x-right", type=int, default=seg_cfg.get("crop_x_right", 2200))
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    class_names = model.names

    files = collect_hdr_files(Path(args.input))
    print(f"Found {len(files)} .hdr file(s)\n")

    for i, hdr in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {hdr.name}")
        hyspex = str(hdr).replace(".hdr", ".hyspex")
        if not os.path.exists(hyspex) or os.stat(hyspex).st_size < 3e8:
            print("  ! skip")
            continue
        try:
            spectrum = SpectrumCamera(str(hdr))
        except Exception as e:
            print(f"  !! load error: {e}")
            continue
        rgb = spectrum.image_rgb[:, args.crop_x_left:args.crop_x_right]
        rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        results = model.predict(rgb_u8, conf=args.conf, imgsz=args.imgsz,
                              device=args.device, verbose=False)
        r = results[0]
        overlay = r.plot()
        cv2.imwrite(str(out_dir / f"{hdr.stem}_pred.jpg"), overlay)

        rows = []
        if r.boxes is not None and len(r.boxes) > 0:
            for j, box in enumerate(r.boxes):
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                rows.append({
                    "kernel": j + 1,
                    "class": class_names[cls],
                    "confidence": round(conf, 3),
                    "x1": round(x1, 1), "y1": round(y1, 1),
                    "x2": round(x2, 1), "y2": round(y2, 1),
                })
        csv_path = out_dir / f"{hdr.stem}_pred.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["kernel", "class", "confidence",
                                                   "x1", "y1", "x2", "y2"])
            writer.writeheader()
            writer.writerows(rows)
        n_intact = sum(1 for r_ in rows if r_["class"] == "intact")
        n_broken = sum(1 for r_ in rows if r_["class"] == "broken")
        print(f"  > {len(rows)} grains (intact={n_intact}, broken={n_broken})")


if __name__ == "__main__":
    main()
