"""
Pre-annotate hyperspectral images with SAM and export to X-AnyLabeling JSON.

Pipeline:
  .hdr/.hyspex -> SpectrumCamera (RGB normalized by spectralon)
              -> crop x [crop_x_left:crop_x_right]
              -> save as JPG (uint8)
              -> SAM AutomaticMaskGenerator
              -> filter by area
              -> contour -> polygon -> LabelMe-style JSON (compatible X-AnyLabeling)

Usage:
  python yolo_seg/preannotate.py --input <dir|hdr> --output <dir> \
      [--sam-model models/sam_b.pt] [--sam-type vit_b] [--device cuda]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spectrum import SpectrumCamera


def load_default_config():
    cfg_path = Path(__file__).resolve().parent.parent / "config.yml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return {}


def mask_to_polygon(mask: np.ndarray, epsilon_ratio: float = 0.002) -> list | None:
    """Convert a binary mask to a simplified polygon (list of [x, y] points)."""
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 3:
        return None
    epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    if len(approx) < 3:
        return None
    return [[float(p[0][0]), float(p[0][1])] for p in approx]


def build_labelme_json(image_name: str, height: int, width: int,
                      polygons: list, default_label: str) -> dict:
    shapes = [
        {
            "label": default_label,
            "points": poly,
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": "polygon",
            "flags": {},
            "attributes": {},
        }
        for poly in polygons
    ]
    return {
        "version": "2.4.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def collect_hdr_files(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix == ".hdr":
        return [input_path]
    return sorted(input_path.rglob("*.hdr"))


def process_one(hdr_file: Path, output_dir: Path, mask_gen,
               crop_x_left: int, crop_x_right: int,
               area_min: int, area_max: int,
               default_label: str, epsilon_ratio: float) -> bool:
    hyspex = str(hdr_file).replace(".hdr", ".hyspex")
    if not os.path.exists(hyspex):
        print(f"  ! skip (no .hyspex): {hdr_file.name}")
        return False
    if os.stat(hyspex).st_size < 3e8:
        print(f"  ! skip (file too small): {hdr_file.name}")
        return False

    stem = hdr_file.stem
    out_jpg = output_dir / f"{stem}.jpg"
    out_json = output_dir / f"{stem}.json"
    if out_jpg.exists() and out_json.exists():
        print(f"  - already done: {stem}")
        return True

    spectrum = SpectrumCamera(str(hdr_file))
    rgb = spectrum.image_rgb[:, crop_x_left:crop_x_right]
    rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(out_jpg), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))

    masks = mask_gen.generate(rgb)
    masks = [m for m in masks if area_min < m["area"] < area_max]
    print(f"  > {len(masks)} masks (after area filter)")

    polygons = []
    for m in masks:
        poly = mask_to_polygon(m["segmentation"], epsilon_ratio=epsilon_ratio)
        if poly is not None:
            polygons.append(poly)

    js = build_labelme_json(out_jpg.name, rgb_u8.shape[0], rgb_u8.shape[1],
                           polygons, default_label)
    with open(out_json, "w") as f:
        json.dump(js, f, indent=2)
    print(f"  > wrote {out_jpg.name} + {out_json.name} ({len(polygons)} polygons)")
    return True


def main():
    cfg = load_default_config()
    seg_cfg = cfg.get("segment_kernels", {})

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help=".hdr file or directory containing .hdr files")
    ap.add_argument("--output", required=True, help="Output directory for JPG + JSON pairs")
    ap.add_argument("--sam-model", default=seg_cfg.get("model_path", "models/sam_b.pt"))
    ap.add_argument("--sam-type", default=seg_cfg.get("model_type", "vit_b"))
    ap.add_argument("--device", default=seg_cfg.get("device", "cuda"))
    ap.add_argument("--crop-x-left", type=int, default=seg_cfg.get("crop_x_left", 400))
    ap.add_argument("--crop-x-right", type=int, default=seg_cfg.get("crop_x_right", 2200))
    ap.add_argument("--area-min", type=int, default=seg_cfg.get("area_min", 700))
    ap.add_argument("--area-max", type=int, default=seg_cfg.get("area_max", 5000))
    ap.add_argument("--default-label", default="grain",
                   help="Label assigned to every pre-annotation (you re-label in X-AnyLabeling)")
    ap.add_argument("--epsilon", type=float, default=0.002,
                   help="Polygon simplification ratio (higher = fewer points)")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SAM {args.sam_type} from {args.sam_model} on {args.device}...")
    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_model)
    sam.to(device=args.device)
    mask_gen = SamAutomaticMaskGenerator(model=sam, box_nms_thresh=0.25)

    files = collect_hdr_files(input_path)
    print(f"Found {len(files)} .hdr file(s)\n")

    ok = 0
    for i, hdr in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {hdr.name}")
        try:
            if process_one(hdr, output_dir, mask_gen,
                          args.crop_x_left, args.crop_x_right,
                          args.area_min, args.area_max,
                          args.default_label, args.epsilon):
                ok += 1
        except Exception as e:
            print(f"  !! error: {e}")

    print(f"\nDone: {ok}/{len(files)} processed -> {output_dir}")
    print("\nNext step: open the output folder in X-AnyLabeling, fix masks,")
    print("relabel each polygon as 'intact' or 'broken', then save.")


if __name__ == "__main__":
    main()
