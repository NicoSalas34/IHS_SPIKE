"""
Pre-annotate hyperspectral images with YOLO-seg and export to X-AnyLabeling JSON.

Pipeline:
  .hdr/.hyspex -> SpectrumCamera (RGB normalized by spectralon)
              -> crop x [crop_x_left:crop_x_right]
              -> save as JPG (uint8)
              -> YOLO-seg inference
              -> mask -> polygon -> LabelMe-style JSON (compatible X-AnyLabeling)

Usage:
  python yolo_seg/preannotate.py --input <dir|hdr> --output <dir> \
      [--weights runs/segment/seg/grains/weights/best.pt] [--conf 0.25] [--device cuda]
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
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
                       shapes: list) -> dict:
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


def collect_jpg_files(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() in (".jpg", ".jpeg"):
        return [input_path]
    return sorted(input_path.glob("*.jpg")) + sorted(input_path.glob("*.jpeg"))


def is_loadable(hdr_file: Path) -> bool:
    hyspex = str(hdr_file).replace(".hdr", ".hyspex")
    return os.path.exists(hyspex) and os.stat(hyspex).st_size >= 3e8


def process_jpg(jpg_file: Path, output_dir: Path, model: YOLO,
                conf: float, imgsz: int, device: str,
                epsilon_ratio: float) -> bool:
    out_json = output_dir / f"{jpg_file.stem}.json"
    if out_json.exists():
        print(f"  - already done: {jpg_file.name}")
        return True

    img = cv2.imread(str(jpg_file))
    if img is None:
        print(f"  ! cannot read: {jpg_file.name}")
        return False
    h, w = img.shape[:2]

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb, conf=conf, imgsz=imgsz, device=device, verbose=False)
    r = results[0]

    shapes = []
    if r.masks is not None:
        for mask_tensor, box in zip(r.masks.data, r.boxes):
            label = model.names[int(box.cls.item())]
            mask = mask_tensor.cpu().numpy()
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            poly = mask_to_polygon((mask > 0.5).astype(np.uint8), epsilon_ratio=epsilon_ratio)
            if poly is None:
                continue
            shapes.append({
                "label": label,
                "points": poly,
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "polygon",
                "flags": {},
                "attributes": {},
            })

    js = build_labelme_json(jpg_file.name, h, w, shapes)
    with open(out_json, "w") as f:
        json.dump(js, f, indent=2)
    n_grain = sum(1 for s in shapes if s["label"] == "grain")
    n_broken = sum(1 for s in shapes if s["label"] == "broken")
    print(f"  > wrote {out_json.name} ({n_grain} grain, {n_broken} broken)")
    return True


def process_one(hdr_file: Path, output_dir: Path, model: YOLO,
                crop_x_left: int, crop_x_right: int,
                conf: float, imgsz: int, device: str,
                epsilon_ratio: float, spectrum=None) -> bool:
    if not is_loadable(hdr_file):
        hyspex = str(hdr_file).replace(".hdr", ".hyspex")
        if not os.path.exists(hyspex):
            print(f"  ! skip (no .hyspex): {hdr_file.name}")
        else:
            print(f"  ! skip (file too small): {hdr_file.name}")
        return False

    stem = hdr_file.stem
    out_jpg = output_dir / f"{stem}.jpg"
    out_json = output_dir / f"{stem}.json"
    if out_jpg.exists() and out_json.exists():
        print(f"  - already done: {stem}")
        return True

    if spectrum is None:
        spectrum = SpectrumCamera(str(hdr_file))
    rgb = spectrum.image_rgb[:, crop_x_left:crop_x_right]
    rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(out_jpg), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))

    h, w = rgb_u8.shape[:2]
    results = model.predict(rgb_u8, conf=conf, imgsz=imgsz, device=device, verbose=False)
    r = results[0]

    shapes = []
    if r.masks is not None:
        for mask_tensor, box in zip(r.masks.data, r.boxes):
            label = model.names[int(box.cls.item())]
            mask = mask_tensor.cpu().numpy()
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            poly = mask_to_polygon((mask > 0.5).astype(np.uint8), epsilon_ratio=epsilon_ratio)
            if poly is None:
                continue
            shapes.append({
                "label": label,
                "points": poly,
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "polygon",
                "flags": {},
                "attributes": {},
            })

    js = build_labelme_json(out_jpg.name, h, w, shapes)
    with open(out_json, "w") as f:
        json.dump(js, f, indent=2)
    n_grain = sum(1 for s in shapes if s["label"] == "grain")
    n_broken = sum(1 for s in shapes if s["label"] == "broken")
    print(f"  > wrote {out_jpg.name} + {out_json.name} "
          f"({n_grain} grain, {n_broken} broken)")
    return True


def main():
    cfg = load_default_config()
    seg_cfg = cfg.get("segment_kernels", {})

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help=".hdr file or directory containing .hdr files")
    ap.add_argument("--output", required=True, help="Output directory for JPG + JSON pairs")
    ap.add_argument("--weights", default="runs/segment/seg/grains/weights/best.pt",
                    help="Path to YOLO-seg best.pt weights")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=seg_cfg.get("device", "cuda"))
    ap.add_argument("--crop-x-left", type=int, default=seg_cfg.get("crop_x_left", 400))
    ap.add_argument("--crop-x-right", type=int, default=seg_cfg.get("crop_x_right", 2200))
    ap.add_argument("--epsilon", type=float, default=0.0005,
                    help="Polygon simplification ratio (higher = fewer points)")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO-seg weights from {args.weights}...")
    model = YOLO(args.weights)

    hdr_files = collect_hdr_files(input_path)
    jpg_files = collect_jpg_files(input_path) if input_path.is_dir() else []

    if not hdr_files and jpg_files:
        print(f"No .hdr found — running inference on {len(jpg_files)} JPG(s)\n")
        ok = 0
        for i, jpg in enumerate(jpg_files):
            print(f"[{i+1}/{len(jpg_files)}] {jpg.name}")
            try:
                if process_jpg(jpg, output_dir, model,
                               args.conf, args.imgsz, args.device, args.epsilon):
                    ok += 1
            except Exception as e:
                print(f"  !! error: {e}")
        print(f"\nDone: {ok}/{len(jpg_files)} processed -> {output_dir}")
        return

    files = hdr_files
    print(f"Found {len(files)} .hdr file(s)\n")

    executor = ThreadPoolExecutor(max_workers=1)
    pending_spectrum = None
    pending_file = None

    for f in files:
        if is_loadable(f):
            pending_spectrum = executor.submit(SpectrumCamera, str(f))
            pending_file = f
            break

    ok = 0
    for i, hdr in enumerate(files):
        print(f"[{i+1}/{len(files)}] {hdr.name}")
        try:
            spectrum = None
            if pending_file == hdr and pending_spectrum is not None:
                spectrum = pending_spectrum.result()

            pending_spectrum = None
            pending_file = None
            for next_file in files[i + 1:]:
                if is_loadable(next_file):
                    pending_spectrum = executor.submit(SpectrumCamera, str(next_file))
                    pending_file = next_file
                    break

            if process_one(hdr, output_dir, model,
                           args.crop_x_left, args.crop_x_right,
                           args.conf, args.imgsz, args.device,
                           args.epsilon, spectrum=spectrum):
                ok += 1
        except Exception as e:
            print(f"  !! error: {e}")

    executor.shutdown(wait=False)
    print(f"\nDone: {ok}/{len(files)} processed -> {output_dir}")


if __name__ == "__main__":
    main()
