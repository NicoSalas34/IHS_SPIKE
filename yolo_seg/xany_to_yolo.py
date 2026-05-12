"""
Convert X-AnyLabeling (LabelMe-style) JSON annotations to a YOLO-seg dataset.

Input  : directory with pairs <stem>.jpg + <stem>.json (polygons labeled
         'intact' / 'broken')
Output : YOLO dataset layout
         <output>/
           data.yaml
           images/{train,val}/*.jpg
           labels/{train,val}/*.txt

Each label line: class_id x1 y1 x2 y2 ... xn yn   (normalized 0..1)

Usage:
  python yolo_seg/xany_to_yolo.py --input <annotated_dir> --output <dataset_dir> \
      [--classes intact broken] [--val-ratio 0.2] [--seed 42]
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import yaml


def parse_json(json_path: Path, class_to_id: dict, skip_labels: set) -> tuple[list[str], int, int, str]:
    with open(json_path) as f:
        data = json.load(f)
    h, w = data["imageHeight"], data["imageWidth"]
    img_name = data.get("imagePath") or f"{json_path.stem}.jpg"
    lines = []
    skipped = 0
    for shp in data.get("shapes", []):
        if shp.get("shape_type") != "polygon":
            continue
        label = shp.get("label", "")
        if label in skip_labels:
            continue
        if label not in class_to_id:
            skipped += 1
            continue
        pts = shp.get("points", [])
        if len(pts) < 3:
            continue
        normalized = []
        for x, y in pts:
            nx = max(0.0, min(1.0, x / w))
            ny = max(0.0, min(1.0, y / h))
            normalized.extend([f"{nx:.6f}", f"{ny:.6f}"])
        lines.append(f"{class_to_id[label]} " + " ".join(normalized))
    return lines, skipped, h, w, img_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Directory with annotated JPG + JSON pairs")
    ap.add_argument("--output", required=True, help="Output dataset directory")
    ap.add_argument("--classes", nargs="+", default=["grain", "broken"],
                   help="Class names in order (class index 0, 1, ...)")
    ap.add_argument("--skip-labels", nargs="*", default=[],
                   help="Labels to silently ignore (e.g. unrelabeled pre-annotations)")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    class_to_id = {name: i for i, name in enumerate(args.classes)}
    skip_labels = set(args.skip_labels)

    json_files = sorted(input_dir.glob("*.json"))
    pairs = []
    for jp in json_files:
        img = input_dir / f"{jp.stem}.jpg"
        if not img.exists():
            print(f"! no image for {jp.name}")
            continue
        pairs.append((img, jp))
    if not pairs:
        raise SystemExit(f"No JPG+JSON pairs found in {input_dir}")

    random.seed(args.seed)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"Total: {len(pairs)} | train: {len(train_pairs)} | val: {len(val_pairs)}")

    for split, items in (("train", train_pairs), ("val", val_pairs)):
        img_out = output_dir / "images" / split
        lbl_out = output_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_path, json_path in items:
            lines, skipped, _, _, _ = parse_json(json_path, class_to_id, skip_labels)
            if skipped:
                print(f"  ~ {json_path.name}: {skipped} polygon(s) with unknown label skipped")
            shutil.copy2(img_path, img_out / img_path.name)
            (lbl_out / f"{img_path.stem}.txt").write_text("\n".join(lines))

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(args.classes),
        "names": args.classes,
    }
    with open(output_dir / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    print(f"\nDataset ready: {output_dir}")
    print(f"data.yaml -> {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
