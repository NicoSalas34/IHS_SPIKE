# YOLO-seg pipeline — grain detection (intact / broken)

Replaces SAM for grain segmentation, and adds classification of each grain as
`intact` or `broken` in a single model pass.

## Layout

- `preannotate.py` — runs SAM on `.hyspex` files and exports JPG + LabelMe-style
  JSON ready for X-AnyLabeling. Every polygon is pre-labeled `grain`.
- `xany_to_yolo.py` — converts the corrected/labeled JSONs to a YOLO-seg dataset
  (with train/val split + `data.yaml`).
- `train.py` — trains a YOLO-seg model on the dataset.
- `predict.py` — runs a trained model on new `.hyspex` files.

## Workflow

### 1. Install ultralytics

```bash
source ihs_spike_env/bin/activate
pip install ultralytics
```

### 2. Pre-annotate with SAM

Pick a representative subset of acquisitions (~30–50 images is a good start) and run:

```bash
python yolo_seg/preannotate.py \
    --input "/media/salas/Crucial X9/.../Temp" \
    --output yolo_seg/to_annotate
```

This writes `<stem>.jpg` + `<stem>.json` pairs into `yolo_seg/to_annotate/`.
Every polygon has the placeholder label `grain`.

### 3. Annotate in X-AnyLabeling

Open `yolo_seg/to_annotate/` in X-AnyLabeling. For each image:

1. Fix incorrect or missing polygons (delete bad SAM masks, draw missing grains).
2. **Change every label from `grain` to either `intact` or `broken`.**
   Any polygon left with label `grain` is silently ignored by the next step.
3. Save (Ctrl+S). The JSON is updated in place.

Tips:
- Use the `R` shortcut in X-AnyLabeling to switch between labels quickly.
- Define the two classes (`intact`, `broken`) in the label panel before
  starting so the dropdown is populated.

### 4. Build the YOLO dataset

```bash
python yolo_seg/xany_to_yolo.py \
    --input yolo_seg/to_annotate \
    --output yolo_seg/datasets/grains \
    --classes intact broken \
    --val-ratio 0.2
```

This produces:

```
yolo_seg/datasets/grains/
    data.yaml
    images/{train,val}/*.jpg
    labels/{train,val}/*.txt
```

### 5. Train

```bash
python yolo_seg/train.py \
    --data yolo_seg/datasets/grains/data.yaml \
    --model yolo26s-seg.pt \
    --epochs 100 \
    --imgsz 1280 \
    --batch 8
```

If `yolo26s-seg.pt` is not yet available on the Ultralytics hub when you run
this, fall back to the current latest seg `s` model
(`--model yolo11s-seg.pt`).

Outputs go to `runs/seg/grains/`. Best weights: `runs/seg/grains/weights/best.pt`.

### 6. Predict

```bash
python yolo_seg/predict.py \
    --weights runs/seg/grains/weights/best.pt \
    --input "/media/salas/Crucial X9/.../Temp" \
    --output yolo_seg/predictions
```

Writes for each acquisition:
- `<stem>_pred.jpg` — overlay with masks, classes, confidences.
- `<stem>_pred.csv` — one row per grain (class, confidence, bbox).

## Notes

- The RGB used everywhere (annotation + train + inference) is the same:
  spectralon-normalized bands 82/54/14, cropped on x by
  `crop_x_left`/`crop_x_right` from `config.yml`. Train/test consistency is
  what matters most.
- Pre-annotation reuses `config.yml` defaults (`area_min`, `area_max`, crop,
  SAM model). Override with CLI flags if needed.
- `xany_to_yolo.py` skips any polygon still labeled `grain` so you can
  annotate in several passes without breaking the dataset.
