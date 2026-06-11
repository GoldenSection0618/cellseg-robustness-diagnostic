# Supervised Adaptation Protocol

This protocol is separate from the zero-shot PoW track. It asks how much
target-domain supervised adaptation can improve over the completed zero-shot
baselines when DSB2018 training masks are available.

## Scope

Primary supervised method:

- YOLO segmentation fine-tuning with DSB2018 instance masks converted to YOLO
  polygon labels.

Optional later method:

- Cellpose fine-tuning, if the YOLO path is already stable and a comparable training
  budget can be recorded.

Do not mix supervised results into the zero-shot ranking. Report them as Protocol B.

## Split Policy

Use `stage1_train` only, because it contains per-instance PNG masks. Keep the split
deterministic and record the image ids used by every run.

Initial smoke:

- deterministic 20-image subset using the same evenly spaced selection style as the
  PoW clean subset;
- 80/20 train/val assignment by stable order within that subset;
- no training.

Future training:

- fixed full `stage1_train` train/val split with a recorded seed or stable index
  rule;
- no leakage from validation labels into training configuration decisions beyond
  explicitly named smoke/debug runs.

## Label Conversion

For each DSB2018 instance mask:

1. load the per-instance binary PNG masks;
2. trace the exterior contour for each instance;
3. simplify the contour conservatively;
4. use a tight rectangular polygon fallback for tiny masks whose contour cannot form
   at least three points;
5. normalize polygon coordinates to image width and height;
6. write one YOLO segmentation row per instance:

```text
0 x1 y1 x2 y2 ... xn yn
```

The current task has one class: `cell`.

Conversion smoke outputs:

- `results/supervised/yolo_label_smoke/labels/*.txt`
- `results/supervised/yolo_label_smoke/images.txt`
- `results/supervised/yolo_label_smoke/train.txt`
- `results/supervised/yolo_label_smoke/val.txt`
- `results/supervised/yolo_label_smoke/data.yaml`
- `results/supervised/yolo_label_smoke_manifest.csv`
- `results/supervised/yolo_label_smoke_summary.csv`
- `figures/supervised_yolo_label_smoke_overlays.png`

The smoke is successful only if labels are non-empty for all selected training
images, every instance is converted, polygon rows have at least three points,
normalized coordinates stay within `[0, 1]`, and overlay examples visibly align with
cell boundaries. The manifest records how many instances used the tiny-mask
rectangle fallback.

## Training Metadata

Every supervised training run must record:

- number of train and validation images;
- annotation type and conversion script;
- model name and pretrained weights;
- image size;
- epoch count;
- batch size;
- optimizer and learning-rate settings when they differ from defaults;
- training wall time;
- GPU name and peak memory if available;
- inference latency;
- output checkpoint path.

Write this metadata to `results/supervised/`.

## Evaluation

YOLO's native training metrics are not enough for comparison with the zero-shot
baselines. After prediction, convert YOLO masks back to labeled instance masks and
reuse the repository's existing object-level metrics:

- object F1;
- precision and recall;
- matched IoU and Dice;
- absolute count error;
- latency.

The first training run should be a tiny smoke run. A full supervised baseline should
only run after label conversion, prediction export, and metric conversion are all
verified.

## Tiny Training Smoke

The tiny training smoke should use pretrained YOLO segmentation weights when network
access is available. Store downloaded YOLO weights under `model_assets/yolo/`, which
is not tracked by git.

Expected tracked outputs:

- `results/supervised/yolo_tiny_train_smoke_metadata.csv`
- `results/supervised/yolo_tiny_train_smoke_summary.csv`

Ultralytics trainer internals, including checkpoints, are reproducible smoke
artifacts and are ignored under `results/supervised/yolo_tiny_train_smoke_run/`.

The completed smoke used `model_assets/yolo/yolo11n-seg.pt`, 16 train images, 4 val
images, 1 epoch, `imgsz=256`, `batch=2`, CUDA, and AMP disabled. Validation saw 375
instances, confirming that the YOLO label path is wired correctly.
