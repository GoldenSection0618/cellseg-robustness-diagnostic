# Supervised Adaptation Protocol and Results

## Question and Scope

This protocol asks whether supervised YOLO-seg fine-tuning with DSB2018 instance
masks can improve on the classical baseline and close the gap to the zero-shot
Cellpose-SAM result. It is reported separately from the zero-shot comparison because
it uses target-domain labels.

The completed supervised method is single-class YOLO segmentation. Each DSB2018
instance mask is converted to one normalized polygon row with class id `0` (`cell`).
The repository evaluates predicted masks with the same object-level metrics used for
zero-shot methods rather than relying only on the trainer's native metrics.

This document covers the fixed-budget baseline and three completed diagnostic axes:
confidence threshold, label budget, and model capacity. It does not cover Cellpose
fine-tuning, post-processing experiments, or VLM mask output.

## Data Split and Label Conversion

All supervised runs draw from the 670 `stage1_train` images, which provide
per-instance PNG masks. A deterministic stable-order 80/20 split produces a 536-image
training pool and a shared 134-image held-out validation pool. The validation image
ids are unchanged across the fixed-budget, label-budget, and capacity comparisons.

The conversion pipeline traces each instance's exterior contour, simplifies it
conservatively, normalizes coordinates to the source image width and height, and uses
a tight rectangular polygon only for masks without a usable contour. It records the
source image id, split membership, instance totals, dropped instances, and fallback
usage. The task has one target class: all annotated DSB2018 instances are `cell`.

| Training set | Images | Train instances | Validation images | Validation instances | Dropped instances |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed budget | 100 | 5,167 | 134 | 5,599 | 0 |
| Nested budget | 250 | 11,533 | 134 | 5,599 | 0 |
| Full training pool | 536 | 23,862 | 134 | 5,599 | 0 |

The 250-image set contains the original 100 training images, and the 536-image set
contains the 250-image set. The conversion manifests and split record are available
in [the label-budget manifest](../results/supervised/yolo_label_budget_diagnostic_manifest.csv),
[split table](../results/supervised/yolo_label_budget_diagnostic_split.csv), and
[conversion summary](../results/supervised/yolo_label_budget_diagnostic_summary.csv).

## Training and Evaluation

The fixed-budget and label-budget runs fine-tune pretrained `yolo11n-seg.pt` for 50
epochs at `imgsz=512`, with `batch=8`, AMP disabled, and `patience=50`. The capacity
diagnostic uses the same full training pool and epoch/image-size settings with
`yolo11m-seg.pt`. Training metadata records the exact weights, settings, runtime,
hardware, and checkpoint path for every run.

Predicted polygons are rasterized back into labeled instance masks and compared with
the DSB2018 masks using object F1, precision, recall, matched IoU/Dice, absolute
count error, and latency. The standard reported operating point is `conf=0.25`;
the fixed-budget threshold diagnostic separately evaluates `0.05`, `0.10`, `0.25`,
`0.40`, and `0.60` without retraining.

The short label and training checks established that polygon conversion, training,
prediction export, and repository-metric evaluation were connected correctly. They
are retained as audit artifacts, not as estimates of supervised performance:
[label conversion summary](../results/supervised/yolo_label_smoke_summary.csv),
[tiny training summary](../results/supervised/yolo_tiny_train_smoke_summary.csv), and
[tiny evaluation summary](../results/supervised/yolo_tiny_train_smoke_eval_summary.csv).

## Held-out Comparison

All rows below use the same 134 held-out validation images. Values are mean metrics;
the supervised rows use the standard `conf=0.25` operating point.

| Method | Labels | F1 | Precision | Recall | Count error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cellpose-SAM | 0 | 0.9200 | 0.9456 | 0.9007 | 2.9328 |
| YOLO11m | 536 images | 0.8680 | 0.8525 | 0.8921 | 4.8582 |
| YOLO11n | 536 images | 0.8649 | 0.8440 | 0.8942 | 4.2090 |
| YOLO11n | 250 images | 0.8576 | 0.8400 | 0.8845 | 6.2090 |
| YOLO11n | 100 images | 0.8530 | 0.8419 | 0.8737 | 6.0896 |
| Otsu + watershed | 0 | 0.6442 | 0.6103 | 0.7219 | 19.8806 |

Cellpose-SAM remains ahead of all completed supervised runs on both F1 and count
error. YOLO11n improves as the label budget grows from 100 to 536 images, especially
in count error, but the change in F1 is modest. YOLO11m increases F1 by 0.0031 over
YOLO11n at the full budget while increasing count error, so the completed capacity
comparison does not show a material improvement from the larger model.

The machine-readable comparisons are the
[label-budget held-out summary](../results/supervised/yolo_label_budget_diagnostic_val_comparison_summary.csv)
and [capacity held-out summary](../results/supervised/yolo_capacity_diagnostic_val_comparison_summary.csv).
The visual summary is [the held-out Protocol A/B comparison](../figures/protocol_ab_heldout_val_comparison.png).

## Operating-point Diagnostic

The frozen 100-image YOLO11n checkpoint was evaluated at a predeclared confidence
grid. This measures the precision-recall trade-off of one model; it does not alter
the training comparison above.

| Confidence | F1 | Precision | Recall | Count error |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.7607 | 0.6855 | 0.8826 | 16.5224 |
| 0.10 | 0.8023 | 0.7509 | 0.8794 | 10.3134 |
| 0.25 | 0.8530 | 0.8419 | 0.8737 | 6.0896 |
| 0.40 | 0.8695 | 0.8911 | 0.8566 | 5.7090 |
| 0.60 | 0.8412 | 0.9376 | 0.7774 | 10.6418 |

The diagnostic has a clear interior optimum near `conf=0.40`: a low threshold
creates false positives, while a high threshold reduces recall. Even at 0.40, the
100-image model remains below the held-out Cellpose-SAM F1 of 0.9200. The complete
table and figures are the [threshold summary](../results/supervised/yolo_threshold_diagnostic_summary.csv),
[F1 curve](../figures/supervised_yolo_threshold_diagnostic_f1.png), and
[count-error curve](../figures/supervised_yolo_threshold_diagnostic_count_error.png).

## Training Record

| Run | Model | Train images | Time | Peak GPU memory |
| --- | --- | ---: | ---: | ---: |
| Fixed budget | YOLO11n | 100 | 351 s | 2,575 MB |
| Nested budget | YOLO11n | 250 | 571 s | 2,560 MB |
| Full training pool | YOLO11n | 536 | 975 s | 2,658 MB |
| Capacity diagnostic | YOLO11m | 536 | 2,827 s | 6,837 MB |

Detailed records are available for the [100-image run](../results/supervised/yolo_fixed_budget_train_metadata.csv),
[250-image run](../results/supervised/yolo_label_budget_diagnostic_budget_250_train_metadata.csv),
[536-image YOLO11n run](../results/supervised/yolo_label_budget_diagnostic_full_train_pool_train_metadata.csv),
and [536-image YOLO11m run](../results/supervised/yolo_capacity_diagnostic_yolo11m_train_metadata.csv).

## Interpretation Boundaries

- The held-out validation split is fixed across the reported comparisons, but the
  threshold diagnostic also evaluates that split. The threshold curve is therefore a
  diagnostic of the fixed-budget model, not an independently selected final operating
  point.
- The observed label-budget trend supports a benefit from more labels, but it does
  not isolate label quantity from all other possible training changes.
- The YOLO11m result tests one larger capacity point. It does not establish a general
  scaling law for YOLO segmentation models.
- These supervised results do not revise the zero-shot ranking. They answer a
  different question: how a standard YOLO-seg fine-tuning path performs when target
  annotations are available.
