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

Full-split training policy:

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

The first training run was a tiny smoke run. The fixed-budget supervised baseline
was run only after label conversion, prediction export, and metric conversion were
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

## Tiny Evaluation Smoke

The tiny evaluation smoke converts YOLO predictions back to labeled instance masks
and evaluates them with the same repository metrics used by the zero-shot protocols.

Expected tracked outputs:

- `results/supervised/yolo_tiny_train_smoke_metrics.csv`
- `results/supervised/yolo_tiny_train_smoke_eval_summary.csv`
- `figures/supervised_yolo_tiny_train_smoke_overlays.png`

The completed 1-epoch tiny model predicts masks, but they do not match cell
instances well: mean object F1 is 0.0 on the four-image validation smoke. This is a
pipeline check, not a supervised baseline performance claim.

## Fixed YOLO Baseline Decision

The first Protocol B baseline should be a fixed-budget small-label YOLO baseline,
not an all-label training-pool run. This matches the README question: how much
task-specific supervised adaptation can buy when only a small labeled split is
available.

Fixed dataset policy:

- source data: all 670 `stage1_train` images;
- deterministic stable-order 80/20 pool split;
- validation pool: held-out 20% of `stage1_train` images, expected 134 images;
- training budget: 100 labeled images sampled evenly from the 80% training pool;
- no validation images used for training or training-budget selection.

Fixed training budget:

- model: `model_assets/yolo/yolo11n-seg.pt`;
- task: segmentation;
- epochs: 50;
- image size: 512;
- batch size: 8;
- workers: 2;
- AMP: disabled, matching the smoke path and avoiding extra Ultralytics downloads;
- optimizer and augmentation: Ultralytics defaults unless explicitly recorded;
- early stopping: disabled for the first baseline by setting patience equal to the
  epoch budget, so the run uses the declared fixed budget.

Fixed evaluation:

- predict on the held-out validation pool;
- convert YOLO masks back to repository labeled instance masks;
- report object F1, precision, recall, matched IoU, Dice, count error, and latency;
- compare against zero-shot baselines only on the same held-out validation image ids.

This baseline is intentionally modest. It is large enough to test whether supervised
adaptation changes the conclusion, but small enough to stay aligned with the
diagnostic, small-scale scope of the repository.

## Fixed YOLO Label Conversion

The fixed-budget dataset is prepared by `scripts/prepare_yolo_fixed_budget.py`.

Tracked outputs:

- `results/supervised/yolo_fixed_budget/images/*.png`
- `results/supervised/yolo_fixed_budget/labels/*.txt`
- `results/supervised/yolo_fixed_budget/images.txt`
- `results/supervised/yolo_fixed_budget/train.txt`
- `results/supervised/yolo_fixed_budget/val.txt`
- `results/supervised/yolo_fixed_budget/data.yaml`
- `results/supervised/yolo_fixed_budget_manifest.csv`
- `results/supervised/yolo_fixed_budget_split.csv`
- `results/supervised/yolo_fixed_budget_summary.csv`
- `figures/supervised_yolo_fixed_budget_overlays.png`

The completed conversion uses 100 training images, 134 held-out validation images,
and leaves 436 training-pool images unused. It converts all 10,766 selected
instances to YOLO segmentation polygons, with 0 dropped instances and 7 tiny-mask
rectangle fallbacks. The overlay figure confirms that polygon labels align with
ground-truth cell boundaries.

## Fixed YOLO Baseline Result

The fixed-budget supervised baseline is trained by
`scripts/run_yolo_fixed_budget_train.py` and evaluated by
`scripts/evaluate_yolo_fixed_budget.py`.

Tracked outputs:

- `results/supervised/yolo_fixed_budget_train_metadata.csv`
- `results/supervised/yolo_fixed_budget_train_summary.csv`
- `results/supervised/yolo_fixed_budget_metrics.csv`
- `results/supervised/yolo_fixed_budget_eval_summary.csv`
- `results/supervised/yolo_fixed_budget_val_comparison_metrics.csv`
- `results/supervised/yolo_fixed_budget_val_comparison_summary.csv`
- `figures/supervised_yolo_fixed_budget_eval_overlays.png`

The completed run used `yolo11n-seg.pt`, 100 training images, 134 held-out
validation images, 50 epochs, `imgsz=512`, `batch=8`, `workers=2`, AMP disabled,
and `patience=50`. Training completed in 351.421 seconds on the local RTX 4060
Laptop GPU, with peak allocated CUDA memory of 2575.14 MB.

The primary repository-metric evaluation uses `conf=0.25`, the conventional
Ultralytics prediction operating point. On the held-out validation split, YOLO
fixed-budget supervised reaches mean object F1 0.8530, precision 0.8419, recall
0.8737, mean matched IoU 0.8182, mean matched Dice 0.8957, and mean absolute count
error 6.0896.

On the same 134 held-out validation image ids, the clean zero-shot comparison is:

- Cellpose-SAM: mean object F1 0.9200, mean absolute count error 2.9328;
- YOLO fixed-budget supervised: mean object F1 0.8530, mean absolute count error
  6.0896;
- Otsu + watershed: mean object F1 0.6442, mean absolute count error 19.8806.

This means the fixed-budget YOLO baseline is a strong supervised result relative to
the classical lower bound, but it does not replace Cellpose-SAM as the strongest
current baseline under the repository object-level metrics.

## YOLO Follow-up Diagnostic Plan

The fixed-budget YOLO result is a valid completed baseline. Follow-up runs are
reported as diagnostic extensions.

Follow-up diagnostics answer narrower questions:

1. Is the gap explainable by a poor confidence operating point, without retraining?
2. Is YOLO mainly limited by the 100-image label budget?
3. Is YOLO mainly limited by the nano model capacity?
4. Is direct YOLO-seg fine-tuning poorly matched to dense microscopy instance masks?

Guardrails:

- keep the 134-image held-out validation ids fixed for final comparison;
- make follow-up output names explicit;
- do not tune repeatedly on the held-out validation split and then report only the
  best run as if it were a new baseline;
- report all attempted follow-up runs, including no-improvement results;
- compare against Cellpose-SAM and Otsu only on the same image ids used by the
  follow-up evaluation;
- stop each diagnostic axis after the predeclared small set of runs.

Follow-up sequence:

1. Operating-point diagnostic on `conf=0.05`, `0.10`, `0.25`, `0.40`, and `0.60`.
2. Label-budget diagnostic. Keep model size and training recipe fixed, then train
   predeclared budgets: 100, 250, and full 536-image training pool.
3. Model-capacity diagnostic. Use the full train pool and compare the completed
   `yolo11n-seg` run against a larger upper-probe model, `yolo11m-seg`.
4. Optional post-processing diagnostic with a small predeclared mask filtering rule.

Suggested output names:

- `results/supervised/yolo_threshold_diagnostic_metrics.csv`
- `results/supervised/yolo_threshold_diagnostic_summary.csv`
- `figures/supervised_yolo_threshold_diagnostic_f1.png`
- `figures/supervised_yolo_threshold_diagnostic_count_error.png`
- `results/supervised/yolo_label_budget_diagnostic_summary.csv`
- `results/supervised/yolo_capacity_diagnostic_val_comparison_summary.csv`

If the optional post-processing diagnostic is run later, it should use a distinct
`results/supervised/yolo_postprocessing_diagnostic_*.csv` output family.

Interpretation rules:

- If threshold sensitivity closes most of the gap, report YOLO as operating-point
  sensitive rather than claiming the v1 baseline was invalid.
- If larger label budgets improve monotonically, report the v1 gap as label-budget
  limited.
- If the larger YOLO model clearly improves over YOLO11n on the full train pool,
  report a capacity limitation and consider filling in an intermediate `yolo11s-seg`
  point.
- If none of these axes closes the gap, report evidence that Cellpose-SAM's
  cell-specific prior remains stronger for this dataset under the tested budgets.
- In all cases, keep the v1 result in comparison tables.

## YOLO Threshold Diagnostic Result

`scripts/run_yolo_threshold_diagnostic.py` evaluates the frozen fixed-budget YOLO
v1 checkpoint on the same 134 held-out validation images without retraining. The
predeclared confidence grid is `0.05`, `0.10`, `0.25`, `0.40`, and `0.60`.

Outputs:

- `results/supervised/yolo_threshold_diagnostic_metrics.csv`
- `results/supervised/yolo_threshold_diagnostic_summary.csv`
- `figures/supervised_yolo_threshold_diagnostic_f1.png`
- `figures/supervised_yolo_threshold_diagnostic_count_error.png`

Summary:

| Confidence | Mean object F1 | Mean precision | Mean recall | Mean absolute count error |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.7607 | 0.6855 | 0.8826 | 16.5224 |
| 0.10 | 0.8023 | 0.7509 | 0.8794 | 10.3134 |
| 0.25 | 0.8530 | 0.8419 | 0.8737 | 6.0896 |
| 0.40 | 0.8695 | 0.8911 | 0.8566 | 5.7090 |
| 0.60 | 0.8412 | 0.9376 | 0.7774 | 10.6418 |

The best threshold in this diagnostic is `conf=0.40`, with mean object F1 0.8695.
It remains below Cellpose-SAM's 0.9200 mean object F1 on the same held-out
validation ids.

## YOLO Label-Budget Diagnostic Conversion

`scripts/prepare_yolo_label_budget_diagnostic.py` prepares nested label-budget
datasets for the training-side diagnostic without running training. The existing
100-image fixed-budget YOLO v1 result is reused as the first point on the budget
curve. The script only creates the two additional budgets:

- `budget_250`: the original 100 training image ids plus 150 additional train-pool
  images;
- `full_train_pool`: all 536 train-pool images.

Both budgets reuse the same 134 held-out validation image ids from Protocol B v1.
The nesting checks are part of the script: original 100 ids must be a subset of
`budget_250`, and `budget_250` must be a subset of `full_train_pool`.

Outputs:

- `results/supervised/yolo_label_budget_diagnostic_manifest.csv`
- `results/supervised/yolo_label_budget_diagnostic_split.csv`
- `results/supervised/yolo_label_budget_diagnostic_summary.csv`
- `results/supervised/yolo_label_budget_diagnostic/budget_250/`
- `results/supervised/yolo_label_budget_diagnostic/full_train_pool/`

Conversion summary:

| Budget | Train images | Val images | Train instances | Val instances | Dropped instances |
| --- | ---: | ---: | ---: | ---: | ---: |
| budget_250 | 250 | 134 | 11533 | 5599 | 0 |
| full_train_pool | 536 | 134 | 23862 | 5599 | 0 |

## YOLO Label-Budget Diagnostic Result

The label-budget diagnostic trains YOLO11n-seg on the fixed 100-image budget,
`budget_250`, and `full_train_pool`, then evaluates each run on the same 134
held-out validation images. The current diagnostic recipe is 50 epochs,
`imgsz=512`, `batch=8`, `workers=2`, AMP disabled, and repository metric evaluation
at `conf=0.25`.

Outputs:

- `results/supervised/yolo_label_budget_diagnostic_budget_250_train_metadata.csv`
- `results/supervised/yolo_label_budget_diagnostic_budget_250_train_summary.csv`
- `results/supervised/yolo_label_budget_diagnostic_budget_250_metrics.csv`
- `results/supervised/yolo_label_budget_diagnostic_budget_250_eval_summary.csv`
- `results/supervised/yolo_label_budget_diagnostic_full_train_pool_train_metadata.csv`
- `results/supervised/yolo_label_budget_diagnostic_full_train_pool_train_summary.csv`
- `results/supervised/yolo_label_budget_diagnostic_full_train_pool_metrics.csv`
- `results/supervised/yolo_label_budget_diagnostic_full_train_pool_eval_summary.csv`
- `results/supervised/yolo_label_budget_diagnostic_val_comparison_metrics.csv`
- `results/supervised/yolo_label_budget_diagnostic_val_comparison_summary.csv`
- `figures/supervised_yolo_label_budget_diagnostic_comparison.png`
- `figures/supervised_yolo_label_budget_diagnostic_budget_250_eval_overlays.png`
- `figures/supervised_yolo_label_budget_diagnostic_full_train_pool_eval_overlays.png`

Training took 351.421 seconds for the 100-image run, 571.402 seconds for
`budget_250`, and 975.039 seconds for `full_train_pool`.

Held-out validation comparison on the same 134 image ids:

| Method | Train images | Mean object F1 | Mean precision | Mean recall | Mean absolute count error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cellpose-SAM | 0 | 0.9200 | 0.9456 | 0.9007 | 2.9328 |
| YOLO label-budget full train pool | 536 | 0.8649 | 0.8440 | 0.8942 | 4.2090 |
| YOLO label-budget 250 | 250 | 0.8576 | 0.8400 | 0.8845 | 6.2090 |
| YOLO fixed-budget 100 | 100 | 0.8530 | 0.8419 | 0.8737 | 6.0896 |
| Otsu + watershed | 0 | 0.6442 | 0.6103 | 0.7219 | 19.8806 |

The full train-pool run improves over the smaller YOLO budgets, especially in mean
absolute count error, but it remains below Cellpose-SAM on mean object F1 and count
error. The label-budget diagnostic therefore supports supervised YOLO as a useful
Protocol B line, but label budget alone does not explain the main gap to
Cellpose-SAM.

## YOLO11m Capacity Diagnostic Result

The capacity diagnostic trains `YOLO11m-seg` on the same `full_train_pool` dataset
and evaluates on the same 134 held-out validation images. The purpose is to test
whether a substantially larger YOLO-seg model changes the label-budget conclusion.

Outputs:

- `results/supervised/yolo_capacity_diagnostic_yolo11m_train_metadata.csv`
- `results/supervised/yolo_capacity_diagnostic_yolo11m_train_summary.csv`
- `results/supervised/yolo_capacity_diagnostic_yolo11m_metrics.csv`
- `results/supervised/yolo_capacity_diagnostic_yolo11m_eval_summary.csv`
- `results/supervised/yolo_capacity_diagnostic_val_comparison_metrics.csv`
- `results/supervised/yolo_capacity_diagnostic_val_comparison_summary.csv`
- `figures/supervised_yolo_capacity_diagnostic_yolo11m_eval_overlays.png`
- `figures/supervised_yolo_capacity_diagnostic_comparison.png`

The run used `yolo11m-seg.pt`, 50 epochs, `imgsz=512`, `batch=8`, `workers=3`,
AMP disabled, and repository metric evaluation at `conf=0.25`. Training took
2827.244 seconds.

Held-out validation comparison on the same 134 image ids:

| Method | Train images | Mean object F1 | Mean precision | Mean recall | Mean absolute count error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cellpose-SAM | 0 | 0.9200 | 0.9456 | 0.9007 | 2.9328 |
| YOLO11m full train pool | 536 | 0.8680 | 0.8525 | 0.8921 | 4.8582 |
| YOLO11n full train pool | 536 | 0.8649 | 0.8440 | 0.8942 | 4.2090 |
| Otsu + watershed | 0 | 0.6442 | 0.6103 | 0.7219 | 19.8806 |

YOLO11m gives only a small mean object-F1 increase over YOLO11n and does not improve
mean absolute count error. This capacity probe does not close the gap to
Cellpose-SAM, so there is no current need to add an intermediate YOLO11s run as a
main PoW result.
