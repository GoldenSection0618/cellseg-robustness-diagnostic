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

The first training run should be a tiny smoke run. The fixed-budget supervised
baseline should only run after label conversion, prediction export, and metric
conversion are all verified.

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
- batch size: 4, reduced only if CUDA memory requires it;
- workers: 0 for reproducibility in this environment;
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
validation images, 50 epochs, `imgsz=512`, `batch=4`, `workers=0`, AMP disabled,
and `patience=50`. Training completed in 533.474 seconds on the local RTX 4060
Laptop GPU, with peak allocated CUDA memory of 1706.18 MB.

The primary repository-metric evaluation uses `conf=0.25`, the conventional
Ultralytics prediction operating point. On the held-out validation split, YOLO
fixed-budget supervised reaches mean object F1 0.8571, precision 0.8494, recall
0.8734, mean matched IoU 0.8230, mean matched Dice 0.8990, and mean absolute count
error 6.7612.

On the same 134 held-out validation image ids, the clean zero-shot comparison is:

- Cellpose-SAM: mean object F1 0.9100, mean absolute count error 3.1194;
- YOLO fixed-budget supervised: mean object F1 0.8571, mean absolute count error
  6.7612;
- Otsu + watershed: mean object F1 0.6442, mean absolute count error 19.8806.

This means the fixed-budget YOLO baseline is a strong supervised result relative to
the classical lower bound, but it does not replace Cellpose-SAM as the strongest
current baseline under the repository object-level metrics.

## YOLO Follow-up Diagnostic Plan

The fixed-budget YOLO result is a valid completed baseline and must remain frozen as
Protocol B v1. It should not be replaced, renamed, or overwritten because it did not
overtake Cellpose-SAM. A corrective rerun is justified only if a pre-result logic
defect is found, such as data leakage, broken label conversion, an incorrect
checkpoint, a mask-conversion bug, or a clearly inappropriate evaluation setting.

The `conf=0.001` value inherited from the tiny smoke path was one such evaluation
logic issue: it was useful for confirming that predictions could be produced, but it
was not an appropriate primary operating point for a trained YOLO baseline. The
recorded v1 result uses `conf=0.25`. Further changes should be treated as separately
named follow-up diagnostics, not corrections to v1.

Follow-up diagnostics should answer narrower questions:

1. Is YOLO performance mainly limited by the confidence operating point?
2. Is it mainly limited by the 100-image label budget?
3. Is it mainly limited by the nano model capacity?
4. Is it mainly limited by YOLO-seg's fit to dense microscopy instance masks?

Guardrails:

- keep the 134-image held-out validation ids fixed for final comparison;
- keep Protocol B v1 outputs unchanged and make follow-up output names explicit;
- do not tune repeatedly on the held-out validation split and then report only the
  best run as if it were a new baseline;
- report all attempted follow-up runs, including no-improvement results;
- compare against Cellpose-SAM and Otsu only on the same image ids used by the
  follow-up evaluation;
- stop each diagnostic axis after the predeclared small set of runs.

Recommended follow-up sequence:

1. Operating-point diagnostic. Train nothing new. Evaluate the frozen v1 checkpoint
   on a small predeclared confidence grid, for example `0.05`, `0.10`, `0.25`,
   `0.40`, and `0.60`. This should be reported as threshold sensitivity, not as a
   replacement for v1. If a separate calibration split is created from the training
   pool, the 134-image held-out validation split should be used only for final
   readout.
2. Label-budget diagnostic. Keep model size and training recipe fixed, then train
   predeclared budgets such as 100, 250, and the full 536-image training pool. This
   answers whether the gap to Cellpose-SAM is mainly label-budget limited.
3. Model-capacity diagnostic. Keep the 100-image budget fixed and compare
   `yolo11n-seg` against one larger model such as `yolo11s-seg`, using the same
   training/evaluation contract. Larger models should be attempted only if GPU memory
   and runtime remain practical.
4. Post-processing diagnostic. Only after the previous diagnostics, test a small
   predeclared mask filtering rule, such as minimum area or confidence filtering, and
   record whether it mainly reduces false positives or also removes true cells.

The intended outcome is not to make YOLO win by repeated tuning. The intended
outcome is to determine whether the v1 gap to Cellpose-SAM is explained by budget,
capacity, operating point, or method fit.

Suggested output names:

- `results/supervised/yolo_threshold_diagnostic_metrics.csv`
- `results/supervised/yolo_threshold_diagnostic_summary.csv`
- `results/supervised/yolo_label_budget_diagnostic_summary.csv`
- `results/supervised/yolo_model_capacity_diagnostic_summary.csv`
- `results/supervised/yolo_postprocessing_diagnostic_summary.csv`

Interpretation rules:

- If threshold sensitivity closes most of the gap, report YOLO as operating-point
  sensitive rather than claiming the v1 baseline was invalid.
- If larger label budgets improve monotonically, report the v1 gap as label-budget
  limited.
- If a larger YOLO model improves at the same 100-image budget, report a capacity
  limitation.
- If none of these axes closes the gap, report evidence that Cellpose-SAM's
  cell-specific prior remains stronger for this dataset under the tested budgets.
- In all cases, keep the v1 result in comparison tables.
