# Historical Experiment Plan and Execution Record

> **Status: historical record.** This document records the incremental plan and the
> evidence produced as the benchmark was built. It is not the current project task
> list. The reader-facing results are in the [README](../README.md); the completed
> zero-shot and supervised protocols are documented in
> [pow_report.md](pow_report.md) and [supervised_protocol.md](supervised_protocol.md).

The benchmark was built in small increments. Each stage was intended to leave
verifiable outputs and a commit before the next stage. Phases 1--11 and the completed
YOLO entries below describe finished work. Unchecked cross-version, prompted-SAM2,
and VLM entries are deliberately separate follow-up protocols, not missing
requirements for the reported results.

## Phase 1: Data Audit

Goal: prove that the local DSB2018 data can be read consistently.

Outputs:

- `results/dataset/dataset_inventory.csv`
- `results/dataset/dataset_summary.csv`
- `figures/dataset_split_counts.png`
- `figures/dataset_train_instance_count_hist.png`
- `figures/dataset_image_size_scatter.png`

## Phase 2: Evaluation Pipeline

Goal: stabilize the shared instance-mask evaluation contract before comparing
methods.

Required metrics:

- object-level F1;
- mean matched IoU;
- Dice;
- count error;
- false positives and missed objects.

## Phase 3: First Baseline

Goal: run an Otsu + watershed baseline on a deterministic 20-image subset.

Outputs:

- `results/baselines/otsu_watershed_clean_subset_metrics.csv`
- `figures/otsu_watershed_subset_overlay_examples.png`
- `figures/baseline_clean_subset_count_agreement.png`

## Phase 4: Second Baseline

Goal: run Cellpose-SAM / `cpsam` on the same deterministic 20-image subset.

Completed outputs:

- `results/baselines/cellpose_cpsam_clean_subset_metrics.csv`
- `figures/cellpose_cpsam_subset_overlay_examples.png`
- `figures/baseline_clean_subset_count_agreement.png`

## Phase 5: Third Baseline

Goal: run SAM2 automatic mask generation on the same deterministic 20-image subset.

Local model asset:

- `data/checkpoints/sam2.1_hiera_large.pt`

Completed outputs:

- `results/baselines/sam2_amg_clean_subset_metrics.csv`
- `figures/sam2_amg_subset_overlay_examples.png`
- `figures/baseline_clean_subset_count_agreement.png`

## Phase 6: Clean Baseline Comparison

Goal: compare completed clean subset baselines without adding perturbations or
full-scale runs.

Current completed comparison:

- `results/baselines/clean_subset_baseline_metrics_long.csv`
- `results/baselines/clean_subset_baseline_summary.csv`
- `results/baselines/clean_subset_baseline_failure_cases.csv`
- `figures/baseline_clean_subset_metric_comparison.png`
- `figures/baseline_clean_subset_count_error_comparison.png`
- `figures/baseline_clean_subset_latency_comparison.png`
- `figures/baseline_clean_subset_score_distributions.png`
- `figures/baseline_clean_subset_precision_recall.png`
- `figures/supplementary_baseline_clean_subset_image_method_f1_heatmap.png`

## Phase 7: Failure Taxonomy

Goal: name the first set of visible failure modes so later qualitative examples can
be annotated consistently.

Current output:

- `docs/failure_taxonomy.md`

## Phase 8: PoW Robustness Smoke Test

Goal: test the three completed PoW baselines on the same tiny perturbation set before
considering any full robustness sweep.

Current outputs:

- `results/robustness/pow_baseline_robustness_smoke_metrics.csv`
- `results/robustness/pow_baseline_robustness_smoke_summary.csv`
- `figures/robustness_pow_smoke_mean_f1.png`
- `figures/robustness_pow_smoke_relative_f1_drop.png`
- `figures/robustness_pow_smoke_overlay_examples.png`

## Phase 9: PoW Robustness Clean20 Extension

Goal: run the same perturbation set across the deterministic 20-image clean subset
used by the completed clean baselines.

Current outputs:

- `results/robustness/pow_baseline_robustness_clean20_metrics.csv`
- `results/robustness/pow_baseline_robustness_clean20_summary.csv`
- `results/robustness/pow_baseline_robustness_clean20_image_deltas.csv`
- `results/robustness/pow_baseline_robustness_clean20_failure_cases.csv`
- `figures/robustness_pow_clean20_summary.png`
- `figures/robustness_pow_clean20_failure_diagnostics.png`
- `figures/robustness_pow_clean20_overlay_examples.png`

This targeted PoW extension is complete and has been superseded by the staged
full-train robustness run for Otsu + watershed and Cellpose-SAM.

## Phase 10: Staged Full-Train Robustness

Goal: extend the initial five-condition robustness protocol to all 670
`stage1_train` images, one method at a time. The completed full-train report later
adds Poisson noise and intensity scaling, for six perturbations plus the clean
condition.

Current Otsu + watershed and Cellpose-SAM outputs:

- `results/robustness/pow_baseline_robustness_full_train_metrics.csv`
- `results/robustness/pow_baseline_robustness_full_train_summary.csv`
- `results/robustness/pow_baseline_robustness_full_train_image_deltas.csv`
- `results/robustness/pow_baseline_robustness_full_train_failure_cases.csv`
- `results/robustness/pow_baseline_robustness_full_train_no_prediction_cases.csv`
- `figures/robustness_pow_full_train_summary.png`
- `figures/robustness_pow_full_train_failure_diagnostics.png`
- `figures/robustness_pow_full_train_overlay_examples.png`

The full-train output currently contains `otsu_watershed` and `cellpose_cpsam`.
SAM2 AMG full-train execution is deferred because clean20 already shows near-total
collapse under all tested perturbations, and the completed clean20 parameter
sensitivity run did not repair that pattern. The next SAM2 step should be a protocol
change such as prompted SAM2 or post-processing repair, not a full-train run with
the current AMG settings.

## Phase 11: SAM2 AMG Parameter Sensitivity

Goal: test whether simple SAM2 AMG parameter changes can repair the clean20
perturbation collapse before considering any full-train SAM2 run.

Generated outputs:

- `results/robustness/sam2_amg_sensitivity_clean20_clean_screen_metrics.csv`
- `results/robustness/sam2_amg_sensitivity_clean20_clean_screen_summary.csv`
- `results/robustness/sam2_amg_sensitivity_clean20_clean_screen_failed_configs.csv`
- `results/robustness/sam2_amg_sensitivity_clean20_validation_metrics.csv`
- `results/robustness/sam2_amg_sensitivity_clean20_validation_summary.csv`
- `results/robustness/sam2_amg_sensitivity_clean20_failure_cases.csv`
- `figures/robustness_sam2_amg_sensitivity_clean20_clean_screen_f1.png`
- `figures/robustness_sam2_amg_sensitivity_clean20_clean_screen_counts.png`
- `figures/robustness_sam2_amg_sensitivity_clean20_mean_f1.png`
- `figures/robustness_sam2_amg_sensitivity_clean20_count_error.png`

Result: the sensitivity run does not justify expanding SAM2 AMG to full_train.
`stability_score_thresh_0.95` is the best clean setting, but blur and downsample
remain near collapse and Gaussian noise remains weak. The future SAM2 path should
change protocol, not scale the current AMG settings.

## Archived Optional Cross-Version Cellpose Work

Legacy Cellpose3 `cyto3` and one-click restoration are not required for the current
PoW mainline. They may be added later as optional cross-version baselines using a
separate environment or exact legacy model assets.

Current environment status:

- `results/baselines/cellpose_method_availability.csv` records that Cellpose default
  candidate names resolve to `cpsam` under `cellpose==4.1.1`.
- The same audit records that the current restoration API fails during initialization
  with `NameError: name 'CPnet' is not defined`.

## Separate Follow-up Protocols

These protocols should stay separate from the clean zero-shot baseline track:

- optional Cellpose3 default cross-version protocol;
- optional Cellpose3 restoration cross-version protocol;
- optional prompted SAM2 or SAM2 post-processing repair protocol;
- optional SAM3 prompted-concept screening with predeclared text or exemplar prompts;
- YOLO-seg small-label supervised adaptation;
- Vision Banana generative mask-output validity checks; other generative vision
  models may be assessed under the same separate protocol.

## Protocol B: YOLO Supervised Adaptation

Goal: evaluate how much target-domain supervised training can improve over the
zero-shot baselines without mixing results into the zero-shot ranking.

Current completed setup step:

- `docs/supervised_protocol.md`
- `scripts/prepare_yolo_label_smoke.py`
- `results/supervised/yolo_label_smoke/labels/*.txt`
- `results/supervised/yolo_label_smoke/images.txt`
- `results/supervised/yolo_label_smoke/train.txt`
- `results/supervised/yolo_label_smoke/val.txt`
- `results/supervised/yolo_label_smoke/data.yaml`
- `results/supervised/yolo_label_smoke_manifest.csv`
- `results/supervised/yolo_label_smoke_summary.csv`
- `figures/supervised_yolo_label_smoke_overlays.png`

The label-conversion smoke uses the deterministic 20-image subset. It converts 990
instances to YOLO segmentation polygons with zero dropped instances; two tiny
instances use the documented rectangle fallback.

This label-conversion smoke was followed by a tiny YOLO training smoke. A full
supervised baseline should not run until the tiny train/predict/evaluate loop is
verified.

Tiny YOLO training smoke outputs:

- `scripts/run_yolo_tiny_train_smoke.py`
- `results/supervised/yolo_tiny_train_smoke_metadata.csv`
- `results/supervised/yolo_tiny_train_smoke_summary.csv`

The tiny smoke used pretrained `yolo11n-seg.pt` from `model_assets/yolo/`, 1 epoch,
`imgsz=256`, `batch=2`, and the 16/4 train/val split from the label smoke. It is a
pipeline check, not a supervised baseline result. It was followed by prediction
export and conversion back to repository instance metrics.

Tiny YOLO prediction evaluation outputs:

- `scripts/evaluate_yolo_tiny_train_smoke.py`
- `results/supervised/yolo_tiny_train_smoke_metrics.csv`
- `results/supervised/yolo_tiny_train_smoke_eval_summary.csv`
- `figures/supervised_yolo_tiny_train_smoke_overlays.png`

The evaluation smoke confirms that YOLO predictions can be converted back to
repository instance masks and scored with object F1, precision, recall, matched IoU,
Dice, and count error. The 1-epoch smoke model has mean object F1 0.0 on the four
validation images, so this is only a pipeline check. The next Protocol B decision
was to run a longer supervised baseline with a fixed training budget.

Fixed baseline decision:

- train on 100 labeled images sampled evenly from the deterministic 80% training
  pool of `stage1_train`;
- validate on the held-out 20% pool, expected 134 images;
- use pretrained `model_assets/yolo/yolo11n-seg.pt`;
- train for 50 epochs at `imgsz=512`, `batch=8`, `workers=2`, AMP disabled, and
  patience set to the epoch budget;
- evaluate with repository instance metrics on the same held-out validation ids;
- compare supervised and zero-shot methods only on that same validation image set.

Fixed-budget label conversion outputs:

- `scripts/prepare_yolo_fixed_budget.py`
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

Result: the fixed-budget split has 100 train images, 134 held-out validation images,
and 436 unused training-pool images. The selected train/validation labels contain
10,766 converted polygons, 0 dropped instances, and 7 tiny-mask rectangle fallbacks.

Fixed-budget supervised baseline outputs:

- `scripts/run_yolo_fixed_budget_train.py`
- `scripts/evaluate_yolo_fixed_budget.py`
- `scripts/summarize_yolo_fixed_budget_comparison.py`
- `scripts/run_yolo_threshold_diagnostic.py`
- `scripts/prepare_yolo_label_budget_diagnostic.py`
- `results/supervised/yolo_fixed_budget_train_metadata.csv`
- `results/supervised/yolo_fixed_budget_train_summary.csv`
- `results/supervised/yolo_fixed_budget_metrics.csv`
- `results/supervised/yolo_fixed_budget_eval_summary.csv`
- `results/supervised/yolo_fixed_budget_val_comparison_metrics.csv`
- `results/supervised/yolo_fixed_budget_val_comparison_summary.csv`
- `results/supervised/yolo_threshold_diagnostic_metrics.csv`
- `results/supervised/yolo_threshold_diagnostic_summary.csv`
- `results/supervised/yolo_label_budget_diagnostic_manifest.csv`
- `results/supervised/yolo_label_budget_diagnostic_split.csv`
- `results/supervised/yolo_label_budget_diagnostic_summary.csv`
- `results/supervised/yolo_label_budget_diagnostic/budget_250/`
- `results/supervised/yolo_label_budget_diagnostic/full_train_pool/`
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
- `figures/supervised_yolo_fixed_budget_eval_overlays.png`
- `figures/supervised_yolo_threshold_diagnostic_f1.png`
- `figures/supervised_yolo_threshold_diagnostic_count_error.png`
- `figures/supervised_yolo_label_budget_diagnostic_budget_250_eval_overlays.png`
- `figures/supervised_yolo_label_budget_diagnostic_full_train_pool_eval_overlays.png`

Result: the fixed-budget YOLO run used 100 train images, 134 held-out validation
images, 50 epochs, `imgsz=512`, `batch=8`, `workers=2`, AMP disabled, and
`conf=0.25` for repository-metric evaluation. On the same held-out validation ids,
mean object F1 is 0.9200 for Cellpose-SAM, 0.8530 for YOLO fixed-budget supervised,
and 0.6442 for Otsu + watershed. YOLO improves clearly over the classical lower
bound but remains below Cellpose-SAM under the repository object-level metrics.

Current interpretation: the fixed-budget supervised result is valid Protocol B v1.
It should be retained as the first-pass YOLO fine-tuning baseline, not replaced by
later optimization experiments.

Protocol B follow-up decision:

The fixed-budget YOLO result is retained as Protocol B v1 and should not be
overwritten. Follow-up work is justified only as diagnostic extension, not as a
replacement for a valid result. The motivating concern is that direct YOLO-seg
fine-tuning may be under-adapted to dense cell instance segmentation or under-trained
under the 100-image fixed budget. The recommended sequence is:

- operating-point diagnostic on the frozen v1 checkpoint;
- label-budget diagnostic with predeclared budgets;
- model-capacity diagnostic with a larger YOLO-seg model;
- optional predeclared post-processing diagnostic.

The purpose is to identify why YOLO remains below Cellpose-SAM under repository
metrics, not to tune until YOLO wins.

First follow-up result: the frozen v1 checkpoint was evaluated on the predeclared
confidence grid `0.05`, `0.10`, `0.25`, `0.40`, and `0.60` over the same 134
held-out validation images. The best mean object F1 is 0.8695 at `conf=0.40`,
compared with 0.8530 for the `conf=0.25` operating point and 0.9200 for
Cellpose-SAM on the same image ids. This excludes a poor confidence threshold as the
main explanation, so the next diagnostic directly tested the original training-side
concern: label budget first, then model capacity.

The label-budget diagnostic is now trained and evaluated for the 100-image
fixed-budget point, `budget_250`, and `full_train_pool`. The two larger budgets are
nested: `budget_250` contains the original 100 training image ids plus 150
additional train-pool images, while `full_train_pool` contains all 536 train-pool
images and therefore contains `budget_250`. All budgets reuse the same 134 held-out
validation image ids and repository evaluation at `conf=0.25`.

Current held-out validation comparison:

| Method | Protocol | Mean object F1 | Mean precision | Mean recall | Mean absolute count error |
| --- | --- | ---: | ---: | ---: | ---: |
| Cellpose-SAM | zero-shot | 0.9200 | 0.9456 | 0.9007 | 2.9328 |
| YOLO supervised | supervised | 0.8680 | 0.8525 | 0.8921 | 4.8582 |
| Otsu + watershed | zero-shot | 0.6442 | 0.6103 | 0.7219 | 19.8806 |

The full YOLO diagnostic curves are kept in `docs/supervised_protocol.md`; the main
comparison reports the strongest completed YOLO supervised result by mean object F1.

The YOLO11m capacity probe has also been trained on the same 536-image full train
pool and evaluated on the same 134 held-out validation images. YOLO11m reaches mean
object F1 0.8680 and mean absolute count error 4.8582, compared with YOLO11n full
train-pool F1 0.8649 and count error 4.2090. This does not close the gap to
Cellpose-SAM. Detailed capacity outputs and figures are recorded in
`docs/supervised_protocol.md`.
