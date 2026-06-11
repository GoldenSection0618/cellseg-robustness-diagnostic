# Experiment Plan

The benchmark is built in small increments. Each small stage should leave verifiable
outputs and be committed before starting the next stage.

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
- `figures/otsu_watershed_subset_metric_means.png`
- `figures/otsu_watershed_subset_count_scatter.png`

## Phase 4: Second Baseline

Goal: run Cellpose-SAM / `cpsam` on the same deterministic 20-image subset.

Completed outputs:

- `results/baselines/cellpose_cpsam_clean_subset_metrics.csv`
- `figures/cellpose_cpsam_subset_overlay_examples.png`
- `figures/cellpose_cpsam_subset_metric_means.png`
- `figures/cellpose_cpsam_subset_count_scatter.png`

## Phase 5: Third Baseline

Goal: run SAM2 automatic mask generation on the same deterministic 20-image subset.

Local model asset:

- `data/checkpoints/sam2.1_hiera_large.pt`

Completed outputs:

- `results/baselines/sam2_amg_clean_subset_metrics.csv`
- `figures/sam2_amg_subset_overlay_examples.png`
- `figures/sam2_amg_subset_metric_means.png`
- `figures/sam2_amg_subset_count_scatter.png`

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
- `figures/baseline_clean_subset_image_method_f1_heatmap.png`

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
- `figures/robustness_pow_smoke_method_condition_heatmap.png`
- `figures/robustness_pow_smoke_overlay_examples.png`

## Phase 9: PoW Robustness Clean20 Extension

Goal: run the same perturbation set across the deterministic 20-image clean subset
used by the completed clean baselines.

Current outputs:

- `results/robustness/pow_baseline_robustness_clean20_metrics.csv`
- `results/robustness/pow_baseline_robustness_clean20_summary.csv`
- `results/robustness/pow_baseline_robustness_clean20_image_deltas.csv`
- `results/robustness/pow_baseline_robustness_clean20_failure_cases.csv`
- `figures/robustness_pow_clean20_mean_f1.png`
- `figures/robustness_pow_clean20_relative_f1_drop.png`
- `figures/robustness_pow_clean20_method_condition_heatmap.png`
- `figures/robustness_pow_clean20_image_f1_drop_heatmap.png`
- `figures/robustness_pow_clean20_worst_f1_drops.png`
- `figures/robustness_pow_clean20_overlay_examples.png`

This targeted PoW extension is complete and has been superseded by the staged
full-train robustness run for Otsu + watershed and Cellpose-SAM.

## Phase 10: Staged Full-Train Robustness

Goal: extend the same five-condition robustness protocol to all 670 `stage1_train`
images, one method at a time.

Current Otsu + watershed and Cellpose-SAM outputs:

- `results/robustness/pow_baseline_robustness_full_train_metrics.csv`
- `results/robustness/pow_baseline_robustness_full_train_summary.csv`
- `results/robustness/pow_baseline_robustness_full_train_image_deltas.csv`
- `results/robustness/pow_baseline_robustness_full_train_failure_cases.csv`
- `results/robustness/pow_baseline_robustness_full_train_no_prediction_cases.csv`
- `figures/robustness_pow_full_train_mean_f1.png`
- `figures/robustness_pow_full_train_relative_f1_drop.png`
- `figures/robustness_pow_full_train_method_condition_heatmap.png`
- `figures/robustness_pow_full_train_f1_drop_distributions.png`
- `figures/robustness_pow_full_train_worst_f1_drops.png`
- `figures/robustness_pow_full_train_failure_hint_counts.png`
- `figures/robustness_pow_full_train_overlay_examples.png`

The full-train output currently contains `otsu_watershed` and `cellpose_cpsam`.
SAM2 AMG full-train execution is deferred because clean20 already shows near-total
collapse under all tested perturbations. The next SAM2 step should be parameter
sensitivity or post-processing repair, not a full-train run with the current AMG
settings.

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
- `figures/robustness_sam2_amg_sensitivity_clean20_zero_pred_rate.png`
- `figures/robustness_sam2_amg_sensitivity_clean20_count_error.png`

Result: the sensitivity run does not justify expanding SAM2 AMG to full_train.
`stability_score_thresh_0.95` is the best clean setting, but blur and downsample
remain near collapse and Gaussian noise remains weak. The future SAM2 path should
change protocol, not scale the current AMG settings.

## Optional Cross-Version Cellpose Work

Legacy Cellpose3 `cyto3` and one-click restoration are not required for the current
PoW mainline. They may be added later as optional cross-version baselines using a
separate environment or exact legacy model assets.

Current environment status:

- `results/baselines/cellpose_method_availability.csv` records that Cellpose default
  candidate names resolve to `cpsam` under `cellpose==4.1.1`.
- The same audit records that the current restoration API fails during initialization
  with `NameError: name 'CPnet' is not defined`.

## Later Protocols

These protocols should stay separate from the clean zero-shot baseline track:

- optional Cellpose3 default cross-version protocol;
- optional Cellpose3 restoration cross-version protocol;
- optional Cellpose-SAM protocol refinements after the current full-train run;
- optional prompted SAM2 or SAM2 post-processing repair protocol;
- YOLO-seg small-label supervised adaptation;
- Gemini segmentation output-validity checks.

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

Next Protocol B step: inspect the overlay figure and then run a tiny YOLO training
smoke. Do not run a full supervised baseline until the tiny train/predict/evaluate
loop is verified.
