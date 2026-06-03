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
- optional SAM2 AMG parameter-sensitivity smoke test;
- YOLO-seg small-label supervised adaptation;
- Gemini segmentation output-validity checks.
