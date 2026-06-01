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

Goal: run the Cellpose default pretrained model on the same deterministic 20-image
subset.

Current environment status:

- `results/baselines/cellpose_method_availability.csv` records that Cellpose default
  candidate names resolve to `cpsam` under `cellpose==4.1.1`, so this baseline needs
  a separate environment or model-asset decision before running.

Expected outputs:

- metrics under `results/baselines/`;
- qualitative figures under `figures/`;
- summary notes in `technical_memo.md`.

## Phase 5: Third Baseline

Goal: run the Cellpose restoration-enhanced workflow on the same deterministic
20-image subset.

Current environment status:

- `results/baselines/cellpose_method_availability.csv` records that the current
  restoration API fails during initialization with `NameError: name 'CPnet' is not
  defined`, so this baseline also needs an environment decision before running.

Expected outputs:

- metrics under `results/baselines/`;
- qualitative figures under `figures/`;
- summary notes in `technical_memo.md`.

## Phase 6: Fourth Baseline

Goal: run Cellpose-SAM / `cpsam` on the same deterministic 20-image subset.

Completed outputs:

- `results/baselines/cellpose_cpsam_clean_subset_metrics.csv`
- `figures/cellpose_cpsam_subset_overlay_examples.png`
- `figures/cellpose_cpsam_subset_metric_means.png`
- `figures/cellpose_cpsam_subset_count_scatter.png`

## Phase 7: Fifth Baseline

Goal: run SAM2 automatic mask generation on the same deterministic 20-image subset.

Local model asset:

- `data/checkpoints/sam2.1_hiera_large.pt`

Completed outputs:

- `results/baselines/sam2_amg_clean_subset_metrics.csv`
- `figures/sam2_amg_subset_overlay_examples.png`
- `figures/sam2_amg_subset_metric_means.png`
- `figures/sam2_amg_subset_count_scatter.png`

## Phase 8: Clean Baseline Comparison

Goal: compare completed clean subset baselines without adding perturbations or
full-scale runs.

Current completed comparison:

- `results/baselines/clean_subset_baseline_metrics_long.csv`
- `results/baselines/clean_subset_baseline_summary.csv`
- `figures/baseline_clean_subset_metric_comparison.png`
- `figures/baseline_clean_subset_count_error_comparison.png`
- `figures/baseline_clean_subset_latency_comparison.png`

## Deferred Robustness Work

Robustness perturbations are a later analysis track. A small Otsu-only smoke test has
already been recorded, but the current main line should continue with clean baseline
completion first.

## Later Protocols

These protocols should stay separate from the clean zero-shot baseline track:

- Cellpose default protocol;
- Cellpose-SAM full protocol;
- YOLO-seg small-label supervised adaptation;
- Gemini segmentation output-validity checks.
