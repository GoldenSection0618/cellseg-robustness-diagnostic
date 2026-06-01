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
- `figures/baseline_clean_subset_metric_comparison.png`
- `figures/baseline_clean_subset_count_error_comparison.png`
- `figures/baseline_clean_subset_latency_comparison.png`

## Phase 7: Failure Taxonomy

Goal: name the first set of visible failure modes so later qualitative examples can
be annotated consistently.

Current output:

- `docs/failure_taxonomy.md`

## Deferred Robustness Work

Robustness perturbations are a later analysis track. A small Otsu-only smoke test has
already been recorded, but the current main line should continue with clean baseline
completion first.

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
- Cellpose-SAM full protocol;
- optional SAM2 AMG parameter-sensitivity smoke test;
- YOLO-seg small-label supervised adaptation;
- Gemini segmentation output-validity checks.
