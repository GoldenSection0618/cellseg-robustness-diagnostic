# Technical Memo

## Current Objective

Build a compact proof-of-work benchmark for microscopy nucleus instance segmentation
robustness. The first implementation target is an end-to-end skeleton that can:

1. read the local DSB2018 data;
2. produce tabular analysis outputs in categorized `results/` subdirectories;
3. produce visual analysis outputs in `figures/`;
4. support later predictors through a common evaluation pipeline.

## Dataset

The current dataset is the Kaggle 2018 Data Science Bowl dataset, unpacked under
`data/raw/dsb2018/`.

The primary metric-bearing data for the first benchmark pass is:

- `stage1_train/`: images plus per-instance PNG masks;
- `stage1_test/`: held-out images;
- `solutions/stage1_solution.csv`: RLE masks for stage 1 test.

`stage2_test_final/` is retained locally but is not part of the first metric-bearing
evaluation because it does not include local ground-truth masks.

## First Milestone

The first milestone is intentionally small:

- audit dataset structure and image/mask counts;
- write audit tables to `results/dataset/`;
- write dataset diagnostic plots to `figures/`;
- keep all generated outputs reproducible from code.

After this milestone, the next step is to add a classical Otsu + watershed baseline
that uses the same output contract.

## Baseline Smoke Test

The first minimum baseline is an Otsu + watershed run on a deterministic 20-image
subset from `stage1_train/`.

Generated outputs:

- `results/baselines/otsu_watershed_clean_subset_metrics.csv`
- `figures/otsu_watershed_subset_overlay_examples.png`
- `figures/otsu_watershed_subset_metric_means.png`
- `figures/otsu_watershed_subset_count_scatter.png`

Current subset-level summary:

- images: 20
- mean object F1: 0.4685
- mean matched IoU: 0.7307
- mean absolute count error: 63.75

This result is a smoke test of the experiment path, not a tuned classical baseline.
The large count errors are expected for an untuned threshold/watershed pipeline on
heterogeneous DSB2018 images.

## Cellpose-SAM Smoke Test

The next minimum baseline uses Cellpose 4.1.1 with the `cpsam` pretrained model on
the same deterministic 20-image clean subset.

Generated outputs:

- `results/baselines/cellpose_cpsam_clean_subset_metrics.csv`
- `figures/cellpose_cpsam_subset_overlay_examples.png`
- `figures/cellpose_cpsam_subset_metric_means.png`
- `figures/cellpose_cpsam_subset_count_scatter.png`

Current subset-level summary:

- images: 20
- mean object F1: 0.8892
- mean matched IoU: 0.8513
- mean absolute count error: 7.15
- mean latency: 1118.71 ms/image

This is still a smoke test on a small subset. It establishes that the model-backed
baseline path can produce the same metric and figure contract as the classical
baseline.

## Clean Subset Baseline Comparison

The first comparison analysis combines the Otsu + watershed and Cellpose-SAM clean
subset outputs without running any additional model inference.

Generated outputs:

- `results/baselines/clean_subset_baseline_metrics_long.csv`
- `results/baselines/clean_subset_baseline_summary.csv`
- `figures/baseline_clean_subset_metric_comparison.png`
- `figures/baseline_clean_subset_count_error_comparison.png`
- `figures/baseline_clean_subset_latency_comparison.png`

Current comparison summary:

| Method | Images | Mean object F1 | Mean matched IoU | Mean absolute count error | Median latency ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cellpose-SAM | 20 | 0.8892 | 0.8513 | 7.15 | 687.13 |
| Otsu + watershed | 20 | 0.4685 | 0.7307 | 63.75 | 15.72 |

This comparison is a smoke-test analysis product. It shows that baseline outputs can
be aggregated and visualized under the project output contract before expanding to
more perturbations or larger evaluation sets.

## Output Contract

Experiment and analysis scripts should write:

- CSV tables to categorized `results/` subdirectories;
- PNG figures to `figures/`;
- no generated result files into `docs/` or source directories.

Important tabular outputs should be grouped by protocol or analysis category:

- `results/dataset/`
- `results/baselines/`
- `results/robustness/`
- `results/supervised/`
- `results/vlm/`

Each generated CSV should have enough columns to identify the input split, image id,
method or analysis name, and any relevant parameters.

Each generated figure should be produced by a script and saved directly under
`figures/` with a clear, descriptive filename. Figures are intentionally not
subdivided into nested folders.

## Current Scope Boundary

This memo tracks proof-of-work engineering progress. It is not a final paper-style
report. Model comparisons, robustness conclusions, and failure taxonomy should be
added only after the evaluation pipeline is running end to end.
