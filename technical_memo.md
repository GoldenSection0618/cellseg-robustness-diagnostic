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

## Clean Baseline Continuation

README Protocol A is narrowed for the current PoW to three clean zero-shot baselines:
Otsu + watershed, Cellpose-SAM / `cpsam`, and SAM2 automatic mask generation.

Three clean subset baselines have already been recorded:

- Otsu + watershed;
- Cellpose-SAM / `cpsam`;
- SAM2 automatic mask generation.

Legacy Cellpose3 `cyto3` and Cellpose3 one-click restoration are future optional
cross-version baselines, not required PoW mainline work. In the current
`cellpose==4.1.1` environment, `CellposeModel` exposes only `cpsam` as a registered
segmentation model, so treating `cyto3` as a required baseline would either duplicate
Cellpose-SAM or force a separate environment track.

Each clean baseline should follow the same output contract:

- metrics in `results/baselines/`;
- qualitative figures in `figures/`;
- summary notes in this memo after the run.

## Cellpose Method Availability Audit

Legacy Cellpose default and restoration candidates were audited against the current
`cell` environment before deciding whether to add them as optional cross-version
baselines.

Generated outputs:

- `results/baselines/cellpose_method_availability.csv`
- `figures/cellpose_method_availability.png`

Current environment finding:

- `cellpose==4.1.1` exposes `cpsam` as the only registered
  `cellpose.models.CellposeModel` model.
- Candidate default names `cyto3`, `nuclei`, and `transformer_cp3` all resolve to
  `cpsam`, so running them in this environment would duplicate the Cellpose-SAM
  baseline rather than produce a distinct Cellpose default result.
- `cellpose.denoise.CellposeDenoiseModel` lists one-click restoration model names, but
  initialization currently fails with `NameError: name 'CPnet' is not defined`.

This audit supports keeping the PoW Cellpose-family mainline narrowed to `cpsam`.
Legacy Cellpose3 default and restoration should only be added later if a separate
cross-version environment or exact legacy model assets are deliberately introduced.

## SAM2 AMG Smoke Test

The SAM2 automatic mask generator baseline uses `sam2==1.1.0`,
`configs/sam2.1/sam2.1_hiera_l.yaml`, and the local checkpoint
`data/checkpoints/sam2.1_hiera_large.pt` on the same deterministic 20-image clean
subset.

Generated outputs:

- `results/baselines/sam2_amg_clean_subset_metrics.csv`
- `figures/sam2_amg_subset_overlay_examples.png`
- `figures/sam2_amg_subset_metric_means.png`
- `figures/sam2_amg_subset_count_scatter.png`

Current subset-level summary:

- images: 20
- mean object F1: 0.3604
- mean matched IoU: 0.5424
- mean absolute count error: 31.55
- mean latency: 1661.01 ms/image

SAM2 emitted a warning that the optional compiled `_C` extension could not be imported,
so SAM2 skipped its post-processing step. The run still completed and produced masks,
but this environment detail should be retained when interpreting the SAM2 AMG result.

Future work for SAM2 should treat the weak AMG result as an automatic-mask-generation
failure mode rather than a text-prompt issue. A small follow-up smoke test could vary
`points_per_side`, `pred_iou_thresh`, `stability_score_thresh`, and
`min_mask_region_area`, and then repeat the check after fixing the optional SAM2 `_C`
extension post-processing path. This is not required for the current PoW baseline
completion.

## Clean Subset Baseline Comparison

The comparison analysis combines completed clean subset outputs without running any
additional model inference.

Generated outputs:

- `results/baselines/clean_subset_baseline_metrics_long.csv`
- `results/baselines/clean_subset_baseline_summary.csv`
- `results/baselines/clean_subset_baseline_failure_cases.csv`
- `figures/baseline_clean_subset_metric_comparison.png`
- `figures/baseline_clean_subset_count_error_comparison.png`
- `figures/baseline_clean_subset_latency_comparison.png`
- `figures/baseline_clean_subset_score_distributions.png`
- `figures/baseline_clean_subset_precision_recall.png`
- `figures/baseline_clean_subset_image_method_f1_heatmap.png`

Current comparison summary:

| Method | Images | Mean object F1 | Mean matched IoU | Mean absolute count error | Median latency ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cellpose-SAM | 20 | 0.8892 | 0.8513 | 7.15 | 687.13 |
| Otsu + watershed | 20 | 0.4685 | 0.7307 | 63.75 | 15.72 |
| SAM2 AMG | 20 | 0.3604 | 0.5424 | 31.55 | 1600.74 |

This comparison is a smoke-test analysis product. It shows that baseline outputs can
be aggregated and visualized under the project output contract before expanding to
more perturbations or larger evaluation sets.

The richer comparison artifacts are more useful than mean-only plots for the current
PoW: the failure-case table records each method's worst clean-subset images, the
distribution plot shows per-image variance, the precision-recall plot separates
missed-object and false-positive behavior, and the heatmap highlights image-specific
method failures.

## Failure-Case Taxonomy

The first qualitative failure taxonomy is documented in
`docs/failure_taxonomy.md`. It currently covers missed objects, spurious objects,
over-segmentation, under-segmentation, boundary mismatch, and background structure
capture. The taxonomy is based on existing baseline metrics and overlay figures, and
it is intended to guide later failure-example annotation rather than serve as a final
paper taxonomy.

## Otsu Robustness Smoke Test

The first perturbation smoke test runs Otsu + watershed on a deterministic 5-image
subset with five image conditions:

- clean;
- Gaussian noise, sigma 0.08;
- Gaussian blur, sigma 1.5;
- downsample then upsample, scale 0.5;
- contrast inversion.

Generated outputs:

- `results/robustness/otsu_watershed_perturbation_smoke_metrics.csv`
- `results/robustness/otsu_watershed_perturbation_smoke_summary.csv`
- `figures/robustness_otsu_smoke_mean_scores.png`
- `figures/robustness_otsu_smoke_relative_f1_drop.png`
- `figures/robustness_otsu_smoke_overlay_examples.png`

Current smoke-test summary:

| Perturbation | Images | Mean object F1 | Mean matched IoU | Relative F1 drop |
| --- | ---: | ---: | ---: | ---: |
| clean | 5 | 0.6311 | 0.7575 | 0.0000 |
| gaussian_noise | 5 | 0.4156 | 0.7033 | 0.3415 |
| gaussian_blur | 5 | 0.6257 | 0.7808 | 0.0086 |
| downsample_upsample | 5 | 0.6201 | 0.7735 | 0.0174 |
| contrast_inversion | 5 | 0.6282 | 0.7631 | 0.0046 |

This is not a full robustness sweep. It proves that perturbations, metric aggregation,
relative-drop reporting, and robustness figures work on the existing experiment
skeleton.

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
