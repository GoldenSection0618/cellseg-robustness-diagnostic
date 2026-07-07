# Technical Memo

## Current Objective

Build a compact proof-of-work benchmark for microscopy nucleus instance segmentation
robustness. The current implementation is an end-to-end PoW pipeline that can:

1. read the local DSB2018 data;
2. produce tabular analysis outputs in categorized `results/` subdirectories;
3. produce visual analysis outputs in `figures/`;
4. support staged robustness evaluation through a common evaluation pipeline.

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

This milestone established the output contract used by later baseline and robustness
scripts.

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

The Cellpose-family PoW baseline uses Cellpose 4.1.1 with the `cpsam` pretrained
model on the same deterministic 20-image clean subset. The current main
configuration uses `gray_mean` input and `diameter=15`.

Generated outputs:

- `results/baselines/cellpose_cpsam_clean_subset_metrics.csv`
- `figures/cellpose_cpsam_subset_overlay_examples.png`
- `figures/cellpose_cpsam_subset_metric_means.png`
- `figures/cellpose_cpsam_subset_count_scatter.png`

Current subset-level summary:

- images: 20
- mean object F1: 0.9052
- mean matched IoU: 0.8561
- mean absolute count error: 5.50
- mean latency: 3207.20 ms/image

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
| Cellpose-SAM | 20 | 0.9052 | 0.8561 | 5.50 | 1853.75 |
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

## PoW Baseline Robustness Smoke Test

The first cross-method robustness smoke test runs all three completed PoW baselines
on the same deterministic 5-image subset and five image conditions:

- Otsu + watershed;
- Cellpose-SAM / `cpsam`;
- SAM2 AMG.

Generated outputs:

- `results/robustness/pow_baseline_robustness_smoke_metrics.csv`
- `results/robustness/pow_baseline_robustness_smoke_summary.csv`
- `figures/robustness_pow_smoke_mean_f1.png`
- `figures/robustness_pow_smoke_relative_f1_drop.png`
- `figures/robustness_pow_smoke_method_condition_heatmap.png`
- `figures/robustness_pow_smoke_overlay_examples.png`

Current smoke-test summary:

| Method | Clean F1 | Gaussian noise F1 | Blur F1 | Downsample F1 | Inversion F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Otsu + watershed | 0.6311 | 0.4156 | 0.6257 | 0.6201 | 0.6282 |
| Cellpose-SAM | 0.9461 | 0.9092 | 0.9448 | 0.9448 | 0.9550 |
| SAM2 AMG | 0.3675 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

This is still a smoke test rather than a full robustness experiment. It gives a useful
directional signal: Cellpose-SAM is stable under this small perturbation set, Otsu is
mainly affected by Gaussian noise, and SAM2 AMG collapses under all tested
perturbations in this protocol. The SAM2 result should be interpreted with the known
optional `_C` post-processing warning.

## PoW Clean20 Robustness Extension

The robustness protocol has been extended to the deterministic 20-image clean subset
used by the completed clean baselines. This keeps the scope within the current PoW
mainline while covering clean-subset hard cases that the 5-image smoke test skipped.

Generated outputs:

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

Current clean20 robustness summary:

| Method | Clean F1 | Gaussian noise F1 | Blur F1 | Downsample F1 | Inversion F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Otsu + watershed | 0.4685 | 0.3676 | 0.4730 | 0.4892 | 0.4678 |
| Cellpose-SAM | 0.8892 | 0.8578 | 0.8672 | 0.8802 | 0.8960 |
| SAM2 AMG | 0.3604 | 0.0043 | 0.0020 | 0.0016 | 0.0000 |

The clean rows exactly match the existing 20-image clean baseline means, confirming
that the extension uses the same image subset and method implementations. The
20-image results preserve the smoke-test direction: Cellpose-SAM remains the most
stable baseline, Otsu + watershed is mainly degraded by Gaussian noise, and SAM2 AMG
collapses under all tested perturbations. The SAM2 result should still be interpreted
with the known optional `_C` post-processing warning.

The optimized diagnostic outputs add the missing per-image interpretation layer:
`image_deltas.csv` records each image-method-condition drop from clean, while
`failure_cases.csv` keeps the five largest drops for every method and perturbation.
The per-method worst-drop figure avoids the global ranking being dominated only by
SAM2 collapse cases.

## Staged Full-Train Robustness

The full-train robustness run has started in staged form over all 670 `stage1_train`
images and the same five image conditions. The completed full-train methods are
Otsu + watershed and Cellpose-SAM / `cpsam`.

Generated outputs:

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

Current full-train summary:

| Method | Clean F1 | Gaussian noise F1 | Blur F1 | Downsample F1 | Inversion F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Otsu + watershed | 0.5736 | 0.4298 | 0.5818 | 0.5825 | 0.5653 |
| Cellpose-SAM | 0.9178 | 0.8740 | 0.8898 | 0.9006 | 0.9139 |

This confirms the clean20 trend at full-train scale. Gaussian noise is the main Otsu
failure condition, with a 25.1% relative object-F1 drop from clean. Blur and
downsample are slightly higher than clean on average, which is consistent with mild
smoothing reducing some oversegmentation. Cellpose-SAM remains substantially stronger
and more stable, with relative object-F1 drops of 4.8% for Gaussian noise, 3.1% for
blur, 1.9% for downsample, and 0.4% for inversion.

The full-train interpretation layer intentionally uses a small metric set: per-image
object-F1 drop from clean, precision drop, recall drop, absolute count-error delta,
and no-prediction rows. This keeps the analysis focused on robustness failure
direction rather than duplicating every segmentation metric. Cellpose-SAM produced
11 no-prediction rows out of 3350 image-condition rows; these are recorded in
`pow_baseline_robustness_full_train_no_prediction_cases.csv` and are counted as
ordinary failures in the aggregate metrics.

SAM2 AMG full-train robustness is deferred. Clean20 already shows near-total SAM2 AMG
collapse under the tested perturbations, with relative object-F1 drops of 98.8% to
100.0%. Running the same current AMG settings across all 670 images is therefore
lower information gain than first testing SAM2 parameter sensitivity or repairing
the optional post-processing path.

## SAM2 AMG Parameter Sensitivity

The SAM2 AMG parameter-sensitivity experiment has now been run on the deterministic
clean20 subset. It remains a zero-shot AMG experiment: no manual prompts and no
ground-truth-derived prompts are used.

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

The clean-only screen evaluated 15 runnable configurations. `points_per_side_64`
was recorded as a CUDA out-of-memory configuration on this machine and was excluded
from the clean-screen summary. The best clean-screen setting was
`stability_score_thresh_0.95`, with mean object F1 0.4190 versus 0.3683 for the
current default. `points_per_side_32` reached 0.3894, and `crop_n_layers_1` reached
0.3827 but was much slower.

The validation stage carried the top five clean-screen configurations plus the
current default across the same five clean20 conditions. Validation produced 600
metric rows and 30 summary rows. It recorded zero no-prediction rows, so the main
failure is not that AMG returns no masks; it returns masks that do not match the
cell instances well under several perturbations.

Validation summary:

| Config | Clean F1 | Noise F1 | Blur F1 | Downsample F1 | Inversion F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stability_score_thresh_0.95` | 0.4190 | 0.1252 | 0.0286 | 0.0313 | 0.5676 |
| `points_per_side_32` | 0.3894 | 0.1700 | 0.0145 | 0.0272 | 0.5021 |
| `crop_n_layers_1` | 0.3827 | 0.1489 | 0.0334 | 0.0425 | 0.5176 |
| `box_nms_thresh_0.5` | 0.3806 | 0.1800 | 0.0214 | 0.0618 | 0.5026 |
| `points_per_side_16` | 0.3688 | 0.1867 | 0.0122 | 0.0377 | 0.4690 |
| `default_current` | 0.3683 | 0.1799 | 0.0213 | 0.0310 | 0.4996 |

This sensitivity result does not change the SAM2 full-train decision. The best clean
setting improves clean F1 modestly, and contrast inversion often scores higher than
clean, but Gaussian noise remains weak and blur/downsample remain near collapse.
The next SAM2 work should therefore be a different protocol, such as prompted SAM2,
post-processing repair, or a separate checkpoint/runtime investigation, not
full_train execution of the current AMG family.

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
report. The current robustness conclusions apply to the completed PoW scope:
clean-subset baselines, clean20 robustness, and staged Otsu/Cellpose-SAM full-train
robustness, plus SAM2 AMG clean20 parameter sensitivity. Fixed-budget supervised
YOLO is completed as separate Protocol B evidence, not as part of the zero-shot
ranking. VLM output validity, Cellpose3 cross-version baselines, prompted SAM2, and
SAM2 post-processing repair remain separate future protocols.

## Fixed-Budget YOLO Supervised Baseline

Protocol B now includes a fixed-budget YOLO11n-seg supervised run using 100 labeled
training images and a 134-image held-out validation split.

Generated outputs:

- `results/supervised/yolo_fixed_budget_train_metadata.csv`
- `results/supervised/yolo_fixed_budget_train_summary.csv`
- `results/supervised/yolo_fixed_budget_metrics.csv`
- `results/supervised/yolo_fixed_budget_eval_summary.csv`
- `results/supervised/yolo_fixed_budget_val_comparison_metrics.csv`
- `results/supervised/yolo_fixed_budget_val_comparison_summary.csv`
- `figures/supervised_yolo_fixed_budget_eval_overlays.png`

The run used `yolo11n-seg.pt`, 50 epochs, `imgsz=512`, `batch=8`, `workers=2`, AMP
disabled, and `conf=0.25` for repository-metric evaluation. Training took 351.421
seconds on the local RTX 4060 Laptop GPU.

The fixed-budget run is retained as the 100-image point in the YOLO label-budget
diagnostic. The main cross-method comparison below reports the strongest completed
YOLO supervised result by mean object F1; the full YOLO diagnostic curves are
documented in `docs/supervised_protocol.md`.

The YOLO result should not be treated as a failed experiment to be replaced. It is a
valid Protocol B v1 baseline. Any further YOLO work should be framed as a diagnostic
extension with predeclared axes: operating point, label budget, model capacity, and
possibly post-processing. The goal is to identify the cause of the gap to
Cellpose-SAM, not to tune until a preferred ranking appears.

The first YOLO follow-up diagnostic evaluated the frozen v1 checkpoint over
confidence thresholds `0.05`, `0.10`, `0.25`, `0.40`, and `0.60` on the same 134
held-out validation images. The best mean object F1 is 0.8695 at `conf=0.40` and
remains below Cellpose-SAM's 0.9200 on the same image ids.

The label-budget diagnostic split/label conversion is prepared as a nested extension
of the existing Protocol B v1 split. The 100-image v1 result remains the first budget
point. `budget_250` contains the original 100 training images plus 150 additional
train-pool images, and `full_train_pool` contains all 536 train-pool images. Both
reuse the same 134 held-out validation images. Conversion produced 11533 train
instances for `budget_250`, 23862 train instances for `full_train_pool`, 5599 shared
validation instances, and 0 dropped instances.

The label-budget diagnostic has now been trained for 100 images, `budget_250`, and
`full_train_pool`, all evaluated on the same 134 held-out validation images.

| Method | Protocol | Mean object F1 | Mean precision | Mean recall | Mean absolute count error |
| --- | --- | ---: | ---: | ---: | ---: |
| Cellpose-SAM | zero-shot | 0.9200 | 0.9456 | 0.9007 | 2.9328 |
| YOLO supervised | supervised | 0.8680 | 0.8525 | 0.8921 | 4.8582 |
| Otsu + watershed | zero-shot | 0.6442 | 0.6103 | 0.7219 | 19.8806 |

YOLO supervised improves over Otsu but remains below Cellpose-SAM.

The YOLO11m capacity probe was then run on the same full train-pool split. It reaches
mean object F1 0.8680 and mean absolute count error 4.8582 on the same held-out
validation ids, compared with YOLO11n full train-pool F1 0.8649 and count error
4.2090. The larger model does not materially close the gap to Cellpose-SAM, so the
current evidence does not justify adding an intermediate YOLO11s run to the main PoW
record.
