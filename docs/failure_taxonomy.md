# Failure-Case Taxonomy

This taxonomy is a lightweight PoW artifact. It names failure modes visible in the
current clean-subset baselines, robustness smoke tests, clean20 robustness extension,
and staged full-train robustness diagnostics, so later experiments can annotate
failures consistently.

## Evidence Used

- `results/baselines/clean_subset_baseline_summary.csv`
- `results/baselines/clean_subset_baseline_failure_cases.csv`
- `results/robustness/otsu_watershed_perturbation_smoke_summary.csv`
- `results/robustness/pow_baseline_robustness_smoke_summary.csv`
- `results/robustness/pow_baseline_robustness_clean20_summary.csv`
- `results/robustness/pow_baseline_robustness_clean20_image_deltas.csv`
- `results/robustness/pow_baseline_robustness_clean20_failure_cases.csv`
- `results/robustness/pow_baseline_robustness_full_train_image_deltas.csv`
- `results/robustness/pow_baseline_robustness_full_train_failure_cases.csv`
- `results/robustness/pow_baseline_robustness_full_train_no_prediction_cases.csv`
- `figures/otsu_watershed_subset_overlay_examples.png`
- `figures/cellpose_cpsam_subset_overlay_examples.png`
- `figures/sam2_amg_subset_overlay_examples.png`
- `figures/robustness_otsu_smoke_overlay_examples.png`
- `figures/robustness_pow_smoke_overlay_examples.png`
- `figures/robustness_pow_clean20_overlay_examples.png`
- `figures/robustness_pow_clean20_image_f1_drop_heatmap.png`
- `figures/robustness_pow_clean20_worst_f1_drops.png`
- `figures/robustness_pow_full_train_failure_diagnostics.png`

## Instance-Level Failure Modes

| Code | Failure mode | Description | Current signals |
| --- | --- | --- | --- |
| FN | Missed object | A ground-truth nucleus has no matched predicted instance at IoU 0.5. | Low recall; visible green-only objects in overlays. |
| FP | Spurious object | A predicted instance does not match any ground-truth nucleus. | Low precision; red-only regions in overlays. |
| OVER | Over-segmentation | One nucleus is split into multiple predicted instances. | Excess predicted counts, clustered red fragments around one green object. |
| UNDER | Under-segmentation | Multiple nuclei are merged into one predicted instance. | Low predicted counts, large red regions covering several green objects. |
| BOUNDARY | Boundary mismatch | Object is detected but mask boundary is systematically too large, too small, or shifted. | Matched IoU and Dice lower than object F1 would suggest. |
| BG | Background structure capture | Non-nuclear texture, image border, or elongated background structure is segmented as an object. | Red regions along borders or non-nuclear structures. |

## Current Method Notes

### Otsu + Watershed

Current clean-subset summary:

- mean object F1: 0.4685
- mean matched IoU: 0.7307
- mean absolute count error: 63.75

Primary expected failure modes:

- `FP` and `OVER`, because the mean absolute count error is large and precision is
  lower than recall in the clean-subset summary.
- `FN` on low-contrast nuclei and heterogeneous images where fixed thresholding picks
  the wrong foreground.
- Perturbation sensitivity to `gaussian_noise`, where the 5-image smoke test shows a
  relative object-F1 drop of 0.3415.

### Cellpose-SAM / `cpsam`

Current clean-subset summary:

- mean object F1: 0.9052
- mean matched IoU: 0.8561
- mean absolute count error: 5.50

Primary expected failure modes:

- Residual `FN` or `FP` on difficult images rather than a systematic count failure.
- Boundary errors on crowded or irregular nuclei.
- This is the strongest current clean-subset baseline, so later robustness analysis
  should focus on whether its failures appear under perturbation rather than clean
  input.

### SAM2 AMG

Current clean-subset summary:

- mean object F1: 0.3604
- mean matched IoU: 0.5424
- mean absolute count error: 31.55

Primary expected failure modes:

- `FN`, because recall is low in the clean-subset summary.
- `BG`, because AMG can capture borders, elongated background structures, or
  non-nuclear texture in the qualitative overlays.
- `BOUNDARY`, because SAM2 is a general segmentation model and the automatic mask
  generator is not nucleus-specific.

This is not a text-prompt failure: the current SAM2 baseline uses automatic mask
generation with grid-based point prompts and no language prompt. Parameter sensitivity
and optional SAM2 post-processing repair are future work, not current PoW blockers.

## Annotation Guidance

When adding qualitative failure examples later, annotate each image with:

- method;
- split and image id;
- perturbation;
- primary failure code;
- secondary failure code, if useful;
- short free-text note;
- figure filename that shows the failure.

Do not mix oracle prompts, manual prompts, or target-derived prompts into the main
failure taxonomy. If those are evaluated later, record them as a separate optional
protocol.
