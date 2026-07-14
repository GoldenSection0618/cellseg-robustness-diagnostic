# Failure-Case Taxonomy

This document defines the diagnostic labels used for visible segmentation failures.
It is an evidence guide, not a complete ground-truth annotation set. The labels are
based on the clean-subset baselines, 20-image screening runs, and completed full-train
robustness diagnostics. The reader-facing method conclusions are in the
[README](../README.md) and [zero-shot results report](pow_report.md).

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
- `figures/robustness_pow_clean20_failure_diagnostics.png`
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

Current diagnostic indicators:

- `FP` and `OVER`, because the mean absolute count error is large and precision is
  lower than recall in the clean-subset summary.
- `FN` on difficult images where fixed thresholding selects the wrong foreground.
- Perturbation sensitivity to `gaussian_noise`, where the 5-image smoke test shows a
  relative object-F1 drop of 0.3415.

### Cellpose-SAM / `cpsam`

Current clean-subset summary:

- mean object F1: 0.9052
- mean matched IoU: 0.8561
- mean absolute count error: 5.50

Current diagnostic indicators:

- Residual `FN` or `FP` on difficult images rather than a systematic count failure.
- Boundary errors on crowded or irregular nuclei.
- The full-train diagnostics show that these residual failures become more visible
  under perturbation, particularly under noise.

### SAM2 AMG

Current clean-subset summary:

- mean object F1: 0.3604
- mean matched IoU: 0.5424
- mean absolute count error: 31.55

Current diagnostic indicators:

- `FN`, because recall is low in the clean-subset summary.
- `BG`, because AMG can capture borders, elongated background structures, or
  non-nuclear texture in the qualitative overlays.
- `BOUNDARY`, because SAM2 is a general segmentation model and the automatic mask
  generator is not nucleus-specific.

This is not a text-prompt failure: the current SAM2 baseline uses automatic mask
generation with grid-based point prompts and no language prompt. The 20-image
parameter-sensitivity run improves clean F1 only modestly and does not repair the
blur, downsampling, or noise pattern. Prompted SAM2 and repaired post-processing are
separate protocols outside the reported comparison.

## Annotation Guidance

When adding qualitative failure examples in a separate protocol, annotate each image
with:

- method;
- split and image id;
- perturbation;
- primary failure code;
- secondary failure code, if useful;
- short free-text note;
- figure filename that shows the failure.

Do not mix oracle prompts, manual prompts, or target-derived prompts into this
taxonomy. Record any such evaluation as a separate protocol.
