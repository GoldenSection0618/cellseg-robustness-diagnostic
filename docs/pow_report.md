# PoW Stage Report

This report closes the current zero-shot proof-of-work stage for DSB2018 instance
segmentation robustness. It summarizes what was tested, what the evidence supports,
and what should happen next.

Status note: this is a zero-shot stage report. Its recommendation to run a separate
supervised adaptation protocol has since been followed by Protocol B YOLO results,
including fixed-budget, label-budget, threshold, and capacity diagnostics. Those
results are documented in `docs/supervised_protocol.md` and are not part of the
zero-shot ranking summarized here.

## Scope

The current PoW stage covers zero-shot or out-of-the-box methods only:

- Otsu + watershed as the classical lower-bound reference;
- Cellpose-SAM / `cpsam` as the current bio-adapted foundation-model baseline;
- SAM2 automatic mask generation as a general segmentation foundation-model screen.

The PoW stage does not include supervised fine-tuning, VLM mask generation, prompted
SAM2, legacy Cellpose3 `cyto3`, or Cellpose3 restoration.

## Evidence

Primary result files:

- `results/baselines/clean_subset_baseline_summary.csv`
- `results/robustness/pow_baseline_robustness_clean20_summary.csv`
- `results/robustness/pow_baseline_robustness_full_train_summary.csv`
- `results/robustness/pow_baseline_robustness_full_train_failure_cases.csv`
- `results/robustness/pow_baseline_robustness_full_train_no_prediction_cases.csv`
- `results/robustness/sam2_amg_sensitivity_clean20_clean_screen_summary.csv`
- `results/robustness/sam2_amg_sensitivity_clean20_validation_summary.csv`
- `results/robustness/sam2_amg_sensitivity_clean20_failure_cases.csv`

Primary figures:

- `figures/baseline_clean_subset_metric_comparison.png`
- `figures/robustness_pow_clean20_summary.png`
- `figures/robustness_pow_clean20_failure_diagnostics.png`
- `figures/robustness_pow_full_train_summary.png`
- `figures/robustness_pow_full_train_failure_diagnostics.png`
- `figures/robustness_sam2_amg_sensitivity_clean20_mean_f1.png`

## Method Conclusions

Cellpose-SAM is the strongest current zero-shot baseline. On the full `stage1_train`
robustness run, its mean object F1 is 0.9178 on clean images and remains between
0.8740 and 0.9155 across the tested perturbations.

Otsu + watershed remains useful as a classical lower bound. It is interpretable and
fast, but its full-train clean F1 is 0.5736 and Gaussian noise drops it to 0.4298.
Its main value is diagnostic contrast, not competitive segmentation quality.

SAM2 AMG is not a good current mainline robustness baseline. The clean20 robustness
run showed collapse under perturbations, and the later parameter-sensitivity run did
not repair that pattern. The issue is not primarily empty output: the sensitivity
validation recorded zero no-prediction rows across 600 image-condition rows, but the
returned AMG masks match cell instances poorly under blur, downsample, and Gaussian
noise.

## Robustness Summary

Full-train Otsu + watershed and Cellpose-SAM:

| Method | Clean F1 | Gaussian F1 | Poisson F1 | Blur F1 | Downsample F1 | Intensity F1 | Inversion F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cellpose-SAM | 0.9178 | 0.8740 | 0.8806 | 0.8898 | 0.9006 | 0.9155 | 0.9139 |
| Otsu + watershed | 0.5736 | 0.4298 | 0.4606 | 0.5818 | 0.5825 | 0.5744 | 0.5653 |

SAM2 AMG clean20 sensitivity validation:

| Config | Clean F1 | Noise F1 | Blur F1 | Downsample F1 | Inversion F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stability_score_thresh_0.95` | 0.4190 | 0.1252 | 0.0286 | 0.0313 | 0.5676 |
| `points_per_side_32` | 0.3894 | 0.1700 | 0.0145 | 0.0272 | 0.5021 |
| `default_current` | 0.3683 | 0.1799 | 0.0213 | 0.0310 | 0.4996 |

These results support keeping Cellpose-SAM as the main zero-shot baseline, keeping
Otsu + watershed as the classical lower bound, and stopping SAM2 AMG full-train
expansion under the current AMG protocol.

## Failure Interpretation

Otsu + watershed mainly fails through false positives, coarse over-segmentation
hints, and count inflation under Gaussian and Poisson noise.

Cellpose-SAM mainly has residual missed objects and a small number of no-prediction
cases. The full-train diagnostics record 14 no-prediction image-condition rows out
of 4690 Cellpose-SAM rows.

SAM2 AMG mainly fails through poor automatic mask selection and poor cell-instance
alignment under perturbation. It is not a text-prompt failure because the current
SAM2 baseline uses automatic grid prompts, not language prompts.

## Decision

The current zero-shot PoW stage is sufficient for the main diagnostic question:
Cellpose-SAM is the practical zero-shot baseline to carry forward, Otsu + watershed
is the lower-bound reference, and SAM2 AMG should not be scaled to full_train without
a protocol change.

The next experimental protocol should be separate from this PoW stage. In the
subsequent work, that became Protocol B supervised adaptation with YOLO-seg
fine-tuning and capacity diagnostics, asking a new question: how much target-domain
annotation and model capacity improve performance over the zero-shot baselines.
Further SAM2 work should be prompted SAM2 or post-processing repair, not more
current-AMG scaling.
