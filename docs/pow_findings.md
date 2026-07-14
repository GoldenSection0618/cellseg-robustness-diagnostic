# Zero-shot Findings

## Headline Results

- **Cellpose-SAM / `cpsam` is the strongest completed zero-shot baseline.** Its
  full-train mean object F1 is 0.9178 on clean images and remains 0.8740 or higher
  across all tested perturbations.
- **Otsu + watershed is the classical lower bound.** It is fast and interpretable,
  but noise causes false-positive growth and count inflation; its clean F1 is 0.5736.
- **SAM2 AMG is unsuitable under the tested no-manual-prompt configuration.** Its
  masks are poorly matched to cell instances under key perturbations even when masks
  are returned.

The complete numeric record is in the [zero-shot results report](pow_report.md).
The reader-facing comparison, including supervised YOLO-seg, is in the
[README](../README.md).

## What the Evidence Supports

The full-train evaluation supports Cellpose-SAM as the preferred zero-shot method for
this DSB2018 benchmark and perturbation suite. The relative F1 loss is 4.8% under
Gaussian noise, 4.1% under Poisson noise, 3.1% under blur, 1.9% under downsampling,
0.3% under intensity scaling, and 0.4% under inversion.

Otsu + watershed is most vulnerable to Gaussian and Poisson noise, where F1 falls
by 25.1% and 19.7%, respectively. Its main diagnostic value is making false-positive
and count-bias failure patterns visible.

The SAM2 conclusion is narrower: the current AMG pipeline uses automatic grid
prompts and a 20-image sensitivity evaluation. Its limitation is mask quality and
instance alignment, not a text-prompt failure or consistently empty output.

## What the Evidence Does Not Support

- It does not show that prompted SAM2, alternate SAM2 checkpoints, or repaired SAM2
  post-processing cannot work on this dataset.
- It does not establish robustness beyond the six tested perturbations or beyond
  DSB2018 `stage1_train`.
- It does not rank supervised YOLO-seg as a zero-shot method. The held-out supervised
  comparison is a separate protocol described in
  [supervised_protocol.md](supervised_protocol.md).
- It does not include legacy Cellpose3 or restoration as missing comparisons; they
  require a separate cross-version setup.

## Evidence Links

- [Full-train aggregate metrics](../results/robustness/pow_baseline_robustness_full_train_summary.csv)
- [Full-train failure cases](../results/robustness/pow_baseline_robustness_full_train_failure_cases.csv)
- [SAM2 AMG sensitivity validation](../results/robustness/sam2_amg_sensitivity_clean20_validation_summary.csv)
- [Failure taxonomy](failure_taxonomy.md)
