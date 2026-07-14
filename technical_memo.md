# Technical Memo

## Purpose

This memo explains the evidence behind the repository's reported segmentation
results. It is a companion to the reader-facing [README](README.md), not an
execution log. Exact zero-shot result tables are maintained in the
[zero-shot results report](docs/pow_report.md); the concise conclusions are in
[findings](docs/pow_findings.md).

## Dataset and Evaluation

The benchmark uses the Kaggle [2018 Data Science Bowl](docs/data.md) data under
`data/raw/dsb2018/`. The zero-shot robustness results use all 670 images in
`stage1_train`, which provides per-instance masks. The supervised comparison uses a
fixed 536/134 train/held-out-validation split from the same image pool.

All methods are evaluated with the repository's shared instance-matching metrics:
object-level F1, precision, recall, matched-mask overlap, and absolute count error.
This avoids comparing model-native training metrics with a different definition of
an instance. The metric definitions and output layout are specified in the
[output contract](docs/output_contract.md).

## Method Scope

The completed zero-shot comparison contains three distinct method roles:

| Method | Role | Evidence scale |
| --- | --- | --- |
| Otsu + watershed | Interpretable classical lower bound | Full 670-image robustness run |
| Cellpose-SAM / `cpsam` | Bio-adapted zero-shot baseline | Full 670-image robustness run |
| SAM2 AMG | General segmentation-model screen | 20-image robustness and sensitivity runs |

The current Cellpose environment is `cellpose==4.1.1`; its reported baseline is
`cpsam` with grayscale-mean input and `diameter=15`. Legacy Cellpose3 `cyto3` and
restoration are cross-version comparisons, not results in this benchmark. The
environment setup is documented in [docs/environment.md](docs/environment.md).

SAM2 is evaluated with automatic mask generation (AMG), using grid prompts generated
by the method itself. It is therefore not a language-prompt experiment. The optional
SAM2 compiled post-processing extension was unavailable during these runs; SAM2 still
returned masks, but its AMG results should be read within that runtime condition.

## What the Results Show

Cellpose-SAM is the strongest completed zero-shot method. Across the six tested
non-clean perturbations, full-train object F1 ranges from 0.8740 to 0.9155, compared
with 0.9178 on clean images. Its principal residual behavior is a modest recall loss
under noise and 14 no-prediction image-condition rows out of 4,690 evaluated rows.

Otsu + watershed serves as a useful lower bound rather than a competitive method.
Its clean F1 is 0.5736, and Gaussian and Poisson noise increase false positives and
count error substantially. Blur, downsampling, and intensity scaling do not produce
the same degradation pattern.

SAM2 AMG is not adequate for the main robustness comparison under the tested
configuration. On the 20-image subset, AMG returns masks but aligns them poorly with
cell instances under Gaussian noise, blur, and downsampling. A six-configuration
sensitivity run improved clean F1 only modestly and did not change this conclusion.

The supervised YOLO-seg comparison asks a separate question: whether target-domain
labels close the zero-shot gap. The strongest completed result, YOLO11m trained on
536 images, reaches held-out F1 0.8680, below Cellpose-SAM's 0.9200 on the same 134
validation images. Full protocol details are in
[docs/supervised_protocol.md](docs/supervised_protocol.md).

## Evidence and Interpretation

The primary full-train evidence is the
[summary table](results/robustness/pow_baseline_robustness_full_train_summary.csv),
[per-image clean-to-perturbation deltas](results/robustness/pow_baseline_robustness_full_train_image_deltas.csv),
and [failure cases](results/robustness/pow_baseline_robustness_full_train_failure_cases.csv).
The 14 Cellpose-SAM no-prediction cases are listed separately in
[the no-prediction table](results/robustness/pow_baseline_robustness_full_train_no_prediction_cases.csv).

The central figures are the [full-train robustness summary](figures/robustness_pow_full_train_summary.png),
[failure diagnostics](figures/robustness_pow_full_train_failure_diagnostics.png),
[precision-recall view](figures/baseline_clean_subset_precision_recall.png), and
[count-agreement view](figures/baseline_clean_subset_count_agreement.png).

Failure descriptions use the vocabulary in
[docs/failure_taxonomy.md](docs/failure_taxonomy.md). They are diagnostic summaries
of observed metrics and overlays, not ground-truth annotations of every error type.

## Boundaries of the Evidence

- The zero-shot robustness claim applies to DSB2018 `stage1_train` and the six
  tested perturbations: Gaussian noise, Poisson noise, Gaussian blur,
  downsample/upsample, intensity scaling, and contrast inversion.
- SAM2 evidence is limited to the current no-manual-prompt AMG configuration and the
  20-image sensitivity evaluation; it does not establish the performance of prompted
  SAM2 or repaired post-processing.
- The YOLO comparison is supervised and should not be interpreted as another
  zero-shot baseline.
- VLM mask-output validity, legacy Cellpose3, and restoration are separate protocols
  rather than missing rows in the present ranking.
