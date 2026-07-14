# cellseg-robustness-diagnostic

![Project Status](https://img.shields.io/badge/status-PoW%20results-green)
![Task](https://img.shields.io/badge/task-cell%20segmentation-blue)
![Protocol](https://img.shields.io/badge/protocol-robustness%20diagnostic-purple)
![Scope](https://img.shields.io/badge/scope-PoW%20mini--benchmark-green)

A reproducible proof-of-work benchmark for microscopy cell instance segmentation.
The main track evaluates zero-shot and out-of-the-box robustness on DSB2018; a
separate supervised track evaluates YOLO-seg fine-tuning on the same repository
metrics.

## Contents

- [Key Results](#key-results)
- [Main Figures](#main-figures)
- [Benchmark Design](#benchmark-design)
- [Methods](#methods)
- [Metrics](#metrics)
- [Repository Layout](#repository-layout)
- [Reproduce](#reproduce)
- [Documentation](#documentation)
- [Limitations and Follow-up Work](#limitations-and-follow-up-work)

## Key Results

**Cellpose-SAM / `cpsam` is the strongest current zero-shot baseline in this
repository.** It keeps high object-level F1 across the tested full-train
perturbations. Otsu + watershed remains useful as an interpretable classical lower
bound. SAM2 automatic mask generation is not reliable enough under the current
no-prompt AMG protocol to justify full-train expansion.

Full `stage1_train` zero-shot robustness:

| Method | Clean F1 | Gaussian noise | Poisson noise | Blur | Downsample | Intensity scale | Inversion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cellpose-SAM | 0.9178 | 0.8740 | 0.8806 | 0.8898 | 0.9006 | 0.9155 | 0.9139 |
| Otsu + watershed | 0.5736 | 0.4298 | 0.4606 | 0.5818 | 0.5825 | 0.5744 | 0.5653 |

Same 134-image held-out validation split, including the supervised YOLO capacity
probe:

| Method | Protocol | Train images | Mean object F1 | Precision | Recall | Abs. count error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cellpose-SAM | zero-shot | 0 | 0.9200 | 0.9456 | 0.9007 | 2.9328 |
| YOLO11m full train pool | supervised | 536 | 0.8680 | 0.8525 | 0.8921 | 4.8582 |
| YOLO11n full train pool | supervised | 536 | 0.8649 | 0.8440 | 0.8942 | 4.2090 |
| Otsu + watershed | zero-shot | 0 | 0.6442 | 0.6103 | 0.7219 | 19.8806 |

Interpretation:

- Cellpose-SAM is the main zero-shot baseline to carry forward.
- Otsu + watershed provides a transparent lower bound and exposes noise-driven
  count inflation.
- SAM2 AMG mainly fails through automatic-mask-generation behavior on dense
  microscopy images; this is not a language-prompt failure.
- YOLO-seg supervised fine-tuning improves far beyond Otsu, but the completed
  YOLO11m full-train-pool probe does not close the gap to Cellpose-SAM.

## Main Figures

### Protocol A/B Held-out Validation

![Protocol A/B held-out validation comparison](figures/protocol_ab_heldout_val_comparison.png)

*Figure 1. Held-out validation comparison between zero-shot baselines and supervised YOLO capacity probes.*

### Full-train Robustness

![Full-train robustness summary](figures/robustness_pow_full_train_summary.png)

*Figure 2. Full-train robustness summary for Cellpose-SAM and Otsu + watershed across tested perturbations.*

### Failure Diagnostics

![Full-train failure diagnostics](figures/robustness_pow_full_train_failure_diagnostics.png)

*Figure 3. Failure diagnostics for the full-train robustness run.*

### Clean-subset Baseline Behavior

![Clean-subset precision recall](figures/baseline_clean_subset_precision_recall.png)

*Figure 4. Clean-subset precision-recall behavior across baseline methods.*

![Clean-subset count agreement](figures/baseline_clean_subset_count_agreement.png)

*Figure 5. Clean-subset count agreement between true and predicted instance counts.*

## Benchmark Design

| Protocol | Question | Methods | Status |
| --- | --- | --- | --- |
| A. Zero-shot / out-of-the-box robustness | Which methods produce usable masks without target-domain labels or manual prompts? | Otsu + watershed; Cellpose-SAM; SAM2 AMG | Main PoW complete |
| B. Supervised adaptation | How much does target-domain supervised training help? | YOLO-seg fine-tuning | Diagnostic complete through YOLO11m |
| C. Exploratory VLM output validity | Can a mask-output VLM produce parseable and useful masks? | Gemini-style JSON mask prompting | Future optional protocol |

The project keeps zero-shot, supervised, and VLM-style segmentation separate. Their
assumptions differ, so the README reports them as related protocols rather than a
single undifferentiated ranking.

## Methods

### Protocol A

| Method | Role | Main assumption |
| --- | --- | --- |
| Otsu + watershed | Classical lower bound | No training, fixed image-processing pipeline |
| Cellpose-SAM / `cpsam` | Bio-adapted foundation baseline | Current Cellpose 4.x Cellpose-SAM workflow |
| SAM2 AMG | General segmentation foundation-model screen | Automatic grid-point prompting, no manual prompt |

Legacy Cellpose3 `cyto3` and one-click restoration are kept as optional
cross-version work. The current `cell` environment uses `cellpose==4.1.1`, where
the Cellpose-family baseline is `cpsam`.

### Protocol B

YOLO-seg is trained as a supervised real-time segmentation baseline. The current
diagnostics include:

- label-conversion smoke test;
- tiny training and evaluation smoke tests;
- fixed-budget 100-image baseline;
- threshold diagnostic;
- nested label-budget diagnostic at 100, 250, and 536 train-pool images;
- YOLO11m full-train-pool capacity probe.

### Perturbations

| Perturbation | Purpose |
| --- | --- |
| Gaussian noise | Sensor or acquisition noise |
| Poisson noise | Shot-noise-like intensity noise |
| Gaussian blur | Defocus or optical blur |
| Downsample then upsample | Undersampling stress |
| Intensity scaling | Underexposure or overexposure |
| Contrast inversion | Intensity-convention change |

Channel swap and object-scale perturbations are documented as future candidates, not
as completed evidence.

## Metrics

The main comparison uses repository-native instance metrics rather than each model's
native training logs.

| Metric | Purpose |
| --- | --- |
| Object-level F1 | Main instance detection metric |
| Precision / recall | False-positive and missed-object behavior |
| Mean matched IoU / Dice | Mask overlap quality for matched instances |
| Absolute count error | Cell-count reliability |
| Missed-object rate | False-negative tendency |
| FP per true instance | Spurious-instance burden |
| Count bias | Over-counting or under-counting direction |
| Latency | Practical runtime cost |

Over-segmentation and under-segmentation are summarized through failure-case hints
and count-bias diagnostics. A stricter split/merge graph metric is future work.

## Repository Layout

| Path | Contents |
| --- | --- |
| [src/](src/) | Shared loading, evaluation, perturbation, plotting, and visualization code |
| [scripts/](scripts/) | Reproducible experiment, evaluation, analysis, and redraw entrypoints |
| [results/dataset/](results/dataset/) | Dataset audit outputs |
| [results/baselines/](results/baselines/) | Clean-subset baseline metrics and comparisons |
| [results/robustness/](results/robustness/) | Robustness metrics, summaries, image deltas, and failure cases |
| [results/supervised/](results/supervised/) | YOLO label conversion, training metadata, evaluation, and comparisons |
| [figures/](figures/) | Flat PNG figure outputs |
| [docs/](docs/) | Protocol, environment, data, output, and findings documentation |
| [model_assets/](model_assets/) | Local model weights, ignored by git |
| [data/](data/) | Local DSB2018 data, ignored by git |

## Reproduce

Environment setup is documented in [docs/environment.md](docs/environment.md);
dataset source and local layout are documented in [docs/data.md](docs/data.md). The
environment is a conda environment named `cell` with Cellpose-SAM, SAM2,
Ultralytics YOLO, PyTorch, and the repository image-analysis stack installed.

Representative entrypoints:

```bash
python scripts/audit_dataset.py
python scripts/compare_baseline_subset.py
python scripts/run_pow_robustness_smoke.py --output-tag full_train --methods otsu_watershed cellpose_cpsam
python scripts/analyze_pow_robustness_full_train.py
python scripts/summarize_yolo_capacity_diagnostic.py
python scripts/redraw_publication_figures.py
```

The exact environment, data placement, model weights, and long-running experiment
notes are in [docs/](docs/) rather than repeated in the README.

## Documentation

| Document | Purpose |
| --- | --- |
| [technical_memo.md](technical_memo.md) | Current result memo and interpretation |
| [docs/pow_report.md](docs/pow_report.md) | Zero-shot PoW stage report |
| [docs/pow_findings.md](docs/pow_findings.md) | Method ranking, robustness, and failure-mode summary |
| [docs/supervised_protocol.md](docs/supervised_protocol.md) | YOLO supervised adaptation protocol and results |
| [docs/failure_taxonomy.md](docs/failure_taxonomy.md) | Failure-case taxonomy |
| [docs/output_contract.md](docs/output_contract.md) | Expected result and figure organization |
| [docs/experiment_plan.md](docs/experiment_plan.md) | Historical protocol plan and execution record |
| [docs/environment.md](docs/environment.md) | Environment setup |
| [docs/data.md](docs/data.md) | Dataset source and local structure |

## Limitations and Follow-up Work

- The main zero-shot evidence is based on DSB2018 stage 1 train images and a compact
  perturbation set.
- SAM2 is tested in automatic mask generation mode only; prompted SAM2 and
  post-processing repair are separate future protocols.
- Legacy Cellpose3 `cyto3` and restoration workflows are optional cross-version
  baselines, not part of the current mainline.
- The completed YOLO diagnostics do not close the gap to Cellpose-SAM; further YOLO
  work should be framed as post-processing or architecture analysis.
- VLM mask-output validity remains an exploratory future protocol.

## Disclaimer

This repository is a research proof-of-work project. It is not intended for
clinical use, biological decision-making, or production deployment.
