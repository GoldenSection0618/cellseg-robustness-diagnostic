# cellseg-robustness-diagnostic

![Project Status](https://img.shields.io/badge/status-PoW%20results-green)
![Task](https://img.shields.io/badge/task-cell%20segmentation-blue)
![Protocol](https://img.shields.io/badge/protocol-robustness%20diagnostic-purple)
![Scope](https://img.shields.io/badge/scope-PoW%20mini--benchmark-green)

A diagnostic mini-benchmark with a main **zero-shot robustness** protocol, plus separate supervised adaptation and exploratory VLM output-validity protocols for microscopy cell instance segmentation.

This repository now contains a compact proof-of-work implementation with completed
clean-subset baselines, robustness smoke tests, a clean20 robustness extension, and
staged full-train robustness results for Otsu + watershed and Cellpose-SAM. The goal
is not to claim a new state-of-the-art benchmark, but to build a reproducible
diagnostic project that compares representative segmentation paradigms under
controlled conditions.

## Results Summary

The main PoW finding is that **Cellpose-SAM / `cpsam` is the strongest current
zero-shot baseline** for this DSB2018 instance-segmentation diagnostic. It is more
accurate and more robust than Otsu + watershed, while SAM2 automatic mask generation
is not reliable enough under the current no-prompt AMG protocol to justify a
full-train robustness expansion.

Full `stage1_train` zero-shot robustness:

| Method | Clean F1 | Gaussian noise | Poisson noise | Blur | Downsample | Intensity scale | Inversion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cellpose-SAM | 0.9178 | 0.8740 | 0.8806 | 0.8898 | 0.9006 | 0.9155 | 0.9139 |
| Otsu + watershed | 0.5736 | 0.4298 | 0.4606 | 0.5818 | 0.5825 | 0.5744 | 0.5653 |

Same 134-image held-out validation split, including the supervised YOLO capacity
probe:

| Method | Protocol | Train images | Mean object F1 | Mean precision | Mean recall | Mean abs. count error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cellpose-SAM | zero-shot | 0 | 0.9200 | 0.9456 | 0.9007 | 2.9328 |
| YOLO11m full train pool | supervised | 536 | 0.8680 | 0.8525 | 0.8921 | 4.8582 |
| YOLO11n full train pool | supervised | 536 | 0.8649 | 0.8440 | 0.8942 | 4.2090 |
| Otsu + watershed | zero-shot | 0 | 0.6442 | 0.6103 | 0.7219 | 19.8806 |

Key interpretation:

* Cellpose-SAM remains stable across the tested full-train perturbations.
* Otsu + watershed is a useful classical lower bound, but noise causes count
  inflation and false-positive-heavy failures.
* SAM2 AMG fails under the current automatic-grid protocol on clean20 robustness and
  parameter sensitivity; this is an AMG protocol issue, not a text-prompt issue.
* YOLO-seg supervised fine-tuning improves far beyond Otsu, but the completed
  YOLO11m full-train-pool probe still does not close the gap to Cellpose-SAM.

Primary figures:

![Protocol A/B held-out validation comparison](figures/protocol_ab_heldout_val_comparison.png)

![Full-train robustness summary](figures/robustness_pow_full_train_summary.png)

![Full-train failure diagnostics](figures/robustness_pow_full_train_failure_diagnostics.png)

Detailed stage reports are in `technical_memo.md`, `docs/pow_report.md`,
`docs/pow_findings.md`, and `docs/supervised_protocol.md`.

## At a Glance

| Protocol | Question | Methods / status | Reported as |
| --- | --- | --- | --- |
| A. Zero-shot / out-of-the-box robustness | Which methods produce usable instance masks without target-domain labels or manual prompts? | Otsu + watershed; Cellpose-SAM / `cpsam`; SAM2 automatic mask generator | Main PoW diagnostic comparison |
| B. Supervised adaptation | How much performance can small-label task-specific training buy? | YOLO-seg fine-tuned; optional Cellpose fine-tuned | Separate supervised protocol |
| C. Exploratory VLM segmentation | Can a mask-output VLM produce parseable, valid, useful outputs under a fixed prompt? | Gemini 2.5 Flash segmentation | Separate exploratory protocol |

## Motivation

Cell segmentation has moved through several technical stages:

1. classical image processing pipelines, such as thresholding, morphology, and watershed;
2. specialist deep-learning cell segmentation models, such as Cellpose;
3. restoration-enhanced segmentation workflows, such as Cellpose3 image restoration;
4. bio-adapted foundation models, such as Cellpose-SAM, CellSAM, and Segment Anything for Microscopy;
5. general segmentation foundation models, such as SAM and SAM2;
6. general vision-language models that can produce structured segmentation outputs under language prompts;
7. supervised real-time segmentation models, such as YOLO-seg, when task-specific labels are available.

These model families are often discussed together, but they do not operate under the same assumptions. Some are zero-shot or out-of-the-box models. Some rely on automatic geometric prompts. Some require language prompts. Some require supervised fine-tuning on target-domain masks.

This project is designed to make those assumptions explicit.

## Project Goal

The benchmark asks:

> Under controlled microscopy image perturbations, how do classical, specialist, bio-adapted foundation, and general segmentation methods differ in zero-shot or out-of-the-box robustness, latency, and failure modes?

Two secondary questions are evaluated separately:

* how much performance task-specific supervised adaptation can buy when a small labeled training split is available;
* whether a mask-output VLM can produce parseable, valid, and useful segmentation outputs under a fixed prompt.

The emphasis is on diagnostic value rather than leaderboard-style ranking.

## Scope

This project will focus on:

* microscopy cell or nucleus instance segmentation;
* small-scale, reproducible experiments;
* open datasets with ground-truth masks;
* controlled image perturbations;
* consistent evaluation metrics;
* clear separation between zero-shot, supervised, and VLM-based protocols;
* qualitative failure-case analysis.

This project will not attempt to:

* provide clinical or biological validation;
* claim a production-ready segmentation system;
* cover every available cell segmentation model;
* tune every method to its best possible performance;
* compare supervised and zero-shot methods in a single undifferentiated ranking.

## Experimental Protocols

### Protocol A: Zero-shot / Out-of-the-box Robustness

This is the main protocol.

All methods receive the same image input. No target-domain training labels, human prompts, or ground-truth-derived prompts are used.

PoW methods:

| Method                             | Role                                  | Main Assumption                                  |
| ---------------------------------- | ------------------------------------- | ------------------------------------------------ |
| Otsu + watershed                   | Classical baseline                    | No training, fixed image-processing pipeline     |
| Cellpose-SAM                       | Bio-adapted foundation model          | Current Cellpose 4.x / Cellpose-SAM workflow     |
| SAM2 automatic mask generator      | General segmentation foundation model | Automatic grid-point prompting, no manual prompt |

Legacy Cellpose 3 `cyto3` and Cellpose 3 one-click restoration are treated as optional
future cross-version baselines, not as required methods for the current PoW. The
current `cell` environment uses `cellpose==4.1.1`, where the runnable Cellpose-family
segmentation baseline is `cpsam`.

This protocol is intended to answer:

> If no target-domain labels are available, which methods produce usable instance masks under clean and perturbed microscopy images?

### Protocol B: Supervised Adaptation

This protocol is separate from the main zero-shot comparison.

Supervised method status:

| Method              | Role                                       | Main Assumption                    |
| ------------------- | ------------------------------------------ | ---------------------------------- |
| YOLO-seg fine-tuned | Completed supervised real-time segmentation baseline | Uses the same held-out validation split |
| Cellpose fine-tuned | Optional supervised specialist baseline    | Uses the same training split masks |

This protocol is intended to answer:

> If a small number of target-domain labels are available, how much performance can task-specific supervised training buy compared with zero-shot methods?

Supervised results are reported with training-budget metadata, including:

* number of training images;
* annotation type;
* fine-tuning budget;
* training epochs;
* training time;
* GPU memory if available;
* model size;
* inference latency.

### Protocol C: Exploratory VLM Segmentation

General VLMs are not treated as direct drop-in replacements for Cellpose or SAM2 in the main benchmark. They are evaluated separately because their output quality depends on prompt wording, API behavior, output parsing, and model versioning.

Planned VLM setting:

| Method                              | Role                     | Main Assumption                                   |
| ----------------------------------- | ------------------------ | ------------------------------------------------- |
| Gemini 2.5 Flash segmentation       | Exploratory VLM baseline | Fixed JSON-mask prompt, no target-domain training |

Other mask-output VLM APIs may be considered only as optional follow-up experiments, not as part of the core proof-of-work scope.

The VLM protocol will report:

* parse success rate;
* valid mask rate;
* empty output rate;
* output format error rate;
* instance count error;
* mask quality on valid outputs;
* latency;
* estimated cost;
* prompt sensitivity.

## Robustness Perturbations

The benchmark evaluates each method on clean images and a small controlled
perturbation set.

Current PoW perturbations:

| Perturbation              | Purpose                                          |
| ------------------------- | ------------------------------------------------ |
| Gaussian or Poisson noise | Test robustness to shot noise and sensor noise   |
| Gaussian blur             | Test robustness to defocus or optical blur       |
| Downsample then upsample  | Test robustness to undersampling                 |
| Contrast inversion        | Test robustness to intensity convention changes  |
| Intensity scaling         | Test robustness to underexposure or overexposure |

Candidate future perturbations:

| Perturbation        | Purpose                                      |
| ------------------- | -------------------------------------------- |
| Channel swap        | Test robustness to channel-order assumptions |
| Object scale change | Test robustness to object-size variation     |

Most completed perturbations are chosen to stress-test documented robustness claims
around noise, blur, undersampling, contrast inversion, and exposure. Channel order
and object-scale stress tests remain future work rather than part of the current PoW
evidence.

The final perturbation set will be kept small to avoid turning this proof-of-work project into a large benchmark paper.

## Metrics

### Segmentation Metrics

| Metric                  | Purpose                               |
| ----------------------- | ------------------------------------- |
| Object-level $F_1$      | Main instance detection metric        |
| Mean IoU                | Mask overlap quality                  |
| Dice score              | Mask overlap quality                  |
| Count error             | Cell or nucleus counting reliability  |
| Missed-object rate      | False negative behavior               |
| FP per true instance    | Spurious instance burden              |
| Count bias              | Over-counting or under-counting trend |

Over-segmentation and under-segmentation are tracked qualitatively through failure
case hints and count-bias diagnostics in the current PoW. A stricter
overlap-graph split/merge rate is left as a future diagnostic instead of being
reported as a completed primary metric.

### Robustness Metrics

For a segmentation metric $S$, robustness drop will be reported as:

$$
\mathrm{absolute\_drop} = S_{\mathrm{clean}} - S_{\mathrm{perturbed}}
$$

$$
\mathrm{relative\_drop} =
\frac{S_{\mathrm{clean}} - S_{\mathrm{perturbed}}}{S_{\mathrm{clean}}}
$$

### Engineering Metrics

| Metric                      | Purpose                               |
| --------------------------- | ------------------------------------- |
| Inference latency per image | Practical runtime cost                |
| Model setup complexity      | Installation and configuration burden |
| Prompt or parameter burden  | Amount of manual setup required       |
| Output validity             | Especially important for VLM outputs  |
| Output format error rate    | Especially important for VLM outputs  |
| Estimated API cost          | Especially important for VLM outputs  |

## Prompt Policy

Prompting is a key design issue for SAM-style and VLM-style models.

The current policy is:

1. **SAM2 main result** uses automatic mask generation only. This relies on grid-based point prompts generated automatically by the algorithm.
2. **No manual point, box, or mask prompts** are used in the main zero-shot benchmark.
3. **No ground-truth-derived prompts** are used in the main benchmark.
4. Ground-truth box or point prompts may be included only as an optional oracle upper bound, and will not be mixed into the main ranking.
5. VLM segmentation uses a fixed JSON-mask prompt adapted from published Gemini segmentation guidance.
6. VLM prompt sensitivity is analyzed separately instead of selecting the best prompt post hoc.
7. APG, micro-SAM, and CellSAM-style detector-generated prompts are treated as future or optional follow-up directions, not as main-protocol requirements.

Example VLM prompt template for a nucleus dataset:

```text
Give the segmentation masks for all visible individual nuclei in this microscopy image.
Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key "box_2d", the segmentation mask in key "mask", and the text label in the key "label".
Use the label "nucleus" for each individual nucleus.
```

Example VLM prompt template for a cell dataset:

```text
Give the segmentation masks for all visible individual cells in this microscopy image.
Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key "box_2d", the segmentation mask in key "mask", and the text label in the key "label".
Use the label "cell" for each individual cell.
```

## Expected Outputs

Current PoW outputs include:

* categorized `results/<category>/*.csv` tables for metric summaries and audits;
* flat `figures/*.png` outputs for robustness, comparison, and failure-case visualization;
* per-method qualitative examples;
* a short technical memo;
* a failure-case taxonomy in `docs/failure_taxonomy.md`;
* reproducibility notes for model versions, prompts, and evaluation settings.

## Expected Diagnostic Questions

The analysis is expected to focus on questions such as:

* whether Cellpose-SAM shows smaller robustness drops under the perturbations associated with its documented robustness claims;
* whether SAM2 automatic mask generation fails mainly through missed objects, false positives, over-segmentation, or under-segmentation in dense microscopy images;
* whether simple SAM2 AMG parameter changes repair the clean20 perturbation failure pattern;
* whether optional cross-version Cellpose3 baselines are worth adding after the PoW path is stable;
* how much supervised fine-tuning improves performance relative to the zero-shot methods, given the recorded training budget;
* whether VLM segmentation failures are dominated by output validity, JSON parsing, empty outputs, count errors, or mask overlap quality;
* which perturbations produce the clearest qualitative failure modes for each method family.

## Related Work

This project is informed by several lines of work.

### Cellpose and Cellpose3

* [Cellpose GitHub repository](https://github.com/MouseLand/cellpose)
* [Cellpose3: one-click image restoration for improved cellular segmentation](https://www.nature.com/articles/s41592-025-02595-5)
* [Cellpose image restoration documentation](https://cellpose.readthedocs.io/en/latest/restore.html)

Cellpose provides generalist cellular segmentation models. Cellpose3 introduced one-click restoration workflows for denoising, deblurring, and upsampling before segmentation. In this PoW repository, legacy Cellpose3 `cyto3` and one-click restoration are future optional cross-version baselines; the current Cellpose-family mainline uses Cellpose-SAM / `cpsam` from `cellpose==4.1.1`.

### Cellpose-SAM

* [Cellpose-SAM: superhuman generalization for cellular segmentation](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1)
* [Cellpose documentation](https://cellpose.readthedocs.io/)

Cellpose-SAM adapts a SAM-style foundation model backbone to cellular segmentation. It motivates testing robustness to noise, blur, undersampling, contrast inversion, channel order, and object-size variation.

### SAM, SAM2, and Prompting

* [Segment Anything](https://segment-anything.com/)
* [Segment Anything paper](https://arxiv.org/abs/2304.02643)
* [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714)
* [SAM2 GitHub repository](https://github.com/facebookresearch/sam2)
* [SAM2 automatic mask generator example](https://github.com/facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb)

SAM-style models rely on geometric prompts such as points, boxes, or masks. The SAM2 automatic mask generator operationalizes this by sampling grid-based point prompts and filtering candidate masks. In the current PoW, SAM2 is evaluated only in automatic mask generation mode, without manual or ground-truth-derived prompts.

### Foundation Models for Cell Segmentation

* [Revisiting foundation models for cell instance segmentation](https://openreview.net/forum?id=xFO3DFZN45)
* [Segment Anything for Microscopy](https://www.nature.com/articles/s41592-024-02580-4)
* [CellSAM: a foundation model for cell segmentation](https://www.nature.com/articles/s41592-025-02879-w)

These works motivate the distinction between general-purpose segmentation foundation models and microscopy-adapted foundation models. They also highlight the importance of automatic prompt generation for cell instance segmentation.

This proof-of-work project does not plan to implement APG, micro-SAM, or CellSAM-style detector-generated prompts in the main protocol. Those approaches are better treated as future work or optional follow-up experiments because they introduce a separate prompt-generation pipeline. Ground-truth box or point prompts, if ever included, should be reported only as oracle upper bounds.

### VLM-based Segmentation

* [Conversational image segmentation with Gemini 2.5](https://developers.googleblog.com/conversational-image-segmentation-gemini-2-5/)
* [Gemini image understanding documentation](https://ai.google.dev/gemini-api/docs/image-understanding)

VLM-based segmentation is included as an exploratory protocol. The focus is not only mask accuracy, but also output validity, JSON parse success, output format errors, prompt sensitivity, latency, and cost.

### YOLO-based Instance Segmentation

* [Ultralytics YOLO segmentation documentation](https://docs.ultralytics.com/tasks/segment/)
* [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/)

YOLO-seg is tracked as a supervised real-time segmentation baseline, not as a
zero-shot method. Protocol B has completed label-conversion, tiny training, and
tiny evaluation smokes, plus fixed-budget, threshold, label-budget, and YOLO11m
capacity diagnostics evaluated on the same 134-image held-out validation split.

### Classical Segmentation Baselines

* [scikit-image watershed example](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html)
* [CellProfiler IdentifyPrimaryObjects documentation](https://cellprofiler-manual.s3.amazonaws.com/CPmanual/IdentifyPrimaryObjects.html)

Classical thresholding and watershed baselines are included to provide an interpretable lower-bound reference.

## Current Status

This repository now has a proof-of-work implementation and recorded results rather
than only a protocol design. The current implementation uses the DSB2018 local
dataset, deterministic clean subsets, and staged full-train robustness outputs.

Implemented PoW artifacts:

1. dataset audit tables in `results/dataset/` and dataset figures in `figures/`;
2. shared instance-mask metrics and visualization helpers under `src/`;
3. clean-subset baselines for Otsu + watershed, Cellpose-SAM / `cpsam`, and SAM2 AMG;
4. clean-subset comparison and failure-case tables in `results/baselines/`, with comparison figures in `figures/`;
5. small Otsu-only and three-baseline perturbation smoke tests, a 20-image clean-subset robustness extension, staged Otsu/Cellpose-SAM full-train robustness runs, and SAM2 AMG clean20 parameter-sensitivity results in `results/robustness/`;
6. YOLO supervised adaptation protocol, a 20-image label-conversion smoke test, fixed-budget YOLO split/label conversion, fixed-budget held-out validation results, threshold diagnostic results, nested label-budget diagnostic results for 100, 250, and full train-pool budgets, and a YOLO11m capacity probe in `results/supervised/`;
7. root-level `technical_memo.md` with current summaries and limitations;
8. PoW support docs under `docs/`, including data, environment, supervised protocol, output contract, experiment plan, checklist, failure taxonomy, findings, and stage report.

Post-PoW follow-ups:

1. keep the current PoW mainline focused on the completed zero-shot robustness artifacts;
2. keep SAM2 AMG full-train robustness deferred because clean20 parameter sensitivity did not repair the failure pattern;
3. treat any further SAM2 work as a different protocol, such as prompted SAM2 or post-processing repair;
4. keep legacy Cellpose3 `cyto3` and one-click restoration as optional cross-version work;
5. treat further YOLO work as optional post-processing or architecture work, since the completed YOLO11m capacity probe does not close the gap to Cellpose-SAM.

## Reproducibility Principles

The project will follow these principles:

* separate zero-shot and supervised protocols;
* keep exploratory VLM results separate from the main segmentation ranking;
* avoid test-set prompt tuning;
* avoid ground-truth-derived prompts in the main benchmark;
* record model versions and configuration files;
* keep perturbations deterministic;
* report failures instead of hiding invalid outputs;
* avoid overclaiming beyond the tested datasets.

## Disclaimer

This repository is a research and proof-of-work project. It is not intended for clinical use, biological decision-making, or production deployment.
