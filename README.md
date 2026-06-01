# cellseg-robustness-diagnostic

![Project Status](https://img.shields.io/badge/status-design%20stage-lightgrey)
![Task](https://img.shields.io/badge/task-cell%20segmentation-blue)
![Protocol](https://img.shields.io/badge/protocol-robustness%20diagnostic-purple)
![Scope](https://img.shields.io/badge/scope-PoW%20mini--benchmark-green)

A diagnostic mini-benchmark with a main **zero-shot robustness** protocol, plus separate supervised adaptation and exploratory VLM output-validity protocols for microscopy cell instance segmentation.

This repository is currently in the planning and protocol-design stage. The goal is not to claim a new state-of-the-art benchmark, but to build a compact, reproducible proof-of-work project that compares several representative segmentation paradigms under controlled conditions.

## At a Glance

| Protocol | Question | Planned methods | Reported as |
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

The planned benchmark asks:

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

## Planned Experimental Protocols

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

Planned methods:

| Method              | Role                                       | Main Assumption                    |
| ------------------- | ------------------------------------------ | ---------------------------------- |
| YOLO-seg fine-tuned | Supervised real-time segmentation baseline | Uses the same training split masks |
| Cellpose fine-tuned | Optional supervised specialist baseline    | Uses the same training split masks |

This protocol is intended to answer:

> If a small number of target-domain labels are available, how much performance can task-specific supervised training buy compared with zero-shot methods?

Supervised results will be reported with training-budget metadata, including:

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

## Planned Robustness Perturbations

The benchmark will evaluate each method on clean images and controlled perturbations.

Candidate perturbations:

| Perturbation              | Purpose                                          |
| ------------------------- | ------------------------------------------------ |
| Gaussian or Poisson noise | Test robustness to shot noise and sensor noise   |
| Gaussian blur             | Test robustness to defocus or optical blur       |
| Downsample then upsample  | Test robustness to undersampling                 |
| Contrast inversion        | Test robustness to intensity convention changes  |
| Intensity scaling         | Test robustness to underexposure or overexposure |
| Channel swap              | Test robustness to channel-order assumptions     |
| Object scale change       | Test robustness to object-size variation         |

Most perturbations are chosen to stress-test documented robustness claims around noise, blur, undersampling, contrast inversion, channel order, and object scale. Intensity scaling is treated as an additional exposure or brightness stress test rather than a central claim-matching perturbation.

The final perturbation set will be kept small to avoid turning this proof-of-work project into a large benchmark paper.

## Planned Metrics

### Segmentation Metrics

| Metric                  | Purpose                               |
| ----------------------- | ------------------------------------- |
| Object-level $F_1$      | Main instance detection metric        |
| Mean IoU                | Mask overlap quality                  |
| Dice score              | Mask overlap quality                  |
| Count error             | Cell or nucleus counting reliability  |
| Missed-object rate      | False negative behavior               |
| False-positive rate     | Spurious instance behavior            |
| Over-segmentation rate  | One object split into multiple masks  |
| Under-segmentation rate | Multiple objects merged into one mask |

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

The planned policy is:

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

Planned outputs include:

* `results/*.csv` for metric summaries;
* `figures/*.png` for robustness and failure-case visualization;
* per-method qualitative examples;
* a short technical memo;
* a failure-case taxonomy;
* reproducibility notes for model versions, prompts, and evaluation settings.

## Expected Diagnostic Questions

The analysis is expected to focus on questions such as:

* whether Cellpose-SAM shows smaller robustness drops under the perturbations associated with its documented robustness claims;
* whether SAM2 automatic mask generation fails mainly through missed objects, false positives, over-segmentation, or under-segmentation in dense microscopy images;
* whether SAM2 AMG quality improves under a small future parameter-sensitivity check, for example grid density, IoU threshold, stability threshold, minimum mask area, and optional post-processing support;
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

SAM-style models rely on geometric prompts such as points, boxes, or masks. The SAM2 automatic mask generator operationalizes this by sampling grid-based point prompts and filtering candidate masks. For this project, SAM2 is planned to be evaluated only in automatic mask generation mode, without manual or ground-truth-derived prompts.

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

YOLO-seg is planned as a supervised real-time segmentation baseline, not as a zero-shot method. It will be evaluated under a separate fine-tuning protocol if implementation time allows.

### Classical Segmentation Baselines

* [scikit-image watershed example](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html)
* [CellProfiler IdentifyPrimaryObjects documentation](https://cellprofiler-manual.s3.amazonaws.com/CPmanual/IdentifyPrimaryObjects.html)

Classical thresholding and watershed baselines are included to provide an interpretable lower-bound reference.

## Current Status

This repository is currently at the initial design stage.

Planned next steps:

1. select one or two open microscopy datasets with ground-truth instance masks;
2. define the exact evaluation contract;
3. implement classical and Cellpose baselines;
4. add perturbation generation scripts;
5. add SAM2 automatic mask generation;
6. add evaluation and visualization scripts;
7. optionally add YOLO fine-tuning and VLM segmentation protocols;
8. document APG or CellSAM-style prompt generation as future work rather than a main-protocol requirement;
9. optionally add a small SAM2 AMG parameter-sensitivity smoke test after the PoW baseline path is stable;
10. write a short technical memo summarizing results and failure cases.

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
