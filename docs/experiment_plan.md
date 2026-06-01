# Experiment Plan

The benchmark is built in small increments so every stage leaves verifiable outputs.

## Phase 1: Data Audit

Goal: prove that the local DSB2018 data can be read consistently.

Outputs:

- `results/dataset/dataset_inventory.csv`
- `results/dataset/dataset_summary.csv`
- `figures/dataset_split_counts.png`
- `figures/dataset_train_instance_count_hist.png`
- `figures/dataset_image_size_scatter.png`

## Phase 2: Classical Baseline

Goal: run an Otsu + watershed baseline on a small fixed subset.

Outputs:

- metrics CSV under `results/baselines/`;
- qualitative overlays under `figures/`;
- notes added to `technical_memo.md`.

## Phase 3: Metric Hardening

Goal: stabilize instance matching and segmentation metrics before adding larger
models.

Required metrics:

- object-level F1;
- mean matched IoU;
- Dice;
- count error;
- false positives and missed objects.

## Phase 3a: Perturbation Smoke Test

Goal: verify that controlled image perturbations can be applied and evaluated on the
existing small subset before running any full robustness sweep.

Initial perturbations:

- clean;
- Gaussian noise;
- Gaussian blur;
- downsample then upsample;
- contrast inversion.

Initial method:

- Otsu + watershed only.

## Phase 4: Zero-Shot Model Protocols

Goal: run pretrained or out-of-the-box methods under the same data and output
contract.

Initial methods:

- Cellpose 4.1.1 default;
- Cellpose-SAM workflow;
- SAM2 automatic mask generation.

## Phase 5: Separate Follow-Up Protocols

Goal: keep supervised and VLM experiments separate from the main zero-shot ranking.

Follow-up protocols:

- YOLO-seg small-label supervised adaptation;
- Gemini segmentation output-validity checks.
