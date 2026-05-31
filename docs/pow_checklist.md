# Proof-of-Work Checklist

This checklist tracks the minimum evidence needed for the project to be a credible
proof-of-work benchmark.

## Repository Structure

- [x] Local data ignored by git.
- [x] Environment setup documented.
- [x] Dataset source and local layout documented.
- [x] Root-level `technical_memo.md` added.
- [x] Shared source modules under `src/`.
- [x] Reproducible run scripts under `scripts/`.
- [x] Generated tabular outputs under categorized `results/` subdirectories.
- [x] Generated figure outputs under `figures/`.

## Data Pipeline

- [x] DSB2018 stage 1 train images can be enumerated.
- [x] Per-instance PNG masks can be counted and loaded.
- [x] Stage 1 solution RLE file can be read.
- [x] Dataset audit CSV is generated.
- [x] Dataset audit figures are generated.

## Evaluation Pipeline

- [x] Internal instance-mask representation defined.
- [x] Ground-truth mask loading implemented.
- [x] Prediction mask loading or generation implemented.
- [x] Instance IoU matching implemented.
- [x] Object-level F1 implemented.
- [x] Count error implemented.
- [x] Dice or mean IoU implemented.

## First Baseline

- [x] Otsu + watershed predictor implemented.
- [x] Baseline runs on a small fixed subset.
- [x] Baseline metrics written to `results/baselines/`.
- [x] Baseline qualitative overlays written to `figures/`.

## Model Baseline Smoke Tests

- [x] Cellpose-SAM runs on the same small fixed subset.
- [x] Cellpose-SAM metrics written to `results/baselines/`.
- [x] Cellpose-SAM qualitative overlays written to `figures/`.

## Later Protocols

- [ ] Cellpose default protocol.
- [ ] Cellpose-SAM full protocol.
- [ ] SAM2 automatic mask generator protocol.
- [ ] YOLO-seg supervised adaptation protocol.
- [ ] Gemini output-validity protocol.
