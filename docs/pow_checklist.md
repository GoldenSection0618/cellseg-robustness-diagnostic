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

- [ ] Internal instance-mask representation defined.
- [ ] Ground-truth mask loading implemented.
- [ ] Prediction mask loading or generation implemented.
- [ ] Instance IoU matching implemented.
- [ ] Object-level F1 implemented.
- [ ] Count error implemented.
- [ ] Dice or mean IoU implemented.

## First Baseline

- [ ] Otsu + watershed predictor implemented.
- [ ] Baseline runs on a small fixed subset.
- [ ] Baseline metrics written to `results/baselines/`.
- [ ] Baseline qualitative overlays written to `figures/`.

## Later Protocols

- [ ] Cellpose default protocol.
- [ ] Cellpose-SAM protocol.
- [ ] SAM2 automatic mask generator protocol.
- [ ] YOLO-seg supervised adaptation protocol.
- [ ] Gemini output-validity protocol.
