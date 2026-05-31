# Proof-of-Work Checklist

This checklist tracks the minimum evidence needed for the project to be a credible
proof-of-work benchmark.

## Repository Structure

- [x] Local data ignored by git.
- [x] Environment setup documented.
- [x] Dataset source and local layout documented.
- [x] Root-level `technical_memo.md` added.
- [ ] Shared source modules under `src/`.
- [ ] Reproducible run scripts under `scripts/`.
- [ ] Generated CSV outputs under `csv/`.
- [ ] Generated figure outputs under `figures/`.

## Data Pipeline

- [ ] DSB2018 stage 1 train images can be enumerated.
- [ ] Per-instance PNG masks can be counted and loaded.
- [ ] Stage 1 solution RLE file can be read.
- [ ] Dataset audit CSV is generated.
- [ ] Dataset audit figures are generated.

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
- [ ] Baseline metrics written to `csv/`.
- [ ] Baseline qualitative overlays written to `figures/`.

## Later Protocols

- [ ] Cellpose default protocol.
- [ ] Cellpose-SAM protocol.
- [ ] SAM2 automatic mask generator protocol.
- [ ] YOLO-seg supervised adaptation protocol.
- [ ] Gemini output-validity protocol.
