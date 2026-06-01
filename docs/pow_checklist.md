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
- [x] Otsu + watershed runs on a small fixed subset.
- [x] Otsu + watershed metrics written to `results/baselines/`.
- [x] Otsu + watershed qualitative overlays written to `figures/`.

## Second Baseline

- [x] Cellpose default availability audited in the current `cell` environment.
- [ ] Cellpose default predictor implemented.
- [ ] Cellpose default runs on the same small fixed subset.
- [ ] Cellpose default metrics written to `results/baselines/`.
- [ ] Cellpose default qualitative overlays written to `figures/`.

## Third Baseline

- [x] Cellpose restoration availability audited in the current `cell` environment.
- [ ] Cellpose restoration workflow selected for a runnable environment.
- [ ] Cellpose restoration runs on the same small fixed subset.
- [ ] Cellpose restoration metrics written to `results/baselines/`.
- [ ] Cellpose restoration qualitative overlays written to `figures/`.

## Fourth Baseline

- [x] Cellpose-SAM predictor implemented.
- [x] Cellpose-SAM runs on the same small fixed subset.
- [x] Cellpose-SAM metrics written to `results/baselines/`.
- [x] Cellpose-SAM qualitative overlays written to `figures/`.

## Fifth Baseline

- [x] SAM2 checkpoint documented.
- [x] SAM2 automatic mask generator predictor implemented.
- [x] SAM2 automatic mask generator runs on the same small fixed subset.
- [x] SAM2 metrics written to `results/baselines/`.
- [x] SAM2 qualitative overlays written to `figures/`.

## Clean Baseline Comparison

- [x] Otsu + watershed, Cellpose-SAM, and SAM2 subset metrics combined.
- [x] Clean subset comparison summary written to `results/baselines/`.
- [x] Clean subset comparison figures written to `figures/`.

## Deferred Robustness Record

- [x] Controlled perturbation utilities implemented.
- [x] Minimal perturbation smoke test runs on a small fixed subset.
- [x] Robustness smoke metrics written to `results/robustness/`.
- [x] Robustness smoke figures written to `figures/`.

## Later Protocols After Clean Baselines

- [ ] Cellpose default protocol.
- [ ] Cellpose-SAM full protocol.
- [ ] SAM2 automatic mask generator protocol.
- [ ] YOLO-seg supervised adaptation protocol.
- [ ] Gemini output-validity protocol.
