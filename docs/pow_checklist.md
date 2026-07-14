# Proof-of-Work Checklist

This checklist tracks the minimum evidence needed for the project to be a credible
proof-of-work benchmark.

## Repository Structure

- [x] Local data ignored by git.
- [x] Environment setup documented.
- [x] Dataset source and local layout documented.
- [x] Failure-case taxonomy documented.
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

- [x] Cellpose-SAM predictor implemented.
- [x] Cellpose-SAM runs on the same small fixed subset.
- [x] Cellpose-SAM metrics written to `results/baselines/`.
- [x] Cellpose-SAM qualitative overlays written to `figures/`.
- [x] Cellpose-SAM parameter diagnostic completed.
- [x] Cellpose-SAM main predictor locked to `gray_mean` input and 15 px diameter.

## Third Baseline

- [x] SAM2 checkpoint documented.
- [x] SAM2 automatic mask generator predictor implemented.
- [x] SAM2 automatic mask generator runs on the same small fixed subset.
- [x] SAM2 metrics written to `results/baselines/`.
- [x] SAM2 qualitative overlays written to `figures/`.

## Clean Baseline Comparison

- [x] Otsu + watershed, Cellpose-SAM, and SAM2 subset metrics combined.
- [x] Clean subset comparison summary written to `results/baselines/`.
- [x] Clean subset failure cases written to `results/baselines/`.
- [x] Clean subset comparison figures written to `figures/`.

## Optional Cross-Version Cellpose Baselines

- [x] Legacy Cellpose default availability audited in the current `cell` environment.
- [x] Cellpose restoration availability audited in the current `cell` environment.
- [ ] Decide whether to add a separate Cellpose3 environment for legacy `cyto3`.
- [ ] Decide whether to add a separate Cellpose3 restoration baseline.

## Deferred Robustness Record

- [x] Controlled perturbation utilities implemented.
- [x] Otsu-only perturbation smoke test runs on a small fixed subset.
- [x] PoW three-baseline robustness smoke test runs on the same small fixed subset.
- [x] Robustness smoke metrics written to `results/robustness/`.
- [x] Robustness smoke figures written to `figures/`.
- [x] Same robustness protocol extended to the deterministic 20-image clean subset.
- [x] Clean20 robustness metrics written to `results/robustness/`.
- [x] Clean20 robustness figures written to `figures/`.
- [x] Clean20 per-image robustness deltas and failure cases written to `results/robustness/`.
- [x] Clean20 consolidated summary and failure diagnostic figures written to `figures/`.
- [x] Robustness runner supports staged `full_train` runs with method selection and resume/overwrite protection.
- [x] Otsu + watershed full-train robustness run completed on all 670 `stage1_train` images.
- [x] Cellpose-SAM full-train robustness run added with `--resume`.
- [x] Cellpose-SAM input/diameter protocol locked and full-train results refreshed.
- [x] SAM2 AMG full-train robustness deferred pending parameter-sensitivity or post-processing work.
- [x] Full-train per-image deltas, failure cases, and no-prediction cases written to `results/robustness/`.
- [x] Full-train consolidated summary and failure diagnostic figures written to `figures/`.
- [x] SAM2 AMG clean20 parameter-sensitivity clean-screen run completed.
- [x] SAM2 AMG clean20 parameter-sensitivity validation run completed.
- [x] SAM2 AMG sensitivity metrics, failed-config record, failure cases, and figures written.

## Later Protocols After PoW Clean Baselines

- [ ] Optional Cellpose3 default cross-version protocol.
- [ ] Optional Cellpose3 restoration cross-version protocol.
- [ ] Optional prompted SAM2 or SAM2 post-processing repair protocol.
- [ ] Gemini output-validity protocol.

## Protocol B: YOLO Supervised Adaptation

- [x] Supervised adaptation protocol documented.
- [x] DSB2018-to-YOLO segmentation label-conversion smoke script implemented.
- [x] Deterministic 20-image YOLO label smoke completed.
- [x] YOLO label smoke manifest, summary, labels, data YAML, and overlay figure written.
- [x] Tiny YOLO training smoke.
- [x] YOLO prediction-to-repository-metrics evaluation smoke.
- [x] Fixed YOLO supervised baseline budget decided.
- [x] Fixed-budget YOLO split and label conversion completed.
- [x] Fixed-budget supervised baseline.
- [x] YOLO threshold diagnostic.
- [x] Nested YOLO label-budget conversion.
- [x] YOLO label-budget diagnostic for 100, 250, and full train-pool budgets.
- [x] YOLO11m full train-pool capacity diagnostic.
