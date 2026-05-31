# Technical Memo

## Current Objective

Build a compact proof-of-work benchmark for microscopy nucleus instance segmentation
robustness. The first implementation target is an end-to-end skeleton that can:

1. read the local DSB2018 data;
2. produce tabular analysis outputs in `csv/`;
3. produce visual analysis outputs in `figures/`;
4. support later predictors through a common evaluation pipeline.

## Dataset

The current dataset is the Kaggle 2018 Data Science Bowl dataset, unpacked under
`data/raw/dsb2018/`.

The primary metric-bearing data for the first benchmark pass is:

- `stage1_train/`: images plus per-instance PNG masks;
- `stage1_test/`: held-out images;
- `solutions/stage1_solution.csv`: RLE masks for stage 1 test.

`stage2_test_final/` is retained locally but is not part of the first metric-bearing
evaluation because it does not include local ground-truth masks.

## First Milestone

The first milestone is intentionally small:

- audit dataset structure and image/mask counts;
- write audit tables to `csv/`;
- write dataset diagnostic plots to `figures/`;
- keep all generated outputs reproducible from code.

After this milestone, the next step is to add a classical Otsu + watershed baseline
that uses the same output contract.

## Output Contract

Experiment and analysis scripts should write:

- CSV tables to `csv/`;
- PNG figures to `figures/`;
- no generated result files into `docs/` or source directories.

Each generated CSV should have enough columns to identify the input split, image id,
method or analysis name, and any relevant parameters.

Each generated figure should be produced by a script and saved directly under
`figures/`.

## Current Scope Boundary

This memo tracks proof-of-work engineering progress. It is not a final paper-style
report. Model comparisons, robustness conclusions, and failure taxonomy should be
added only after the evaluation pipeline is running end to end.
