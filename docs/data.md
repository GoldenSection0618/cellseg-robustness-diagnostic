# Data

This project uses the 2018 Data Science Bowl dataset from Kaggle:

https://www.kaggle.com/competitions/data-science-bowl-2018

The dataset is used for microscopy nucleus instance segmentation experiments. The
local copy is unpacked under `data/raw/dsb2018/`, which is ignored by git.

## Local Structure

```text
data/raw/dsb2018/
  stage1_train/
    <image_id>/
      images/
        <image_id>.png
      masks/
        <instance_mask>.png
  stage1_test/
    <image_id>/
      images/
        <image_id>.png
  stage2_test_final/
    <image_id>/
      images/
        <image_id>.png
  solutions/
    stage1_solution.csv
    stage1_train_labels.csv
    stage1_sample_submission.csv
    stage2_sample_submission_final.csv
data/checkpoints/
  sam2.1_hiera_large.pt
```

`stage1_train/` contains training images and per-instance binary masks. Each file in
an image's `masks/` directory is one annotated nucleus instance.

`stage1_test/` contains the 65-image stage 1 test set. Its ground truth is stored in
`solutions/stage1_solution.csv` as run-length encoded masks.

`stage2_test_final/` contains additional test images without local ground-truth masks,
so it is not used for the main metric-based evaluation unless labels are added later.

`solutions/stage1_train_labels.csv` provides the stage 1 training masks in Kaggle RLE
format and can be used to cross-check the PNG mask directories.

## Local Checkpoints

Large model checkpoints are stored under `data/checkpoints/`, which is ignored by
git.

The current SAM2 baseline expects:

```text
data/checkpoints/sam2.1_hiera_large.pt
```

This checkpoint is intended for SAM2 automatic mask generation only, without manual
prompts and without ground-truth-derived prompts.
