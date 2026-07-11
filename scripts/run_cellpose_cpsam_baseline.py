#!/usr/bin/env python
"""Run a small Cellpose-SAM baseline on DSB2018 train images."""

from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from cellpose import models

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cellseg_robustness.data import image_to_gray_float, load_train_example, stage1_train_image_dirs
from cellseg_robustness.metrics import compute_instance_metrics, relabel_sequential
from cellseg_robustness.paths import FIGURES_DIR, RESULT_SUBDIRS, ensure_output_dirs
from cellseg_robustness.plot_style import save_png
from cellseg_robustness.visualization import overlay_truth_prediction


METHOD = "cellpose_cpsam"
IOU_THRESHOLD = 0.5
DEFAULT_LIMIT = 20
CELLPOSE_INPUT_MODE = "gray_mean"
CELLPOSE_DIAMETER = 15.0


def selected_image_dirs(limit: int) -> list[Path]:
    """Use the same deterministic subset as the Otsu baseline."""
    image_dirs = stage1_train_image_dirs()
    if limit >= len(image_dirs):
        return image_dirs
    indices = np.linspace(0, len(image_dirs) - 1, num=limit, dtype=int)
    return [image_dirs[int(index)] for index in indices]


def cellpose_input(image: np.ndarray) -> np.ndarray:
    """Prepare DSB2018 image input for Cellpose-SAM."""
    if CELLPOSE_INPUT_MODE == "gray_mean":
        return (image_to_gray_float(image) * 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] > 3:
        return image[..., :3]
    return image


def predict_cellpose(model: models.CellposeModel, image: np.ndarray) -> np.ndarray:
    """Predict instance masks with Cellpose-SAM."""
    model_input = cellpose_input(image)
    prediction, *_ = model.eval(
        model_input,
        channel_axis=-1 if model_input.ndim == 3 else None,
        normalize=True,
        diameter=CELLPOSE_DIAMETER,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        min_size=15,
    )
    return relabel_sequential(prediction.astype(np.int32))


def save_overlay_grid(examples: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]) -> None:
    columns = 2
    rows = int(np.ceil(len(examples) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(9, 4.2 * rows))
    axes_array = np.atleast_1d(axes).ravel()

    for ax, (image_id, image, truth, prediction) in zip(axes_array, examples):
        ax.imshow(overlay_truth_prediction(image, truth, prediction))
        ax.set_title(f"{image_id[:10]}...  truth=green pred=red")
        ax.axis("off")

    for ax in axes_array[len(examples) :]:
        ax.axis("off")

    fig.tight_layout()
    save_png(fig, FIGURES_DIR / "cellpose_cpsam_subset_overlay_examples.png")
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    use_gpu = torch.cuda.is_available()
    model = models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam")
    rows: list[dict[str, object]] = []
    overlay_examples: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []

    for image_dir in selected_image_dirs(DEFAULT_LIMIT):
        image_id, image, truth = load_train_example(image_dir)
        start = time.perf_counter()
        prediction = predict_cellpose(model, image)
        latency_ms = (time.perf_counter() - start) * 1000
        metrics = compute_instance_metrics(truth, prediction, iou_threshold=IOU_THRESHOLD)

        rows.append(
            {
                "split": "stage1_train_subset",
                "image_id": image_id,
                "method": METHOD,
                "perturbation": "clean",
                "iou_threshold": IOU_THRESHOLD,
                "gpu": use_gpu,
                "latency_ms": round(latency_ms, 3),
                **asdict(metrics),
            }
        )

        if len(overlay_examples) < 6:
            overlay_examples.append((image_id, image, truth, prediction))

    metrics_df = pd.DataFrame(rows)
    output_path = RESULT_SUBDIRS["baselines"] / "cellpose_cpsam_clean_subset_metrics.csv"
    metrics_df.to_csv(output_path, index=False)

    save_overlay_grid(overlay_examples)

    print(f"Wrote {output_path}")
    print(f"Wrote {FIGURES_DIR / 'cellpose_cpsam_subset_overlay_examples.png'}")


if __name__ == "__main__":
    main()
