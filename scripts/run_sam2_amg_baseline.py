#!/usr/bin/env python
"""Run a small SAM2 automatic mask generator baseline on DSB2018 train images."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellseg-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cellseg_robustness.data import image_to_gray_float, load_train_example, stage1_train_image_dirs
from cellseg_robustness.metrics import compute_instance_metrics, relabel_sequential
from cellseg_robustness.paths import FIGURES_DIR, RESULT_SUBDIRS, ensure_output_dirs
from cellseg_robustness.visualization import overlay_truth_prediction


METHOD = "sam2_amg"
IOU_THRESHOLD = 0.5
DEFAULT_LIMIT = 20
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT = REPO_ROOT / "data" / "checkpoints" / "sam2.1_hiera_large.pt"


def selected_image_dirs(limit: int) -> list[Path]:
    """Use the same deterministic subset as the Otsu and Cellpose-SAM baselines."""
    image_dirs = stage1_train_image_dirs()
    if limit >= len(image_dirs):
        return image_dirs
    indices = np.linspace(0, len(image_dirs) - 1, num=limit, dtype=int)
    return [image_dirs[int(index)] for index in indices]


def sam2_rgb_input(image: np.ndarray) -> np.ndarray:
    """Convert grayscale/RGB/RGBA microscopy image to uint8 RGB for SAM2."""
    if image.ndim == 2:
        gray = image_to_gray_float(image)
        rgb = np.repeat(gray[..., None], 3, axis=2)
    else:
        rgb = image[..., :3]
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.float32)
            max_value = float(rgb.max()) if rgb.size else 0.0
            if max_value > 1.0:
                rgb /= 255.0 if max_value <= 255.0 else max_value
            rgb = np.clip(rgb, 0.0, 1.0)
        else:
            return rgb
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def masks_to_instance_labels(masks: list[dict[str, object]], shape: tuple[int, int]) -> np.ndarray:
    """Convert overlapping AMG masks into one non-overlapping instance label image."""
    labeled = np.zeros(shape, dtype=np.int32)
    occupied = np.zeros(shape, dtype=bool)

    def sort_key(mask: dict[str, object]) -> tuple[float, float, int]:
        return (
            float(mask.get("predicted_iou", 0.0)),
            float(mask.get("stability_score", 0.0)),
            int(mask.get("area", 0)),
        )

    instance_id = 1
    for mask in sorted(masks, key=sort_key, reverse=True):
        segmentation = np.asarray(mask["segmentation"], dtype=bool)
        remaining = segmentation & ~occupied
        if int(remaining.sum()) < 15:
            continue
        labeled[remaining] = instance_id
        occupied |= remaining
        instance_id += 1

    return relabel_sequential(labeled)


def predict_sam2(generator: SAM2AutomaticMaskGenerator, image: np.ndarray) -> np.ndarray:
    """Predict instance masks with SAM2 automatic mask generation."""
    rgb = sam2_rgb_input(image)
    masks = generator.generate(rgb)
    return masks_to_instance_labels(masks, image.shape[:2])


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
    fig.savefig(FIGURES_DIR / "sam2_amg_subset_overlay_examples.png", dpi=160)
    plt.close(fig)


def save_metric_bars(metrics: pd.DataFrame) -> None:
    summary = metrics[["object_f1", "mean_matched_iou", "mean_matched_dice"]].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(summary.index, summary.values, color=["#2563eb", "#16a34a", "#dc2626"])
    ax.set_ylim(0, 1)
    ax.set_title("SAM2 AMG Subset Mean Metrics")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sam2_amg_subset_metric_means.png", dpi=160)
    plt.close(fig)


def save_count_scatter(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(metrics["true_instances"], metrics["pred_instances"], alpha=0.75, color="#0f766e")
    max_count = int(max(metrics["true_instances"].max(), metrics["pred_instances"].max()))
    ax.plot([0, max_count], [0, max_count], color="#111827", linewidth=1, linestyle="--")
    ax.set_title("SAM2 AMG Count Agreement")
    ax.set_xlabel("Ground-truth instances")
    ax.set_ylabel("Predicted instances")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sam2_amg_subset_count_scatter.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    if not SAM2_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing SAM2 checkpoint: {SAM2_CHECKPOINT}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(SAM2_CONFIG, str(SAM2_CHECKPOINT), device=device)
    generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=24,
        points_per_batch=64,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.9,
        box_nms_thresh=0.7,
        crop_n_layers=0,
        min_mask_region_area=15,
        output_mode="binary_mask",
    )

    rows: list[dict[str, object]] = []
    overlay_examples: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []

    autocast_enabled = device == "cuda"
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        for image_dir in selected_image_dirs(DEFAULT_LIMIT):
            image_id, image, truth = load_train_example(image_dir)
            start = time.perf_counter()
            prediction = predict_sam2(generator, image)
            latency_ms = (time.perf_counter() - start) * 1000
            metrics = compute_instance_metrics(truth, prediction, iou_threshold=IOU_THRESHOLD)

            rows.append(
                {
                    "split": "stage1_train_subset",
                    "image_id": image_id,
                    "method": METHOD,
                    "perturbation": "clean",
                    "iou_threshold": IOU_THRESHOLD,
                    "sam2_config": SAM2_CONFIG,
                    "checkpoint": SAM2_CHECKPOINT.relative_to(REPO_ROOT).as_posix(),
                    "gpu": device == "cuda",
                    "latency_ms": round(latency_ms, 3),
                    **asdict(metrics),
                }
            )

            if len(overlay_examples) < 6:
                overlay_examples.append((image_id, image, truth, prediction))

    metrics_df = pd.DataFrame(rows)
    output_path = RESULT_SUBDIRS["baselines"] / "sam2_amg_clean_subset_metrics.csv"
    metrics_df.to_csv(output_path, index=False)

    save_overlay_grid(overlay_examples)
    save_metric_bars(metrics_df)
    save_count_scatter(metrics_df)

    print(f"Wrote {output_path}")
    print(f"Wrote {FIGURES_DIR / 'sam2_amg_subset_overlay_examples.png'}")
    print(f"Wrote {FIGURES_DIR / 'sam2_amg_subset_metric_means.png'}")
    print(f"Wrote {FIGURES_DIR / 'sam2_amg_subset_count_scatter.png'}")


if __name__ == "__main__":
    main()
