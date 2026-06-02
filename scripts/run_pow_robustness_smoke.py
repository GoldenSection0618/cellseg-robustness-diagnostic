#!/usr/bin/env python
"""Run a small robustness smoke test across completed PoW baselines."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellseg-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from cellpose import models
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cellseg_robustness.data import load_train_example, stage1_train_image_dirs
from cellseg_robustness.metrics import compute_instance_metrics
from cellseg_robustness.paths import FIGURES_DIR, RESULT_SUBDIRS, ensure_output_dirs
from cellseg_robustness.perturbations import Perturbation, apply_perturbation, smoke_test_perturbations
from cellseg_robustness.visualization import overlay_truth_prediction
from run_cellpose_cpsam_baseline import predict_cellpose
from run_otsu_watershed_baseline import otsu_watershed_predict
from run_sam2_amg_baseline import (
    SAM2_CHECKPOINT,
    SAM2_CONFIG,
    predict_sam2,
)


IOU_THRESHOLD = 0.5
DEFAULT_LIMIT = 5
METHOD_ORDER = ["otsu_watershed", "cellpose_cpsam", "sam2_amg"]
METHOD_LABELS = {
    "otsu_watershed": "Otsu + watershed",
    "cellpose_cpsam": "Cellpose-SAM",
    "sam2_amg": "SAM2 AMG",
}
METHOD_COLORS = {
    "otsu_watershed": "#f97316",
    "cellpose_cpsam": "#0891b2",
    "sam2_amg": "#6366f1",
}


def selected_image_dirs(limit: int) -> list[Path]:
    """Use the same deterministic spread as the clean subset scripts."""
    image_dirs = stage1_train_image_dirs()
    if limit >= len(image_dirs):
        return image_dirs
    indices = np.linspace(0, len(image_dirs) - 1, num=limit, dtype=int)
    return [image_dirs[int(index)] for index in indices]


def build_predictors() -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    """Initialize each completed PoW baseline once."""
    use_gpu = torch.cuda.is_available()
    cellpose_model = models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam")

    if not SAM2_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing SAM2 checkpoint: {SAM2_CHECKPOINT}")
    sam2_device = "cuda" if use_gpu else "cpu"
    sam2_model = build_sam2(SAM2_CONFIG, str(SAM2_CHECKPOINT), device=sam2_device)
    sam2_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=24,
        points_per_batch=64,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.9,
        box_nms_thresh=0.7,
        crop_n_layers=0,
        min_mask_region_area=15,
        output_mode="binary_mask",
    )

    def predict_otsu(image: np.ndarray) -> np.ndarray:
        return otsu_watershed_predict(image)

    def predict_cpsam(image: np.ndarray) -> np.ndarray:
        return predict_cellpose(cellpose_model, image)

    def predict_sam2_amg(image: np.ndarray) -> np.ndarray:
        return predict_sam2(sam2_generator, image)

    return {
        "otsu_watershed": predict_otsu,
        "cellpose_cpsam": predict_cpsam,
        "sam2_amg": predict_sam2_amg,
    }


def summarize(metrics: pd.DataFrame, perturbations: list[Perturbation]) -> pd.DataFrame:
    summary = (
        metrics.groupby(["method", "perturbation"], as_index=False)
        .agg(
            images=("image_id", "count"),
            mean_object_f1=("object_f1", "mean"),
            mean_matched_iou=("mean_matched_iou", "mean"),
            mean_matched_dice=("mean_matched_dice", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_absolute_count_error=("absolute_count_error", "mean"),
            median_latency_ms=("latency_ms", "median"),
            mean_latency_ms=("latency_ms", "mean"),
        )
        .round(4)
    )

    clean_f1 = (
        summary[summary["perturbation"] == "clean"]
        .set_index("method")["mean_object_f1"]
        .to_dict()
    )
    summary["clean_mean_object_f1"] = summary["method"].map(clean_f1)
    summary["absolute_object_f1_drop"] = (
        summary["clean_mean_object_f1"] - summary["mean_object_f1"]
    ).round(4)
    summary["relative_object_f1_drop"] = np.where(
        summary["clean_mean_object_f1"] > 0,
        summary["absolute_object_f1_drop"] / summary["clean_mean_object_f1"],
        0.0,
    ).round(4)
    summary["method_label"] = summary["method"].map(METHOD_LABELS)
    summary["method"] = pd.Categorical(summary["method"], categories=METHOD_ORDER, ordered=True)
    summary["perturbation"] = pd.Categorical(
        summary["perturbation"],
        categories=[perturbation.name for perturbation in perturbations],
        ordered=True,
    )
    return summary.sort_values(["method", "perturbation"]).reset_index(drop=True)


def save_mean_f1_plot(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for method in METHOD_ORDER:
        method_summary = summary[summary["method"] == method]
        ax.plot(
            method_summary["perturbation"].astype(str),
            method_summary["mean_object_f1"],
            marker="o",
            linewidth=2,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    ax.set_ylim(0, 1)
    ax.set_title("PoW Robustness Smoke: Mean Object F1")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Mean object F1")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "robustness_pow_smoke_mean_f1.png", dpi=160)
    plt.close(fig)


def save_relative_drop_plot(summary: pd.DataFrame) -> None:
    plot_frame = summary[summary["perturbation"] != "clean"].copy()
    perturbations = plot_frame["perturbation"].astype(str).drop_duplicates().tolist()
    x = np.arange(len(perturbations))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for index, method in enumerate(METHOD_ORDER):
        method_summary = plot_frame[plot_frame["method"] == method]
        ax.bar(
            x + (index - 1) * width,
            method_summary["relative_object_f1_drop"],
            width=width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_title("PoW Robustness Smoke: Relative Object F1 Drop")
    ax.set_xlabel("Perturbation")
    ax.set_ylabel("Relative F1 drop from clean")
    ax.set_xticks(x, labels=perturbations, rotation=20)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "robustness_pow_smoke_relative_f1_drop.png", dpi=160)
    plt.close(fig)


def save_method_condition_heatmap(summary: pd.DataFrame) -> None:
    pivot = summary.pivot(index="method_label", columns="perturbation", values="mean_object_f1")
    pivot = pivot.loc[[METHOD_LABELS[method] for method in METHOD_ORDER]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(pivot.shape[1]), labels=[str(column) for column in pivot.columns], rotation=20)
    ax.set_yticks(range(pivot.shape[0]), labels=pivot.index)
    ax.set_title("PoW Robustness Smoke: Mean Object F1 Heatmap")

    for row_index in range(pivot.shape[0]):
        for column_index in range(pivot.shape[1]):
            value = pivot.iat[row_index, column_index]
            text_color = "white" if value < 0.55 else "black"
            ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", color=text_color)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Mean object F1")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "robustness_pow_smoke_method_condition_heatmap.png", dpi=160)
    plt.close(fig)


def save_overlay_examples(
    examples: list[tuple[str, str, str, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    columns = 3
    rows = int(np.ceil(len(examples) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, 3.8 * rows))
    axes_array = np.atleast_1d(axes).ravel()

    for ax, (method, image_id, perturbation, image, truth, prediction) in zip(axes_array, examples):
        ax.imshow(overlay_truth_prediction(image, truth, prediction))
        ax.set_title(f"{METHOD_LABELS[method]}\n{image_id[:8]}... {perturbation}")
        ax.axis("off")

    for ax in axes_array[len(examples) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "robustness_pow_smoke_overlay_examples.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    perturbations = smoke_test_perturbations()
    image_dirs = selected_image_dirs(DEFAULT_LIMIT)
    predictors = build_predictors()
    autocast_enabled = torch.cuda.is_available()

    rows: list[dict[str, object]] = []
    overlay_examples: list[tuple[str, str, str, np.ndarray, np.ndarray, np.ndarray]] = []

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        for image_dir in image_dirs:
            image_id, image, truth = load_train_example(image_dir)
            for perturbation in perturbations:
                perturbed = apply_perturbation(image, perturbation)
                for method in METHOD_ORDER:
                    start = time.perf_counter()
                    prediction = predictors[method](perturbed)
                    latency_ms = (time.perf_counter() - start) * 1000
                    metrics = compute_instance_metrics(truth, prediction, iou_threshold=IOU_THRESHOLD)
                    rows.append(
                        {
                            "split": "stage1_train_tiny_subset",
                            "image_id": image_id,
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "perturbation": perturbation.name,
                            "perturbation_params": perturbation.params,
                            "iou_threshold": IOU_THRESHOLD,
                            "latency_ms": round(latency_ms, 3),
                            **asdict(metrics),
                        }
                    )

                    if image_id == image_dirs[0].name and len(overlay_examples) < 15:
                        overlay_examples.append((method, image_id, perturbation.name, perturbed, truth, prediction))

    metrics_df = pd.DataFrame(rows)
    summary_df = summarize(metrics_df, perturbations)

    metrics_path = RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_smoke_metrics.csv"
    summary_path = RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_smoke_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    save_mean_f1_plot(summary_df)
    save_relative_drop_plot(summary_df)
    save_method_condition_heatmap(summary_df)
    save_overlay_examples(overlay_examples)

    print(f"Wrote {metrics_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {FIGURES_DIR / 'robustness_pow_smoke_mean_f1.png'}")
    print(f"Wrote {FIGURES_DIR / 'robustness_pow_smoke_relative_f1_drop.png'}")
    print(f"Wrote {FIGURES_DIR / 'robustness_pow_smoke_method_condition_heatmap.png'}")
    print(f"Wrote {FIGURES_DIR / 'robustness_pow_smoke_overlay_examples.png'}")


if __name__ == "__main__":
    main()
