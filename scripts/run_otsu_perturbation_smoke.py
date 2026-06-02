#!/usr/bin/env python
"""Run the earlier Otsu-only robustness smoke test on a tiny fixed subset."""

from __future__ import annotations

import sys
import time
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellseg-matplotlib")

import matplotlib.pyplot as plt

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cellseg_robustness.data import load_train_example, stage1_train_image_dirs
from cellseg_robustness.metrics import compute_instance_metrics
from cellseg_robustness.paths import FIGURES_DIR, RESULT_SUBDIRS, ensure_output_dirs
from cellseg_robustness.perturbations import apply_perturbation, smoke_test_perturbations
from cellseg_robustness.visualization import overlay_truth_prediction
from run_otsu_watershed_baseline import otsu_watershed_predict


METHOD = "otsu_watershed"
IOU_THRESHOLD = 0.5
DEFAULT_LIMIT = 5


def selected_image_dirs(limit: int) -> list[Path]:
    image_dirs = stage1_train_image_dirs()
    if limit >= len(image_dirs):
        return image_dirs
    indices = np.linspace(0, len(image_dirs) - 1, num=limit, dtype=int)
    return [image_dirs[int(index)] for index in indices]


def save_drop_plot(summary: pd.DataFrame) -> None:
    plot_frame = summary[summary["perturbation"] != "clean"].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(plot_frame["perturbation"], plot_frame["relative_object_f1_drop"], color="#ef4444")
    ax.set_title("Otsu Watershed Robustness Smoke: Relative F1 Drop")
    ax.set_xlabel("Perturbation")
    ax.set_ylabel("Relative object F1 drop")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "robustness_otsu_smoke_relative_f1_drop.png", dpi=160)
    plt.close(fig)


def save_metric_plot(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(summary["perturbation"], summary["mean_object_f1"], marker="o", label="Object F1")
    ax.plot(summary["perturbation"], summary["mean_matched_iou"], marker="o", label="Matched IoU")
    ax.set_ylim(0, 1)
    ax.set_title("Otsu Watershed Robustness Smoke: Mean Scores")
    ax.set_xlabel("Perturbation")
    ax.set_ylabel("Mean score")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "robustness_otsu_smoke_mean_scores.png", dpi=160)
    plt.close(fig)


def save_overlay_examples(
    examples: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    columns = 2
    rows = int(np.ceil(len(examples) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(9, 4.2 * rows))
    axes_array = np.atleast_1d(axes).ravel()

    for ax, (image_id, perturbation, image, truth, prediction) in zip(axes_array, examples):
        ax.imshow(overlay_truth_prediction(image, truth, prediction))
        ax.set_title(f"{image_id[:10]}... {perturbation}")
        ax.axis("off")

    for ax in axes_array[len(examples) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "robustness_otsu_smoke_overlay_examples.png", dpi=160)
    plt.close(fig)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics.groupby("perturbation", as_index=False)
        .agg(
            images=("image_id", "count"),
            mean_object_f1=("object_f1", "mean"),
            mean_matched_iou=("mean_matched_iou", "mean"),
            mean_absolute_count_error=("absolute_count_error", "mean"),
            mean_latency_ms=("latency_ms", "mean"),
        )
        .round(4)
    )
    clean_f1 = float(summary.loc[summary["perturbation"] == "clean", "mean_object_f1"].iloc[0])
    summary["absolute_object_f1_drop"] = (clean_f1 - summary["mean_object_f1"]).round(4)
    summary["relative_object_f1_drop"] = (
        summary["absolute_object_f1_drop"] / clean_f1 if clean_f1 else 0.0
    ).round(4)
    order = [p.name for p in smoke_test_perturbations()]
    summary["perturbation"] = pd.Categorical(summary["perturbation"], categories=order, ordered=True)
    return summary.sort_values("perturbation").reset_index(drop=True)


def main() -> None:
    ensure_output_dirs()
    perturbations = smoke_test_perturbations()
    rows: list[dict[str, object]] = []
    overlay_examples: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []

    for image_dir in selected_image_dirs(DEFAULT_LIMIT):
        image_id, image, truth = load_train_example(image_dir)
        for perturbation in perturbations:
            perturbed = apply_perturbation(image, perturbation)
            start = time.perf_counter()
            prediction = otsu_watershed_predict(perturbed)
            latency_ms = (time.perf_counter() - start) * 1000
            metrics = compute_instance_metrics(truth, prediction, iou_threshold=IOU_THRESHOLD)
            rows.append(
                {
                    "split": "stage1_train_tiny_subset",
                    "image_id": image_id,
                    "method": METHOD,
                    "perturbation": perturbation.name,
                    "perturbation_params": perturbation.params,
                    "iou_threshold": IOU_THRESHOLD,
                    "latency_ms": round(latency_ms, 3),
                    **asdict(metrics),
                }
            )

            if image_id == selected_image_dirs(DEFAULT_LIMIT)[0].name and len(overlay_examples) < 5:
                overlay_examples.append((image_id, perturbation.name, perturbed, truth, prediction))

    metrics_df = pd.DataFrame(rows)
    summary_df = summarize(metrics_df)

    metrics_path = RESULT_SUBDIRS["robustness"] / "otsu_watershed_perturbation_smoke_metrics.csv"
    summary_path = RESULT_SUBDIRS["robustness"] / "otsu_watershed_perturbation_smoke_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    save_metric_plot(summary_df)
    save_drop_plot(summary_df)
    save_overlay_examples(overlay_examples)

    print(f"Wrote {metrics_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {FIGURES_DIR / 'robustness_otsu_smoke_mean_scores.png'}")
    print(f"Wrote {FIGURES_DIR / 'robustness_otsu_smoke_relative_f1_drop.png'}")
    print(f"Wrote {FIGURES_DIR / 'robustness_otsu_smoke_overlay_examples.png'}")


if __name__ == "__main__":
    main()
