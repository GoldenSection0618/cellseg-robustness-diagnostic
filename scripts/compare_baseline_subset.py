#!/usr/bin/env python
"""Compare completed clean-subset baseline outputs."""

from __future__ import annotations

import sys
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellseg-matplotlib")

import matplotlib.pyplot as plt

sys.path.insert(0, str(REPO_ROOT / "src"))

from cellseg_robustness.paths import FIGURES_DIR, RESULT_SUBDIRS, ensure_output_dirs


BASELINE_FILES = {
    "otsu_watershed": "otsu_watershed_clean_subset_metrics.csv",
    "cellpose_cpsam": "cellpose_cpsam_clean_subset_metrics.csv",
}

METRIC_COLUMNS = [
    "object_f1",
    "mean_matched_iou",
    "mean_matched_dice",
    "precision",
    "recall",
    "absolute_count_error",
    "latency_ms",
]


def load_metrics() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for method, filename in BASELINE_FILES.items():
        path = RESULT_SUBDIRS["baselines"] / filename
        frame = pd.read_csv(path)
        frame["method"] = method
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics.groupby("method", as_index=False)
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
    return summary


def save_metric_comparison(summary: pd.DataFrame) -> None:
    plot_frame = summary.set_index("method")[
        ["mean_object_f1", "mean_matched_iou", "mean_matched_dice"]
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_frame.plot(kind="bar", ax=ax, color=["#2563eb", "#16a34a", "#dc2626"])
    ax.set_ylim(0, 1)
    ax.set_title("Clean Subset Baseline Metric Comparison")
    ax.set_xlabel("Method")
    ax.set_ylabel("Mean score")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "baseline_clean_subset_metric_comparison.png", dpi=160)
    plt.close(fig)


def save_count_error_comparison(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        summary["method"],
        summary["mean_absolute_count_error"],
        color=["#f97316", "#0891b2"],
    )
    ax.set_title("Clean Subset Mean Absolute Count Error")
    ax.set_xlabel("Method")
    ax.set_ylabel("Mean absolute count error")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "baseline_clean_subset_count_error_comparison.png", dpi=160)
    plt.close(fig)


def save_latency_comparison(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary["method"], summary["median_latency_ms"], color=["#6366f1", "#14b8a6"])
    ax.set_title("Clean Subset Median Latency")
    ax.set_xlabel("Method")
    ax.set_ylabel("Median latency (ms/image)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "baseline_clean_subset_latency_comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    metrics = load_metrics()
    summary = summarize(metrics)

    combined_path = RESULT_SUBDIRS["baselines"] / "clean_subset_baseline_metrics_long.csv"
    summary_path = RESULT_SUBDIRS["baselines"] / "clean_subset_baseline_summary.csv"
    metrics.to_csv(combined_path, index=False)
    summary.to_csv(summary_path, index=False)

    save_metric_comparison(summary)
    save_count_error_comparison(summary)
    save_latency_comparison(summary)

    print(f"Wrote {combined_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {FIGURES_DIR / 'baseline_clean_subset_metric_comparison.png'}")
    print(f"Wrote {FIGURES_DIR / 'baseline_clean_subset_count_error_comparison.png'}")
    print(f"Wrote {FIGURES_DIR / 'baseline_clean_subset_latency_comparison.png'}")


if __name__ == "__main__":
    main()
