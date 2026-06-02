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
    "sam2_amg": "sam2_amg_clean_subset_metrics.csv",
}

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
    metrics = pd.concat(frames, ignore_index=True)
    metrics["method_label"] = metrics["method"].map(METHOD_LABELS)
    return metrics


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
    summary["method_label"] = summary["method"].map(METHOD_LABELS)
    summary["method_order"] = summary["method"].map({method: i for i, method in enumerate(METHOD_ORDER)})
    summary = summary.sort_values("method_order").drop(columns="method_order")
    return summary


def infer_failure_hint(row: pd.Series) -> str:
    """Assign a coarse failure-taxonomy hint from aggregate per-image metrics."""
    if row["recall"] < 0.35 and row["precision"] < 0.35:
        return "FN+FP"
    if row["recall"] < 0.5 and row["recall"] <= row["precision"]:
        return "FN"
    if row["precision"] < 0.5 and row["precision"] < row["recall"]:
        return "FP/OVER"
    if row["pred_instances"] > row["true_instances"] * 1.5:
        return "OVER"
    if row["pred_instances"] < row["true_instances"] * 0.5:
        return "FN/UNDER"
    if row["object_f1"] >= 0.5 and row["mean_matched_iou"] < 0.65:
        return "BOUNDARY"
    return "MIXED"


def save_failure_cases(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for method in METHOD_ORDER:
        method_metrics = metrics[metrics["method"] == method].copy()
        method_metrics = method_metrics.sort_values(
            ["object_f1", "absolute_count_error"],
            ascending=[True, False],
        ).head(5)
        rows.append(method_metrics)

    failure_cases = pd.concat(rows, ignore_index=True)
    failure_cases["failure_hint"] = failure_cases.apply(infer_failure_hint, axis=1)
    keep_columns = [
        "method",
        "method_label",
        "image_id",
        "object_f1",
        "precision",
        "recall",
        "mean_matched_iou",
        "true_instances",
        "pred_instances",
        "false_positives",
        "false_negatives",
        "absolute_count_error",
        "latency_ms",
        "failure_hint",
    ]
    failure_cases = failure_cases[keep_columns].round(4)
    output_path = RESULT_SUBDIRS["baselines"] / "clean_subset_baseline_failure_cases.csv"
    failure_cases.to_csv(output_path, index=False)
    return failure_cases


def save_metric_comparison(summary: pd.DataFrame) -> None:
    plot_frame = summary.set_index("method_label")[
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
    ax.bar(summary["method_label"], summary["mean_absolute_count_error"], color="#f97316")
    ax.set_title("Clean Subset Mean Absolute Count Error")
    ax.set_xlabel("Method")
    ax.set_ylabel("Mean absolute count error")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "baseline_clean_subset_count_error_comparison.png", dpi=160)
    plt.close(fig)


def save_latency_comparison(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary["method_label"], summary["median_latency_ms"], color="#6366f1")
    ax.set_title("Clean Subset Median Latency")
    ax.set_xlabel("Method")
    ax.set_ylabel("Median latency (ms/image)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "baseline_clean_subset_latency_comparison.png", dpi=160)
    plt.close(fig)


def save_score_distributions(metrics: pd.DataFrame) -> None:
    metric_specs = [
        ("object_f1", "Object F1", (0, 1)),
        ("mean_matched_iou", "Matched IoU", (0, 1)),
        ("precision", "Precision", (0, 1)),
        ("recall", "Recall", (0, 1)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    axes_array = axes.ravel()

    for ax, (column, title, ylim) in zip(axes_array, metric_specs):
        data = [metrics.loc[metrics["method"] == method, column].to_numpy() for method in METHOD_ORDER]
        ax.boxplot(data, tick_labels=[METHOD_LABELS[method] for method in METHOD_ORDER], showfliers=False)
        for index, values in enumerate(data, start=1):
            jitter = ((pd.Series(range(len(values))) % 5).to_numpy() - 2) * 0.025
            ax.scatter(
                index + jitter,
                values,
                s=18,
                alpha=0.6,
                color=METHOD_COLORS[METHOD_ORDER[index - 1]],
            )
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Clean Subset Per-Image Score Distributions", y=0.98)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "baseline_clean_subset_score_distributions.png", dpi=160)
    plt.close(fig)


def save_precision_recall_scatter(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for method in METHOD_ORDER:
        method_metrics = metrics[metrics["method"] == method]
        ax.scatter(
            method_metrics["recall"],
            method_metrics["precision"],
            s=40,
            alpha=0.75,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    ax.plot([0, 1], [0, 1], color="#111827", linewidth=1, linestyle="--")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("Clean Subset Precision-Recall by Image")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "baseline_clean_subset_precision_recall.png", dpi=160)
    plt.close(fig)


def save_image_method_heatmap(metrics: pd.DataFrame) -> None:
    pivot = metrics.pivot(index="image_id", columns="method", values="object_f1")
    pivot = pivot[METHOD_ORDER]
    pivot = pivot.sort_values("cellpose_cpsam", ascending=True)
    display_ids = [f"{image_id[:8]}..." for image_id in pivot.index]

    fig, ax = plt.subplots(figsize=(7, 9))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(METHOD_ORDER)), labels=[METHOD_LABELS[method] for method in METHOD_ORDER])
    ax.set_yticks(range(len(display_ids)), labels=display_ids)
    ax.set_title("Clean Subset Object F1 by Image and Method")
    ax.tick_params(axis="x", rotation=20)

    for row_index in range(pivot.shape[0]):
        for column_index in range(pivot.shape[1]):
            value = pivot.iat[row_index, column_index]
            text_color = "white" if value < 0.55 else "black"
            ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=7)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Object F1")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "baseline_clean_subset_image_method_f1_heatmap.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    metrics = load_metrics()
    summary = summarize(metrics)

    combined_path = RESULT_SUBDIRS["baselines"] / "clean_subset_baseline_metrics_long.csv"
    summary_path = RESULT_SUBDIRS["baselines"] / "clean_subset_baseline_summary.csv"
    metrics.to_csv(combined_path, index=False)
    summary.to_csv(summary_path, index=False)
    save_failure_cases(metrics)

    save_metric_comparison(summary)
    save_count_error_comparison(summary)
    save_latency_comparison(summary)
    save_score_distributions(metrics)
    save_precision_recall_scatter(metrics)
    save_image_method_heatmap(metrics)

    print(f"Wrote {combined_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {RESULT_SUBDIRS['baselines'] / 'clean_subset_baseline_failure_cases.csv'}")
    print(f"Wrote {FIGURES_DIR / 'baseline_clean_subset_metric_comparison.png'}")
    print(f"Wrote {FIGURES_DIR / 'baseline_clean_subset_count_error_comparison.png'}")
    print(f"Wrote {FIGURES_DIR / 'baseline_clean_subset_latency_comparison.png'}")
    print(f"Wrote {FIGURES_DIR / 'baseline_clean_subset_score_distributions.png'}")
    print(f"Wrote {FIGURES_DIR / 'baseline_clean_subset_precision_recall.png'}")
    print(f"Wrote {FIGURES_DIR / 'baseline_clean_subset_image_method_f1_heatmap.png'}")


if __name__ == "__main__":
    main()
