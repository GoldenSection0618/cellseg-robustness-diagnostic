#!/usr/bin/env python
"""Compare completed clean-subset baseline outputs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT / "src"))

from cellseg_robustness.paths import RESULT_SUBDIRS, ensure_output_dirs
from cellseg_robustness.summary import FAILURE_RATE_AGGREGATIONS, add_failure_rate_columns

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
    metrics = add_failure_rate_columns(metrics)
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
            **FAILURE_RATE_AGGREGATIONS,
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


def main() -> None:
    ensure_output_dirs()
    metrics = load_metrics()
    summary = summarize(metrics)

    combined_path = RESULT_SUBDIRS["baselines"] / "clean_subset_baseline_metrics_long.csv"
    summary_path = RESULT_SUBDIRS["baselines"] / "clean_subset_baseline_summary.csv"
    metrics.to_csv(combined_path, index=False)
    summary.to_csv(summary_path, index=False)
    save_failure_cases(metrics)

    print(f"Wrote {combined_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {RESULT_SUBDIRS['baselines'] / 'clean_subset_baseline_failure_cases.csv'}")


if __name__ == "__main__":
    main()
