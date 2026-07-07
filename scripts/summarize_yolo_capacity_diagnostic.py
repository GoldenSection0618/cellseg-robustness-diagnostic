#!/usr/bin/env python
"""Compare YOLO capacity diagnostic results on the fixed held-out val split."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellseg-matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cellseg_robustness.paths import FIGURES_DIR, RESULT_SUBDIRS, ensure_output_dirs


YOLO11N_METRICS = RESULT_SUBDIRS["supervised"] / "yolo_label_budget_diagnostic_full_train_pool_metrics.csv"
YOLO11M_METRICS = RESULT_SUBDIRS["supervised"] / "yolo_capacity_diagnostic_yolo11m_metrics.csv"
VAL_MANIFEST = RESULT_SUBDIRS["supervised"] / "yolo_fixed_budget_manifest.csv"
ZERO_SHOT_METRICS = RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_full_train_metrics.csv"
COMPARISON_METRICS = RESULT_SUBDIRS["supervised"] / "yolo_capacity_diagnostic_val_comparison_metrics.csv"
COMPARISON_SUMMARY = RESULT_SUBDIRS["supervised"] / "yolo_capacity_diagnostic_val_comparison_summary.csv"
FIGURE_PATH = FIGURES_DIR / "supervised_yolo_capacity_diagnostic_comparison.png"

METRIC_COLUMNS = [
    "true_instances",
    "pred_instances",
    "matched_instances",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "object_f1",
    "mean_matched_iou",
    "mean_matched_dice",
    "count_error",
    "absolute_count_error",
    "latency_ms",
]

FIGURE_LABELS = {
    "Cellpose-SAM": "Cellpose-SAM",
    "YOLO11m full train pool": "YOLO11m full",
    "YOLO11n full train pool": "YOLO11n full",
    "Otsu + watershed": "Otsu",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_outputs(overwrite: bool) -> None:
    existing = [path for path in [COMPARISON_METRICS, COMPARISON_SUMMARY, FIGURE_PATH] if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Existing capacity comparison outputs found. Use --overwrite.\n{joined}")
    if overwrite:
        for path in existing:
            path.unlink()


def normalize_yolo(frame: pd.DataFrame, method: str, method_label: str) -> pd.DataFrame:
    out = frame.copy()
    out["method"] = method
    out["method_label"] = method_label
    out["protocol"] = "supervised"
    out["condition"] = "same_held_out_val"
    return out[["image_id", "method", "method_label", "protocol", "condition", *METRIC_COLUMNS]]


def normalize_zero_shot(zero_shot: pd.DataFrame, val_ids: set[str]) -> pd.DataFrame:
    frame = zero_shot[
        (zero_shot["perturbation"] == "clean")
        & (zero_shot["image_id"].isin(val_ids))
        & (zero_shot["method"].isin(["otsu_watershed", "cellpose_cpsam"]))
    ].copy()
    frame["protocol"] = "zero_shot"
    frame["condition"] = "same_held_out_val_clean"
    return frame[["image_id", "method", "method_label", "protocol", "condition", *METRIC_COLUMNS]]


def build_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics.groupby(["method", "method_label", "protocol", "condition"], as_index=False)
        .agg(
            images=("image_id", "count"),
            mean_object_f1=("object_f1", "mean"),
            median_object_f1=("object_f1", "median"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_matched_iou=("mean_matched_iou", "mean"),
            mean_matched_dice=("mean_matched_dice", "mean"),
            mean_absolute_count_error=("absolute_count_error", "mean"),
            median_absolute_count_error=("absolute_count_error", "median"),
            mean_true_instances=("true_instances", "mean"),
            mean_pred_instances=("pred_instances", "mean"),
            no_prediction_rate=("pred_instances", lambda values: float((values == 0).mean())),
            median_latency_ms=("latency_ms", "median"),
            mean_latency_ms=("latency_ms", "mean"),
        )
        .round(4)
    )
    return summary.sort_values(["mean_object_f1", "mean_precision"], ascending=False)


def save_comparison_figure(summary: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    ordered = summary.sort_values("mean_object_f1", ascending=True).copy()
    labels = ordered["method_label"].map(FIGURE_LABELS).fillna(ordered["method_label"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].barh(labels, ordered["mean_object_f1"], color="#4f8a8b")
    axes[0].set_xlabel("Mean object F1")
    axes[0].set_xlim(0, 1)
    axes[0].set_title("Instance matching")
    for value, y_pos in zip(ordered["mean_object_f1"], range(len(ordered))):
        axes[0].text(min(value + 0.015, 0.98), y_pos, f"{value:.3f}", va="center", fontsize=6)
    axes[1].barh(labels, ordered["mean_absolute_count_error"], color="#a46a32")
    axes[1].set_xlabel("Mean absolute count error")
    axes[1].set_title("Count accuracy")
    x_max = max(ordered["mean_absolute_count_error"]) * 1.12
    axes[1].set_xlim(0, x_max)
    for value, y_pos in zip(ordered["mean_absolute_count_error"], range(len(ordered))):
        axes[1].text(value + x_max * 0.015, y_pos, f"{value:.2f}", va="center", fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    prepare_outputs(args.overwrite)

    manifest = pd.read_csv(VAL_MANIFEST)
    val_ids = set(manifest.loc[manifest["split"] == "val", "image_id"])
    yolo11n = pd.read_csv(YOLO11N_METRICS)
    yolo11m = pd.read_csv(YOLO11M_METRICS)
    zero_shot = pd.read_csv(ZERO_SHOT_METRICS)

    comparison = pd.concat(
        [
            normalize_yolo(yolo11n, "yolo11n_full_train_pool", "YOLO11n full train pool"),
            normalize_yolo(yolo11m, "yolo11m_full_train_pool", "YOLO11m full train pool"),
            normalize_zero_shot(zero_shot, val_ids),
        ],
        ignore_index=True,
    )
    expected = {
        "yolo11n_full_train_pool": len(val_ids),
        "yolo11m_full_train_pool": len(val_ids),
        "otsu_watershed": len(val_ids),
        "cellpose_cpsam": len(val_ids),
    }
    actual = comparison.groupby("method")["image_id"].nunique().to_dict()
    if actual != expected:
        raise RuntimeError(f"Unexpected comparison image coverage: {actual}")

    summary = build_summary(comparison)
    comparison.to_csv(COMPARISON_METRICS, index=False)
    summary.to_csv(COMPARISON_SUMMARY, index=False)
    save_comparison_figure(summary)
    print(f"Wrote {COMPARISON_METRICS}")
    print(f"Wrote {COMPARISON_SUMMARY}")
    print(f"Wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
