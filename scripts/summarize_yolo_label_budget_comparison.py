#!/usr/bin/env python
"""Compare YOLO label-budget diagnostic results on the fixed held-out val split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cellseg_robustness.paths import RESULT_SUBDIRS, ensure_output_dirs


FIXED_YOLO_METRICS = RESULT_SUBDIRS["supervised"] / "yolo_fixed_budget_metrics.csv"
BUDGET_250_METRICS = RESULT_SUBDIRS["supervised"] / "yolo_label_budget_diagnostic_budget_250_metrics.csv"
VAL_MANIFEST = RESULT_SUBDIRS["supervised"] / "yolo_fixed_budget_manifest.csv"
ZERO_SHOT_METRICS = RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_full_train_metrics.csv"
COMPARISON_METRICS = RESULT_SUBDIRS["supervised"] / "yolo_label_budget_diagnostic_val_comparison_metrics.csv"
COMPARISON_SUMMARY = RESULT_SUBDIRS["supervised"] / "yolo_label_budget_diagnostic_val_comparison_summary.csv"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_outputs(overwrite: bool) -> None:
    existing = [path for path in [COMPARISON_METRICS, COMPARISON_SUMMARY] if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Existing label-budget comparison outputs found. Use --overwrite.\n{joined}")
    if overwrite:
        for path in existing:
            path.unlink()


def normalize_yolo(frame: pd.DataFrame, method: str, method_label: str, condition: str) -> pd.DataFrame:
    out = frame.copy()
    out["method"] = method
    out["method_label"] = method_label
    out["protocol"] = "supervised"
    out["condition"] = condition
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


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    prepare_outputs(args.overwrite)

    manifest = pd.read_csv(VAL_MANIFEST)
    val_ids = set(manifest.loc[manifest["split"] == "val", "image_id"])
    fixed = pd.read_csv(FIXED_YOLO_METRICS)
    budget_250 = pd.read_csv(BUDGET_250_METRICS)
    zero_shot = pd.read_csv(ZERO_SHOT_METRICS)

    comparison = pd.concat(
        [
            normalize_yolo(fixed, "yolo_fixed_budget_100", "YOLO fixed-budget 100", "held_out_val"),
            normalize_yolo(
                budget_250,
                "yolo_label_budget_250",
                "YOLO label-budget 250",
                "same_held_out_val",
            ),
            normalize_zero_shot(zero_shot, val_ids),
        ],
        ignore_index=True,
    )
    expected = {
        "yolo_fixed_budget_100": len(val_ids),
        "yolo_label_budget_250": len(val_ids),
        "otsu_watershed": len(val_ids),
        "cellpose_cpsam": len(val_ids),
    }
    actual = comparison.groupby("method")["image_id"].nunique().to_dict()
    if actual != expected:
        raise RuntimeError(f"Unexpected comparison image coverage: {actual}")

    summary = build_summary(comparison)
    comparison.to_csv(COMPARISON_METRICS, index=False)
    summary.to_csv(COMPARISON_SUMMARY, index=False)
    print(f"Wrote {COMPARISON_METRICS}")
    print(f"Wrote {COMPARISON_SUMMARY}")


if __name__ == "__main__":
    main()
