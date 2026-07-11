#!/usr/bin/env python
"""Analyze selected diagnostic metrics in the full-train robustness output."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cellseg_robustness.paths import RESULT_SUBDIRS, ensure_output_dirs

METHOD_ORDER = ["otsu_watershed", "cellpose_cpsam"]
METHOD_LABELS = {
    "otsu_watershed": "Otsu",
    "cellpose_cpsam": "Cellpose-SAM",
}
PERTURBATION_LABELS = {
    "clean": "Clean",
    "gaussian_noise": "Noise",
    "poisson_noise": "Poisson",
    "gaussian_blur": "Blur",
    "downsample_upsample": "Downsample",
    "intensity_scale": "Intensity",
    "contrast_inversion": "Inversion",
}
PERTURBATION_ORDER = list(PERTURBATION_LABELS)
NON_CLEAN_PERTURBATIONS = [name for name in PERTURBATION_ORDER if name != "clean"]


def load_metrics() -> pd.DataFrame:
    path = RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_full_train_metrics.csv"
    metrics = pd.read_csv(path)
    metrics = metrics[metrics["method"].isin(METHOD_ORDER)].copy()
    return metrics


def add_clean_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    clean = metrics[metrics["perturbation"] == "clean"][
        ["image_id", "method", "object_f1", "precision", "recall", "absolute_count_error"]
    ].rename(
        columns={
            "object_f1": "clean_object_f1",
            "precision": "clean_precision",
            "recall": "clean_recall",
            "absolute_count_error": "clean_absolute_count_error",
        }
    )
    deltas = metrics.merge(clean, on=["image_id", "method"], how="left", validate="many_to_one")
    deltas["absolute_object_f1_drop"] = deltas["clean_object_f1"] - deltas["object_f1"]
    deltas["relative_object_f1_drop"] = np.where(
        deltas["clean_object_f1"] > 0,
        deltas["absolute_object_f1_drop"] / deltas["clean_object_f1"],
        np.nan,
    )
    deltas["precision_drop"] = deltas["clean_precision"] - deltas["precision"]
    deltas["recall_drop"] = deltas["clean_recall"] - deltas["recall"]
    deltas["absolute_count_error_delta"] = (
        deltas["absolute_count_error"] - deltas["clean_absolute_count_error"]
    )
    deltas["no_prediction"] = deltas["pred_instances"] == 0
    deltas["method"] = pd.Categorical(deltas["method"], categories=METHOD_ORDER, ordered=True)
    deltas["perturbation"] = pd.Categorical(
        deltas["perturbation"], categories=PERTURBATION_ORDER, ordered=True
    )
    return deltas.sort_values(["method", "image_id", "perturbation"]).reset_index(drop=True)


def infer_failure_hint(row: pd.Series) -> str:
    if row["no_prediction"] and row["true_instances"] > 0:
        return "NO_PRED"
    if row["object_f1"] == 0 and row["clean_object_f1"] > 0:
        return "COLLAPSE"
    if row["recall_drop"] > 0.25 and row["precision_drop"] > 0.25:
        return "FN+FP"
    if row["recall_drop"] > 0.25:
        return "FN"
    if row["precision_drop"] > 0.25:
        return "FP/OVER"
    if row["absolute_count_error_delta"] > 25:
        return "COUNT"
    if row["absolute_object_f1_drop"] <= 0:
        return "NO_DROP"
    return "MIXED"


def save_deltas(deltas: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "split",
        "image_id",
        "method",
        "method_label",
        "perturbation",
        "clean_object_f1",
        "object_f1",
        "absolute_object_f1_drop",
        "relative_object_f1_drop",
        "clean_precision",
        "precision",
        "precision_drop",
        "clean_recall",
        "recall",
        "recall_drop",
        "clean_absolute_count_error",
        "absolute_count_error",
        "absolute_count_error_delta",
        "true_instances",
        "pred_instances",
        "false_positives",
        "false_negatives",
        "no_prediction",
    ]
    output = deltas[output_columns].round(4)
    path = RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_full_train_image_deltas.csv"
    output.to_csv(path, index=False)
    return output


def save_failure_cases(deltas: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    non_clean = deltas[deltas["perturbation"] != "clean"].copy()
    for method in METHOD_ORDER:
        for perturbation in NON_CLEAN_PERTURBATIONS:
            subset = non_clean[
                (non_clean["method"] == method) & (non_clean["perturbation"] == perturbation)
            ].copy()
            subset = subset.sort_values(
                ["absolute_object_f1_drop", "absolute_count_error_delta"],
                ascending=[False, False],
            ).head(10)
            rows.append(subset)

    failure_cases = pd.concat(rows, ignore_index=True)
    failure_cases["failure_hint"] = failure_cases.apply(infer_failure_hint, axis=1)
    output_columns = [
        "method",
        "method_label",
        "perturbation",
        "image_id",
        "clean_object_f1",
        "object_f1",
        "absolute_object_f1_drop",
        "relative_object_f1_drop",
        "precision_drop",
        "recall_drop",
        "absolute_count_error_delta",
        "true_instances",
        "pred_instances",
        "false_positives",
        "false_negatives",
        "no_prediction",
        "failure_hint",
    ]
    output = failure_cases[output_columns].round(4)
    path = RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_full_train_failure_cases.csv"
    output.to_csv(path, index=False)
    return output


def save_no_prediction_cases(deltas: pd.DataFrame) -> pd.DataFrame:
    cases = deltas[deltas["no_prediction"]].copy()
    output_columns = [
        "method",
        "method_label",
        "perturbation",
        "image_id",
        "clean_object_f1",
        "object_f1",
        "true_instances",
        "pred_instances",
        "false_positives",
        "false_negatives",
        "absolute_count_error",
        "no_prediction",
    ]
    output = cases[output_columns].round(4)
    path = RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_full_train_no_prediction_cases.csv"
    output.to_csv(path, index=False)
    return output


def main() -> None:
    ensure_output_dirs()
    metrics = load_metrics()
    deltas = add_clean_deltas(metrics)
    delta_output = save_deltas(deltas)
    failure_cases = save_failure_cases(deltas)
    no_prediction_cases = save_no_prediction_cases(deltas)

    print(f"Wrote {RESULT_SUBDIRS['robustness'] / 'pow_baseline_robustness_full_train_image_deltas.csv'}")
    print(f"Wrote {RESULT_SUBDIRS['robustness'] / 'pow_baseline_robustness_full_train_failure_cases.csv'}")
    print(f"Wrote {RESULT_SUBDIRS['robustness'] / 'pow_baseline_robustness_full_train_no_prediction_cases.csv'}")
    print(f"Analyzed {len(delta_output)} image-condition-method rows")
    print(f"Recorded {len(no_prediction_cases)} no-prediction rows")


if __name__ == "__main__":
    main()
