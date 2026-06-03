#!/usr/bin/env python
"""Analyze per-image failure patterns in the clean20 robustness output."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellseg-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cellseg_robustness.paths import FIGURES_DIR, RESULT_SUBDIRS, ensure_output_dirs


METHOD_ORDER = ["otsu_watershed", "cellpose_cpsam", "sam2_amg"]
METHOD_LABELS = {
    "otsu_watershed": "Otsu + watershed",
    "cellpose_cpsam": "Cellpose-SAM",
    "sam2_amg": "SAM2 AMG",
}
PERTURBATION_ORDER = [
    "clean",
    "gaussian_noise",
    "gaussian_blur",
    "downsample_upsample",
    "contrast_inversion",
]
NON_CLEAN_PERTURBATIONS = [name for name in PERTURBATION_ORDER if name != "clean"]


def load_metrics() -> pd.DataFrame:
    path = RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_clean20_metrics.csv"
    return pd.read_csv(path)


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
    deltas["method"] = pd.Categorical(deltas["method"], categories=METHOD_ORDER, ordered=True)
    deltas["perturbation"] = pd.Categorical(
        deltas["perturbation"], categories=PERTURBATION_ORDER, ordered=True
    )
    return deltas.sort_values(["method", "image_id", "perturbation"]).reset_index(drop=True)


def infer_failure_hint(row: pd.Series) -> str:
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
    ]
    output = deltas[output_columns].round(4)
    path = RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_clean20_image_deltas.csv"
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
            ).head(5)
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
        "failure_hint",
    ]
    output = failure_cases[output_columns].round(4)
    path = RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_clean20_failure_cases.csv"
    output.to_csv(path, index=False)
    return output


def save_drop_heatmap(deltas: pd.DataFrame) -> None:
    non_clean = deltas[deltas["perturbation"] != "clean"].copy()
    fig, axes = plt.subplots(3, 1, figsize=(13, 8.5), sharex=False, constrained_layout=True)

    for ax, method in zip(axes, METHOD_ORDER):
        method_frame = non_clean[non_clean["method"] == method]
        pivot = method_frame.pivot(
            index="perturbation",
            columns="image_id",
            values="absolute_object_f1_drop",
        )
        pivot = pivot.loc[NON_CLEAN_PERTURBATIONS]
        image_order = pivot.max(axis=0).sort_values(ascending=False).index
        pivot = pivot[image_order]

        image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=-0.1, vmax=1.0, cmap="magma")
        ax.set_title(f"{METHOD_LABELS[method]}: Object F1 Drop from Clean")
        ax.set_yticks(range(len(pivot.index)), labels=[str(value) for value in pivot.index])
        ax.set_xticks(range(len(pivot.columns)), labels=[image_id[:6] for image_id in pivot.columns])
        ax.tick_params(axis="x", rotation=90, labelsize=7)

    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, pad=0.015)
    colorbar.set_label("Absolute object F1 drop")
    fig.savefig(FIGURES_DIR / "robustness_pow_clean20_image_f1_drop_heatmap.png", dpi=160)
    plt.close(fig)


def save_worst_drop_plot(failure_cases: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
    colors = {"otsu_watershed": "#f97316", "cellpose_cpsam": "#0891b2", "sam2_amg": "#6366f1"}

    for ax, method in zip(axes, METHOD_ORDER):
        top_cases = (
            failure_cases[failure_cases["method"] == method]
            .sort_values("absolute_object_f1_drop", ascending=False)
            .head(5)
            .copy()
        )
        top_cases["case_label"] = top_cases["perturbation"].astype(str) + "\n" + top_cases["image_id"].str[:8]
        ax.bar(range(len(top_cases)), top_cases["absolute_object_f1_drop"], color=colors[method])
        ax.set_title(METHOD_LABELS[method])
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(len(top_cases)), labels=top_cases["case_label"], rotation=45, ha="right")
        ax.tick_params(axis="x", labelsize=8)

    axes[0].set_ylabel("Absolute object F1 drop from clean")
    fig.suptitle("PoW Clean20 Robustness: Largest Per-Method F1 Drops", y=0.98)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "robustness_pow_clean20_worst_f1_drops.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    metrics = load_metrics()
    deltas = add_clean_deltas(metrics)
    delta_output = save_deltas(deltas)
    failure_cases = save_failure_cases(deltas)
    save_drop_heatmap(deltas)
    save_worst_drop_plot(failure_cases)

    print(f"Wrote {RESULT_SUBDIRS['robustness'] / 'pow_baseline_robustness_clean20_image_deltas.csv'}")
    print(f"Wrote {RESULT_SUBDIRS['robustness'] / 'pow_baseline_robustness_clean20_failure_cases.csv'}")
    print(f"Wrote {FIGURES_DIR / 'robustness_pow_clean20_image_f1_drop_heatmap.png'}")
    print(f"Wrote {FIGURES_DIR / 'robustness_pow_clean20_worst_f1_drops.png'}")
    print(f"Analyzed {len(delta_output)} image-condition-method rows")


if __name__ == "__main__":
    main()
