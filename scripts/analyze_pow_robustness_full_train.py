#!/usr/bin/env python
"""Analyze selected diagnostic metrics in the full-train robustness output."""

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


METHOD_ORDER = ["otsu_watershed", "cellpose_cpsam"]
METHOD_LABELS = {
    "otsu_watershed": "Otsu + watershed",
    "cellpose_cpsam": "Cellpose-SAM",
}
METHOD_COLORS = {
    "otsu_watershed": "#f97316",
    "cellpose_cpsam": "#0891b2",
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


def save_drop_distribution_plot(deltas: pd.DataFrame) -> None:
    non_clean = deltas[deltas["perturbation"] != "clean"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    for ax, method in zip(axes, METHOD_ORDER):
        values = [
            non_clean[
                (non_clean["method"] == method) & (non_clean["perturbation"] == perturbation)
            ]["absolute_object_f1_drop"].to_numpy()
            for perturbation in NON_CLEAN_PERTURBATIONS
        ]
        ax.boxplot(values, tick_labels=NON_CLEAN_PERTURBATIONS, showfliers=False)
        for index, series in enumerate(values, start=1):
            if len(series) == 0:
                continue
            jitter = ((np.arange(len(series)) % 9) - 4) * 0.018
            ax.scatter(
                index + jitter,
                series,
                s=8,
                alpha=0.22,
                color=METHOD_COLORS[method],
                linewidths=0,
            )
        ax.axhline(0, color="#111827", linewidth=1)
        ax.set_title(METHOD_LABELS[method])
        ax.tick_params(axis="x", rotation=25)

    axes[0].set_ylabel("Absolute object F1 drop from clean")
    fig.suptitle("Full-Train Robustness: Per-Image F1 Drop Distributions", y=0.98)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "robustness_pow_full_train_f1_drop_distributions.png", dpi=160)
    plt.close(fig)


def save_worst_drop_plot(failure_cases: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), sharey=True)

    for ax, method in zip(axes, METHOD_ORDER):
        top_cases = (
            failure_cases[failure_cases["method"] == method]
            .sort_values("absolute_object_f1_drop", ascending=False)
            .head(8)
            .copy()
        )
        top_cases["case_label"] = top_cases["perturbation"].astype(str) + "\n" + top_cases["image_id"].str[:8]
        ax.bar(range(len(top_cases)), top_cases["absolute_object_f1_drop"], color=METHOD_COLORS[method])
        ax.set_title(METHOD_LABELS[method])
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(len(top_cases)), labels=top_cases["case_label"], rotation=45, ha="right")
        ax.tick_params(axis="x", labelsize=8)

    axes[0].set_ylabel("Absolute object F1 drop from clean")
    fig.suptitle("Full-Train Robustness: Largest Per-Method F1 Drops", y=0.98)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "robustness_pow_full_train_worst_f1_drops.png", dpi=160)
    plt.close(fig)


def save_failure_hint_plot(failure_cases: pd.DataFrame) -> None:
    counts = (
        failure_cases.groupby(["method", "failure_hint"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    counts = counts.reindex(METHOD_ORDER).dropna(how="all").fillna(0)
    hint_order = [column for column in ["NO_PRED", "COLLAPSE", "FN+FP", "FN", "FP/OVER", "COUNT", "MIXED", "NO_DROP"] if column in counts.columns]
    counts = counts[hint_order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(counts))
    x = np.arange(len(counts))
    palette = {
        "NO_PRED": "#7f1d1d",
        "COLLAPSE": "#dc2626",
        "FN+FP": "#ea580c",
        "FN": "#f59e0b",
        "FP/OVER": "#2563eb",
        "COUNT": "#9333ea",
        "MIXED": "#64748b",
        "NO_DROP": "#16a34a",
    }
    for hint in hint_order:
        values = counts[hint].to_numpy()
        ax.bar(x, values, bottom=bottom, label=hint, color=palette.get(hint, "#6b7280"))
        bottom += values
    ax.set_xticks(x, labels=[METHOD_LABELS[method] for method in counts.index])
    ax.set_ylabel("Worst-case rows")
    ax.set_title("Full-Train Robustness: Worst-Case Failure Hints")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "robustness_pow_full_train_failure_hint_counts.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    metrics = load_metrics()
    deltas = add_clean_deltas(metrics)
    delta_output = save_deltas(deltas)
    failure_cases = save_failure_cases(deltas)
    no_prediction_cases = save_no_prediction_cases(deltas)
    save_drop_distribution_plot(deltas)
    save_worst_drop_plot(failure_cases)
    save_failure_hint_plot(failure_cases)

    print(f"Wrote {RESULT_SUBDIRS['robustness'] / 'pow_baseline_robustness_full_train_image_deltas.csv'}")
    print(f"Wrote {RESULT_SUBDIRS['robustness'] / 'pow_baseline_robustness_full_train_failure_cases.csv'}")
    print(f"Wrote {RESULT_SUBDIRS['robustness'] / 'pow_baseline_robustness_full_train_no_prediction_cases.csv'}")
    print(f"Wrote {FIGURES_DIR / 'robustness_pow_full_train_f1_drop_distributions.png'}")
    print(f"Wrote {FIGURES_DIR / 'robustness_pow_full_train_worst_f1_drops.png'}")
    print(f"Wrote {FIGURES_DIR / 'robustness_pow_full_train_failure_hint_counts.png'}")
    print(f"Analyzed {len(delta_output)} image-condition-method rows")
    print(f"Recorded {len(no_prediction_cases)} no-prediction rows")


if __name__ == "__main__":
    main()
