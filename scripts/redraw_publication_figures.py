#!/usr/bin/env python
"""Redraw summary figures from existing CSV outputs with shared plot styling.

This script is the single independent entry point for re-rendering all summary
statistical figures without re-running any model inference.  It reads the CSV
outputs produced by the analysis scripts and overwrites the corresponding PNG
files in figures/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellseg-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cellseg_robustness.paths import FIGURES_DIR, RESULT_SUBDIRS, ensure_output_dirs
from cellseg_robustness.plot_style import DROP_CMAP, METHOD_PALETTE, METRIC_PALETTE, SCORE_CMAP, gradient_colors, save_png


METHOD_ORDER = ["otsu_watershed", "cellpose_cpsam", "sam2_amg"]
METHOD_LABELS = {
    "otsu_watershed": "Otsu",
    "cellpose_cpsam": "Cellpose-SAM",
    "sam2_amg": "SAM2 AMG",
}
BASELINE_METHOD_LABELS = {
    "otsu_watershed": "Otsu + watershed",
    "cellpose_cpsam": "Cellpose-SAM",
    "sam2_amg": "SAM2 AMG",
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
METRIC_LABELS = {
    "mean_object_f1": "Object F1",
    "mean_matched_iou": "Matched IoU",
    "mean_matched_dice": "Matched Dice",
}


def method_color(method: str) -> str:
    return METHOD_PALETTE.get(method, METHOD_PALETTE["neutral"])


def _short_config_labels(config_ids: list) -> dict:
    """Map long config identifiers to short, publication-friendly labels."""
    return {cid: f"Config {i+1}" for i, cid in enumerate(config_ids)}


def redraw_otsu_smoke() -> None:
    summary = pd.read_csv(RESULT_SUBDIRS["robustness"] / "otsu_watershed_perturbation_smoke_summary.csv")
    summary = summary[summary["perturbation"].isin(PERTURBATION_ORDER)].copy()
    summary["label"] = summary["perturbation"].map(PERTURBATION_LABELS)

    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.bar(summary["label"], summary["mean_object_f1"], color=gradient_colors(len(summary)), edgecolor="white")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Mean object F1")
    ax.set_title("Otsu robustness across perturbations")
    ax.spines["bottom"].set_color("#4b5563")
    ax.spines["left"].set_color("#4b5563")
    save_png(fig, FIGURES_DIR / "robustness_otsu_smoke_mean_scores.png")
    plt.close(fig)

    non_clean = summary[summary["perturbation"] != "clean"]
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.bar(non_clean["label"], non_clean["relative_object_f1_drop"], color=gradient_colors(len(non_clean)), edgecolor="white")
    ax.axhline(0, color="#4b5563", linewidth=0.8)
    ax.set_ylabel("Relative F1 drop")
    ax.set_title("Otsu performance loss from clean")
    ax.spines["bottom"].set_color("#4b5563")
    ax.spines["left"].set_color("#4b5563")
    save_png(fig, FIGURES_DIR / "robustness_otsu_smoke_relative_f1_drop.png")
    plt.close(fig)


def redraw_robustness_summary(summary_path: Path, figure_prefix: str) -> None:
    summary = pd.read_csv(summary_path)
    summary = summary[summary["perturbation"].isin(PERTURBATION_ORDER)].copy()
    methods = [method for method in METHOD_ORDER if method in set(summary["method"])]

    pivot = (
        summary.pivot(index="perturbation", columns="method", values="mean_object_f1")
        .reindex(PERTURBATION_ORDER)
        .dropna(how="all")
    )
    pivot = pivot[[method for method in methods if method in pivot.columns]]
    labels = [PERTURBATION_LABELS[item] for item in pivot.index]
    x = np.arange(len(pivot))
    n_methods = len(pivot.columns)
    width = 0.72 / max(n_methods, 1)

    # Hero panel: grouped bar chart of mean F1
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for index, method in enumerate(pivot.columns):
        offset = (index - (n_methods - 1) / 2) * width
        values = pivot[method].to_numpy()
        ax.bar(
            x + offset,
            values,
            width=width,
            label=METHOD_LABELS.get(method, method),
            color=method_color(method),
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_xticks(x, labels=labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Mean object F1")
    ax.set_title("Robustness to image perturbations")
    ax.legend(loc="upper right", ncols=min(n_methods, 3))
    ax.spines["bottom"].set_color("#4b5563")
    ax.spines["left"].set_color("#4b5563")
    save_png(fig, FIGURES_DIR / f"{figure_prefix}_mean_f1.png")
    plt.close(fig)

    # Supporting panel: relative F1 drop
    drop = (
        summary[summary["perturbation"] != "clean"]
        .pivot(index="perturbation", columns="method", values="relative_object_f1_drop")
        .reindex([item for item in PERTURBATION_ORDER if item != "clean"])
    )
    drop = drop[[method for method in methods if method in drop.columns]]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    x = np.arange(len(drop))
    n_methods = len(drop.columns)
    width = 0.72 / max(n_methods, 1)
    for index, method in enumerate(drop.columns):
        offset = (index - (n_methods - 1) / 2) * width
        ax.bar(
            x + offset,
            drop[method].to_numpy(),
            width=width,
            label=METHOD_LABELS.get(method, method),
            color=method_color(method),
            edgecolor="white",
            linewidth=0.5,
        )
    ax.axhline(0, color="#4b5563", linewidth=0.8)
    ax.set_xticks(x, labels=[PERTURBATION_LABELS[item] for item in drop.index])
    ax.set_ylabel("Relative F1 drop")
    ax.set_title("Performance loss from clean")
    ax.legend(loc="upper left", ncols=min(n_methods, 3))
    ax.spines["bottom"].set_color("#4b5563")
    ax.spines["left"].set_color("#4b5563")
    save_png(fig, FIGURES_DIR / f"{figure_prefix}_relative_f1_drop.png")
    plt.close(fig)

    # Supporting panel: condition heatmap (sequential blue)
    fig, ax = plt.subplots(figsize=(6.0, 2.4 + 0.5 * n_methods))
    matrix = pivot.T.to_numpy()
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap=SCORE_CMAP, aspect="auto")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(pivot.columns)), labels=[METHOD_LABELS.get(method, method) for method in pivot.columns])
    ax.set_title("Mean object F1 by condition")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="white" if value >= 0.65 else "#111827", fontsize=7)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("F1")
    save_png(fig, FIGURES_DIR / f"{figure_prefix}_method_condition_heatmap.png")
    plt.close(fig)


def redraw_yolo_threshold() -> None:
    summary = pd.read_csv(RESULT_SUBDIRS["supervised"] / "yolo_threshold_diagnostic_summary.csv")
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(summary["conf"], summary["mean_object_f1"], marker="o", color=METHOD_PALETTE["yolo"], linewidth=1.5)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Mean object F1")
    ax.set_ylim(0, 1)
    ax.set_title("YOLO F1 by confidence threshold")
    ax.spines["bottom"].set_color("#4b5563")
    ax.spines["left"].set_color("#4b5563")
    save_png(fig, FIGURES_DIR / "supervised_yolo_threshold_diagnostic_f1.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(summary["conf"], summary["mean_absolute_count_error"], marker="o", color=METHOD_PALETTE["warning"], linewidth=1.5)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Mean absolute count error")
    ax.set_title("YOLO count error by confidence threshold")
    ax.spines["bottom"].set_color("#4b5563")
    ax.spines["left"].set_color("#4b5563")
    save_png(fig, FIGURES_DIR / "supervised_yolo_threshold_diagnostic_count_error.png")
    plt.close(fig)


def redraw_sam2_sensitivity() -> None:
    clean = pd.read_csv(RESULT_SUBDIRS["robustness"] / "sam2_amg_sensitivity_clean20_clean_screen_summary.csv")
    validation = pd.read_csv(RESULT_SUBDIRS["robustness"] / "sam2_amg_sensitivity_clean20_validation_summary.csv")
    clean = clean.sort_values("mean_object_f1", ascending=True)
    label_map = _short_config_labels(clean["config_id"].tolist())
    clean["label"] = clean["config_id"].map(label_map)

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.barh(clean["label"], clean["mean_object_f1"], color=gradient_colors(len(clean)))
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Mean object F1")
    ax.set_title("SAM2 AMG clean-screen sensitivity")
    ax.spines["bottom"].set_color("#4b5563")
    ax.spines["left"].set_color("#4b5563")
    save_png(fig, FIGURES_DIR / "robustness_sam2_amg_sensitivity_clean20_clean_screen_f1.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.barh(clean["label"], clean["mean_pred_instances"], color=gradient_colors(len(clean)))
    ax.axvline(clean["mean_true_instances"].mean(), color="#4b5563", linewidth=0.8, linestyle="--", label="Mean true count")
    ax.set_xlabel("Mean predicted instances")
    ax.set_title("SAM2 AMG clean-screen counts")
    ax.legend(loc="lower right")
    ax.spines["bottom"].set_color("#4b5563")
    ax.spines["left"].set_color("#4b5563")
    save_png(fig, FIGURES_DIR / "robustness_sam2_amg_sensitivity_clean20_clean_screen_counts.png")
    plt.close(fig)

    clean_rows = validation[validation["perturbation"] == "clean"].sort_values("mean_object_f1", ascending=False)
    top_configs = clean_rows["config_id"].head(6).tolist()
    frame = validation[validation["config_id"].isin(top_configs)].copy()
    label_map = _short_config_labels(top_configs)
    frame["label"] = frame["config_id"].map(label_map)
    pivot = frame.pivot(index="label", columns="perturbation", values="mean_object_f1")
    pivot = pivot[[item for item in PERTURBATION_ORDER if item in pivot.columns]]
    pivot = pivot.sort_values("clean", ascending=True)

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    image = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap=SCORE_CMAP, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), labels=[PERTURBATION_LABELS[item] for item in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_title("SAM2 AMG sensitivity: mean object F1")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("F1")
    save_png(fig, FIGURES_DIR / "robustness_sam2_amg_sensitivity_clean20_mean_f1.png")
    plt.close(fig)

    collapsed = frame.groupby("config_id", as_index=False)["zero_prediction_rate"].mean().sort_values("zero_prediction_rate")
    collapsed["label"] = collapsed["config_id"].map(label_map)
    collapsed = collapsed.sort_values("zero_prediction_rate")
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.barh(collapsed["label"], collapsed["zero_prediction_rate"], color=gradient_colors(len(collapsed)))
    ax.set_xlabel("Mean zero-prediction rate")
    ax.set_title("SAM2 AMG zero-prediction rate")
    ax.spines["bottom"].set_color("#4b5563")
    ax.spines["left"].set_color("#4b5563")
    save_png(fig, FIGURES_DIR / "robustness_sam2_amg_sensitivity_clean20_zero_pred_rate.png")
    plt.close(fig)

    count_error = frame.groupby("config_id", as_index=False)["mean_absolute_count_error"].mean().sort_values("mean_absolute_count_error")
    count_error["label"] = count_error["config_id"].map(label_map)
    count_error = count_error.sort_values("mean_absolute_count_error")
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.barh(count_error["label"], count_error["mean_absolute_count_error"], color=gradient_colors(len(count_error)))
    ax.set_xlabel("Mean absolute count error")
    ax.set_title("SAM2 AMG count error")
    ax.spines["bottom"].set_color("#4b5563")
    ax.spines["left"].set_color("#4b5563")
    save_png(fig, FIGURES_DIR / "robustness_sam2_amg_sensitivity_clean20_count_error.png")
    plt.close(fig)


def redraw_cellpose_parameter_diagnostic() -> None:
    summary = pd.read_csv(RESULT_SUBDIRS["baselines"] / "cellpose_cpsam_parameter_diagnostic_summary.csv")
    short_labels = {
        "rgb_baseline": "RGB default",
        "gray_mean": "Gray mean",
        "gray_max": "Gray max",
        "rgb_invert": "RGB invert",
        "rgb_cellprob_-1": "Cellprob -1",
        "rgb_cellprob_-2": "Cellprob -2",
        "rgb_flow_0": "Flow 0",
        "rgb_diameter_15": "Diameter 15",
        "gray_diameter_15": "Gray + diameter 15",
        "rgb_diameter_30": "Diameter 30",
        "rgb_diameter_60": "Diameter 60",
    }
    summary["label"] = summary["config_id"].map(short_labels).fillna(summary["config_id"])
    ordered = summary.sort_values("mean_object_f1", ascending=True)

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.barh(ordered["label"], ordered["mean_object_f1"], color=gradient_colors(len(ordered)))
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean object F1")
    ax.set_title("Cellpose-SAM parameter diagnostic")
    ax.spines["bottom"].set_color("#4b5563")
    ax.spines["left"].set_color("#4b5563")
    save_png(fig, FIGURES_DIR / "cellpose_cpsam_parameter_diagnostic_f1.png")
    plt.close(fig)

    heldout_path = RESULT_SUBDIRS["baselines"] / "cellpose_cpsam_parameter_diagnostic_heldout_val_summary.csv"
    if heldout_path.exists():
        heldout = pd.read_csv(heldout_path)
        heldout["label"] = heldout["config_id"].map(short_labels).fillna(heldout["config_id"])
        ordered = heldout.sort_values("mean_object_f1", ascending=True)
        fig, ax = plt.subplots(figsize=(6.8, 3.6))
        ax.barh(ordered["label"], ordered["mean_object_f1"], color=gradient_colors(len(ordered)))
        ax.set_xlim(0, 1)
        ax.set_xlabel("Mean object F1")
        ax.set_title("Cellpose-SAM input-mode diagnostic")
        ax.spines["bottom"].set_color("#4b5563")
        ax.spines["left"].set_color("#4b5563")
        save_png(fig, FIGURES_DIR / "cellpose_cpsam_input_mode_lock_heldout_val_f1.png")
        plt.close(fig)


def redraw_baseline_clean_subset() -> None:
    """Redraw baseline clean-subset summary figures from existing CSVs."""
    baseline_files = {
        "otsu_watershed": "otsu_watershed_clean_subset_metrics.csv",
        "cellpose_cpsam": "cellpose_cpsam_clean_subset_metrics.csv",
        "sam2_amg": "sam2_amg_clean_subset_metrics.csv",
    }
    frames: list[pd.DataFrame] = []
    for method, filename in baseline_files.items():
        path = RESULT_SUBDIRS["baselines"] / filename
        frame = pd.read_csv(path)
        frame["method"] = method
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    metrics["method_label"] = metrics["method"].map(BASELINE_METHOD_LABELS)

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
    summary["method_label"] = summary["method"].map(BASELINE_METHOD_LABELS)
    method_rank = {method: i for i, method in enumerate(METHOD_ORDER)}
    summary["method_order"] = summary["method"].map(method_rank)
    summary = summary.sort_values("method_order").drop(columns="method_order")

    method_colors = {method: METHOD_PALETTE[method] for method in METHOD_ORDER}

    # Grouped metric comparison
    metric_cols = ["mean_object_f1", "mean_matched_iou", "mean_matched_dice"]
    plot_frame = summary.set_index("method_label")[metric_cols]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = range(len(plot_frame))
    n_metrics = len(metric_cols)
    width = 0.72 / n_metrics
    metric_colors = [METRIC_PALETTE["object_f1"], METRIC_PALETTE["mean_matched_iou"], METRIC_PALETTE["mean_matched_dice"]]
    for i, metric in enumerate(metric_cols):
        offset = (i - (n_metrics - 1) / 2) * width
        values = plot_frame[metric].to_numpy()
        ax.bar(
            [xi + offset for xi in x],
            values,
            width=width,
            label=METRIC_LABELS[metric],
            color=metric_colors[i],
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_xticks(x, labels=plot_frame.index)
    ax.set_ylim(0, 1.12)
    ax.set_title("Clean subset baseline metrics")
    ax.set_xlabel("Method")
    ax.set_ylabel("Mean score")
    ax.legend(frameon=False, ncols=3, loc="upper right")
    ax.spines["bottom"].set_color("#4b5563")
    ax.spines["left"].set_color("#4b5563")
    save_png(fig, FIGURES_DIR / "baseline_clean_subset_metric_comparison.png")
    plt.close(fig)

    # Count error
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    colors = [method_colors[method] for method in summary["method"]]
    ax.bar(summary["method_label"], summary["mean_absolute_count_error"], color=colors, width=0.62, edgecolor="white")
    ax.set_title("Clean subset mean absolute count error")
    ax.set_xlabel("Method")
    ax.set_ylabel("Mean absolute count error")
    ax.spines["bottom"].set_color("#4b5563")
    ax.spines["left"].set_color("#4b5563")
    save_png(fig, FIGURES_DIR / "baseline_clean_subset_count_error_comparison.png")
    plt.close(fig)

    # Latency
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(summary["method_label"], summary["median_latency_ms"], color=colors, width=0.62, edgecolor="white")
    ax.set_title("Clean subset median latency")
    ax.set_xlabel("Method")
    ax.set_ylabel("Median latency (ms/image)")
    ax.spines["bottom"].set_color("#4b5563")
    ax.spines["left"].set_color("#4b5563")
    save_png(fig, FIGURES_DIR / "baseline_clean_subset_latency_comparison.png")
    plt.close(fig)

    # Score distributions
    metric_specs = [
        ("object_f1", "Object F1", (0, 1)),
        ("mean_matched_iou", "Matched IoU", (0, 1)),
        ("precision", "Precision", (0, 1)),
        ("recall", "Recall", (0, 1)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)
    axes_array = axes.ravel()
    for ax, (column, title, ylim) in zip(axes_array, metric_specs):
        data = [metrics.loc[metrics["method"] == method, column].to_numpy() for method in METHOD_ORDER]
        box = ax.boxplot(
            data,
            tick_labels=[BASELINE_METHOD_LABELS[method] for method in METHOD_ORDER],
            showfliers=False,
            patch_artist=True,
        )
        for patch, method in zip(box["boxes"], METHOD_ORDER):
            patch.set_facecolor(method_colors[method])
            patch.set_alpha(0.28)
            patch.set_edgecolor(method_colors[method])
        for median in box["medians"]:
            median.set_color("#111827")
        for index, values in enumerate(data, start=1):
            jitter = ((pd.Series(range(len(values))) % 5).to_numpy() - 2) * 0.025
            ax.scatter(
                index + jitter,
                values,
                s=12,
                alpha=0.5,
                color=method_colors[METHOD_ORDER[index - 1]],
            )
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.spines["bottom"].set_color("#4b5563")
        ax.spines["left"].set_color("#4b5563")
    axes_array[0].set_ylabel("Per-image score")
    handles = [
        Patch(
            facecolor=method_colors[method],
            edgecolor=method_colors[method],
            alpha=0.45,
            label=BASELINE_METHOD_LABELS[method],
        )
        for method in METHOD_ORDER
    ]
    fig.legend(handles=handles, frameon=False, ncols=3, loc="lower center", bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Clean subset per-image score distributions", y=1.02)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save_png(fig, FIGURES_DIR / "baseline_clean_subset_score_distributions.png")
    plt.close(fig)

    # Precision-recall scatter
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    for method in METHOD_ORDER:
        method_metrics = metrics[metrics["method"] == method]
        ax.scatter(
            method_metrics["recall"],
            method_metrics["precision"],
            s=28,
            alpha=0.75,
            color=method_colors[method],
            label=BASELINE_METHOD_LABELS[method],
        )
    ax.plot([0, 1], [0, 1], color="#111827", linewidth=1, linestyle="--")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("Clean subset precision-recall by image")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=False, loc="upper left")
    ax.spines["bottom"].set_color("#4b5563")
    ax.spines["left"].set_color("#4b5563")
    fig.tight_layout()
    save_png(fig, FIGURES_DIR / "baseline_clean_subset_precision_recall.png")
    plt.close(fig)

    # Image-method heatmap
    pivot = metrics.pivot(index="image_id", columns="method", values="object_f1")
    pivot = pivot[METHOD_ORDER]
    pivot = pivot.sort_values("cellpose_cpsam", ascending=True)
    rank_labels = [str(index) for index in range(1, len(pivot) + 1)]
    fig, ax = plt.subplots(figsize=(5.8, 6.5))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap=SCORE_CMAP)
    ax.set_xticks(range(len(METHOD_ORDER)), labels=[BASELINE_METHOD_LABELS[method] for method in METHOD_ORDER])
    if len(rank_labels) <= 25:
        tick_positions = list(range(len(rank_labels)))
    else:
        tick_positions = list(range(0, len(rank_labels), 5))
    ax.set_yticks(tick_positions, labels=[rank_labels[index] for index in tick_positions])
    ax.set_title("Clean subset object F1 by image and method")
    ax.set_xlabel("Method")
    ax.set_ylabel("Image rank (sorted by Cellpose-SAM F1)")
    ax.tick_params(axis="x", rotation=0)
    if pivot.shape[0] <= 30:
        for row_index in range(pivot.shape[0]):
            for column_index in range(pivot.shape[1]):
                value = pivot.iat[row_index, column_index]
                text_color = "white" if value >= 0.65 else "#111827"
                ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=6)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Object F1")
    fig.tight_layout()
    save_png(fig, FIGURES_DIR / "baseline_clean_subset_image_method_f1_heatmap.png")
    plt.close(fig)


def redraw_dataset_audit() -> None:
    inventory = pd.read_csv(RESULT_SUBDIRS["dataset"] / "dataset_inventory.csv")
    summary = pd.read_csv(RESULT_SUBDIRS["dataset"] / "dataset_summary.csv")
    split_labels = {
        "stage1_train": "Train",
        "stage1_test": "Test stage 1",
        "stage2_test_final": "Test stage 2",
    }

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    labels = [split_labels.get(split, split) for split in summary["split"]]
    bars = ax.bar(labels, summary["image_count"], color=gradient_colors(len(summary)), width=0.62, edgecolor="white")
    for bar, value in zip(bars, summary["image_count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(value)), ha="center", va="bottom", fontsize=7)
    ax.set_title("DSB2018 image counts by split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Images")
    save_png(fig, FIGURES_DIR / "dataset_split_counts.png")
    plt.close(fig)

    train = inventory[inventory["split"] == "stage1_train"]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    _, _, patches = ax.hist(train["mask_count"], bins=30, edgecolor="white")
    for patch, color in zip(patches, gradient_colors(len(patches))):
        patch.set_facecolor(color)
    ax.set_title("DSB2018 Stage 1 Train Instance Counts")
    ax.set_xlabel("PNG instance masks per image")
    ax.set_ylabel("Images")
    save_png(fig, FIGURES_DIR / "dataset_train_instance_count_hist.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    split_colors = {
        "stage1_train": METHOD_PALETTE["cellpose_cpsam"],
        "stage1_test": METHOD_PALETTE["otsu_watershed"],
        "stage2_test_final": METHOD_PALETTE["sam2_amg"],
    }
    for split, frame in inventory.groupby("split"):
        ax.scatter(
            frame["width"],
            frame["height"],
            s=16,
            alpha=0.62,
            color=split_colors.get(split, METHOD_PALETTE["neutral"]),
            label=split_labels.get(split, split),
            linewidths=0,
        )
    ax.set_title("DSB2018 Image Size Distribution")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    ax.legend(frameon=False)
    save_png(fig, FIGURES_DIR / "dataset_image_size_scatter.png")
    plt.close(fig)


def redraw_cellpose_method_availability() -> None:
    audit = pd.read_csv(RESULT_SUBDIRS["baselines"] / "cellpose_method_availability.csv")
    frame = audit.copy()
    frame["available"] = frame["runnable_as_distinct_method"].astype(int)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(frame["method"], frame["available"], color=gradient_colors(len(frame)), edgecolor="white")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Runnable as distinct method")
    ax.set_title("Cellpose-family Method Availability")
    ax.set_yticks([0, 1], labels=["no", "yes"])
    ax.tick_params(axis="x", rotation=0, labelsize=6)
    save_png(fig, FIGURES_DIR / "cellpose_method_availability.png")
    plt.close(fig)


def redraw_individual_clean_subset_baselines() -> None:
    specs = {
        "otsu_watershed": ("otsu_watershed_clean_subset_metrics.csv", "Otsu + watershed", "otsu_watershed_subset"),
        "cellpose_cpsam": ("cellpose_cpsam_clean_subset_metrics.csv", "Cellpose-SAM", "cellpose_cpsam_subset"),
        "sam2_amg": ("sam2_amg_clean_subset_metrics.csv", "SAM2 AMG", "sam2_amg_subset"),
    }
    metric_columns = ["object_f1", "mean_matched_iou", "mean_matched_dice"]
    metric_labels = ["Object F1", "Matched IoU", "Matched Dice"]
    metric_colors = [METRIC_PALETTE["object_f1"], METRIC_PALETTE["mean_matched_iou"], METRIC_PALETTE["mean_matched_dice"]]

    for method, (filename, label, prefix) in specs.items():
        metrics = pd.read_csv(RESULT_SUBDIRS["baselines"] / filename)
        summary = metrics[metric_columns].mean()

        fig, ax = plt.subplots(figsize=(4.8, 3.5))
        ax.bar(metric_labels, summary.values, color=metric_colors, edgecolor="white")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{label}: subset mean metrics")
        ax.set_ylabel("Score")
        save_png(fig, FIGURES_DIR / f"{prefix}_metric_means.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        ax.scatter(
            metrics["true_instances"],
            metrics["pred_instances"],
            alpha=0.75,
            color=METHOD_PALETTE.get(method, METHOD_PALETTE["neutral"]),
            s=28,
            linewidths=0,
        )
        max_count = int(max(metrics["true_instances"].max(), metrics["pred_instances"].max()))
        ax.plot([0, max_count], [0, max_count], color="#111827", linewidth=1, linestyle="--")
        ax.set_title(f"{label}: count agreement")
        ax.set_xlabel("Ground-truth instances")
        ax.set_ylabel("Predicted instances")
        save_png(fig, FIGURES_DIR / f"{prefix}_count_scatter.png")
        plt.close(fig)


def redraw_clean20_diagnostics() -> None:
    deltas = pd.read_csv(RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_clean20_image_deltas.csv")
    failure_cases = pd.read_csv(RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_clean20_failure_cases.csv")
    non_clean = deltas[deltas["perturbation"] != "clean"].copy()
    methods = [method for method in METHOD_ORDER if method in set(non_clean["method"])]

    fig, axes = plt.subplots(len(methods), 1, figsize=(12, 2.8 * len(methods)), sharex=False, constrained_layout=True)
    axes_array = np.atleast_1d(axes)
    image = None
    for ax, method in zip(axes_array, methods):
        frame = non_clean[non_clean["method"] == method]
        pivot = frame.pivot(index="perturbation", columns="image_id", values="absolute_object_f1_drop")
        pivot = pivot.reindex([name for name in PERTURBATION_ORDER if name != "clean"])
        image_order = pivot.max(axis=0).sort_values(ascending=False).index
        pivot = pivot[image_order]
        image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=-0.1, vmax=1.0, cmap=DROP_CMAP)
        ax.set_title(f"{METHOD_LABELS[method]}: object F1 drop from clean")
        ax.set_yticks(range(len(pivot.index)), labels=[PERTURBATION_LABELS[value] for value in pivot.index])
        ax.set_xticks(range(len(pivot.columns)), labels=[str(index) for index in range(1, len(pivot.columns) + 1)])
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes_array.ravel().tolist(), shrink=0.82, pad=0.015)
        colorbar.set_label("Absolute object F1 drop")
    save_png(fig, FIGURES_DIR / "robustness_pow_clean20_image_f1_drop_heatmap.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(methods), figsize=(4.6 * len(methods), 4.6), sharey=True)
    axes_array = np.atleast_1d(axes)
    for ax, method in zip(axes_array, methods):
        top_cases = (
            failure_cases[failure_cases["method"] == method]
            .sort_values("absolute_object_f1_drop", ascending=False)
            .head(5)
            .copy()
        )
        labels = [f"Case {index}" for index in range(1, len(top_cases) + 1)]
        ax.bar(labels, top_cases["absolute_object_f1_drop"], color=gradient_colors(len(top_cases)), edgecolor="white")
        ax.set_title(METHOD_LABELS[method])
        ax.set_ylim(0, 1.05)
    axes_array[0].set_ylabel("Absolute object F1 drop from clean")
    fig.suptitle("Clean20 robustness: largest per-method F1 drops", y=1.02)
    fig.tight_layout()
    save_png(fig, FIGURES_DIR / "robustness_pow_clean20_worst_f1_drops.png")
    plt.close(fig)


def redraw_full_train_diagnostics() -> None:
    deltas = pd.read_csv(RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_full_train_image_deltas.csv")
    failure_cases = pd.read_csv(RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_full_train_failure_cases.csv")
    methods = [method for method in ["otsu_watershed", "cellpose_cpsam"] if method in set(deltas["method"])]
    non_clean = deltas[deltas["perturbation"] != "clean"].copy()
    non_clean_order = [name for name in PERTURBATION_ORDER if name != "clean"]

    fig, axes = plt.subplots(1, len(methods), figsize=(6.0 * len(methods), 4.3), sharey=True)
    axes_array = np.atleast_1d(axes)
    for ax, method in zip(axes_array, methods):
        values = [
            non_clean[
                (non_clean["method"] == method) & (non_clean["perturbation"] == perturbation)
            ]["absolute_object_f1_drop"].to_numpy()
            for perturbation in non_clean_order
        ]
        ax.boxplot(values, tick_labels=[PERTURBATION_LABELS[label] for label in non_clean_order], showfliers=False)
        for index, series in enumerate(values, start=1):
            if len(series) == 0:
                continue
            jitter = ((np.arange(len(series)) % 9) - 4) * 0.018
            ax.scatter(index + jitter, series, s=8, alpha=0.22, color=METHOD_PALETTE[method], linewidths=0)
        ax.axhline(0, color="#111827", linewidth=1)
        ax.set_title(METHOD_LABELS[method])
    axes_array[0].set_ylabel("Absolute object F1 drop from clean")
    fig.suptitle("Full-train robustness: per-image F1 drop distributions", y=1.02)
    fig.tight_layout()
    save_png(fig, FIGURES_DIR / "robustness_pow_full_train_f1_drop_distributions.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(methods), figsize=(5.2 * len(methods), 4.6), sharey=True)
    axes_array = np.atleast_1d(axes)
    for ax, method in zip(axes_array, methods):
        top_cases = (
            failure_cases[failure_cases["method"] == method]
            .sort_values("absolute_object_f1_drop", ascending=False)
            .head(8)
            .copy()
        )
        labels = [f"Case {index}" for index in range(1, len(top_cases) + 1)]
        ax.bar(labels, top_cases["absolute_object_f1_drop"], color=gradient_colors(len(top_cases)), edgecolor="white")
        ax.set_title(METHOD_LABELS[method])
        ax.set_ylim(0, 1.05)
    axes_array[0].set_ylabel("Absolute object F1 drop from clean")
    fig.suptitle("Full-train robustness: largest per-method F1 drops", y=1.02)
    fig.tight_layout()
    save_png(fig, FIGURES_DIR / "robustness_pow_full_train_worst_f1_drops.png")
    plt.close(fig)

    counts = failure_cases.groupby(["method", "failure_hint"], observed=True).size().unstack(fill_value=0)
    counts = counts.reindex(methods).dropna(how="all").fillna(0)
    hint_order = [column for column in ["NO_PRED", "COLLAPSE", "FN+FP", "FN", "FP/OVER", "COUNT", "MIXED", "NO_DROP"] if column in counts.columns]
    counts = counts[hint_order]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bottom = np.zeros(len(counts))
    x = np.arange(len(counts))
    colors = gradient_colors(len(hint_order))
    for hint, color in zip(hint_order, colors):
        values = counts[hint].to_numpy()
        ax.bar(x, values, bottom=bottom, label=hint, color=color, edgecolor="white")
        bottom += values
    ax.set_xticks(x, labels=[METHOD_LABELS[method] for method in counts.index])
    ax.set_ylabel("Worst-case rows")
    ax.set_title("Full-train robustness: worst-case failure hints")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    save_png(fig, FIGURES_DIR / "robustness_pow_full_train_failure_hint_counts.png")
    plt.close(fig)


def redraw_yolo_comparison(summary_path: Path, figure_path: Path, label_map: dict[str, str]) -> None:
    summary = pd.read_csv(summary_path)
    ordered = summary.sort_values("mean_object_f1", ascending=True).copy()
    labels = ordered["method_label"].map(label_map).fillna(ordered["method_label"])
    color_map = {
        "otsu_watershed": METHOD_PALETTE["otsu_watershed"],
        "cellpose_cpsam": METHOD_PALETTE["cellpose_cpsam"],
    }
    colors = [color_map.get(method, METHOD_PALETTE["yolo"]) for method in ordered["method"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    axes[0].barh(labels, ordered["mean_object_f1"], color=colors, edgecolor="white")
    axes[0].set_xlabel("Mean object F1")
    axes[0].set_xlim(0, 1)
    axes[0].set_title("Instance matching")
    axes[1].barh(labels, ordered["mean_absolute_count_error"], color=colors, edgecolor="white")
    axes[1].set_xlabel("Mean absolute count error")
    axes[1].set_title("Count accuracy")
    axes[1].set_yticklabels([])
    x_max = max(ordered["mean_absolute_count_error"]) * 1.12
    axes[1].set_xlim(0, x_max)
    fig.tight_layout()
    save_png(fig, figure_path)
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    redraw_dataset_audit()
    redraw_cellpose_method_availability()
    redraw_otsu_smoke()
    redraw_robustness_summary(
        RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_smoke_summary.csv",
        "robustness_pow_smoke",
    )
    redraw_robustness_summary(
        RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_clean20_summary.csv",
        "robustness_pow_clean20",
    )
    redraw_robustness_summary(
        RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_full_train_summary.csv",
        "robustness_pow_full_train",
    )
    redraw_yolo_threshold()
    redraw_sam2_sensitivity()
    redraw_clean20_diagnostics()
    redraw_full_train_diagnostics()
    redraw_baseline_clean_subset()
    redraw_individual_clean_subset_baselines()
    redraw_cellpose_parameter_diagnostic()
    redraw_yolo_comparison(
        RESULT_SUBDIRS["supervised"] / "yolo_label_budget_diagnostic_val_comparison_summary.csv",
        FIGURES_DIR / "supervised_yolo_label_budget_diagnostic_comparison.png",
        {
            "Cellpose-SAM": "Cellpose-SAM",
            "YOLO label-budget full train pool": "YOLO11n full",
            "YOLO label-budget 250": "YOLO11n 250",
            "YOLO fixed-budget 100": "YOLO11n 100",
            "Otsu + watershed": "Otsu",
        },
    )
    redraw_yolo_comparison(
        RESULT_SUBDIRS["supervised"] / "yolo_capacity_diagnostic_val_comparison_summary.csv",
        FIGURES_DIR / "supervised_yolo_capacity_diagnostic_comparison.png",
        {
            "Cellpose-SAM": "Cellpose-SAM",
            "YOLO11m full train pool": "YOLO11m full",
            "YOLO11n full train pool": "YOLO11n full",
            "Otsu + watershed": "Otsu",
        },
    )
    print("Redrew publication-style summary figures from existing CSV outputs.")


if __name__ == "__main__":
    main()
