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
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch

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
    "gaussian_noise": "Gaussian",
    "poisson_noise": "Poisson",
    "gaussian_blur": "Blur",
    "downsample_upsample": "Down",
    "intensity_scale": "Intensity",
    "contrast_inversion": "Invert",
}
PERTURBATION_ORDER = list(PERTURBATION_LABELS)
METRIC_LABELS = {
    "mean_object_f1": "Object F1",
    "mean_matched_iou": "Matched IoU",
    "mean_matched_dice": "Matched Dice",
}


def method_color(method: str) -> str:
    return METHOD_PALETTE.get(method, METHOD_PALETTE["neutral"])


def method_marker(method: str) -> str:
    return {
        "otsu_watershed": "s",
        "cellpose_cpsam": "o",
        "sam2_amg": "^",
    }.get(method, "o")


def style_axis(ax, *, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis)
    ax.spines["bottom"].set_color("#111827")
    ax.spines["left"].set_color("#111827")
    ax.tick_params(colors="#111827")


def add_point_cloud_region(
    ax,
    points: np.ndarray,
    color: str,
    *,
    xlim: tuple[float, float] = (0, 1.02),
    ylim: tuple[float, float] = (0, 1.02),
    sigma: float = 0.055,
    threshold: float = 0.20,
    alpha: float = 0.14,
) -> None:
    """Add a soft density region around observed 2D points."""
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return

    x_grid = np.linspace(*xlim, 180)
    y_grid = np.linspace(*ylim, 180)
    xx, yy = np.meshgrid(x_grid, y_grid)
    density = np.zeros_like(xx)
    for x_point, y_point in points:
        density += np.exp(-(((xx - x_point) ** 2 + (yy - y_point) ** 2) / (2 * sigma**2)))
    if density.max() <= threshold:
        return
    ax.contourf(
        xx,
        yy,
        density,
        levels=[threshold, density.max()],
        colors=[color],
        alpha=alpha,
        antialiased=True,
        zorder=1,
    )


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Return 2D convex hull vertices using Andrew's monotonic chain."""
    unique = sorted({(float(x), float(y)) for x, y in points if np.isfinite(x) and np.isfinite(y)})
    if len(unique) <= 2:
        return np.asarray(unique)

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return np.asarray(lower[:-1] + upper[:-1])


def add_covariance_ellipse(ax, points: np.ndarray, color: str, *, n_std: float = 1.7) -> None:
    """Add a compact ellipse around a clustered 2D point cloud."""
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        return
    center = points.mean(axis=0)
    covariance = np.cov(points, rowvar=False)
    if not np.isfinite(covariance).all():
        return
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 1e-6)
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    ellipse = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        linewidth=1.6,
        alpha=0.16,
        zorder=1,
    )
    ax.add_patch(ellipse)


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

    non_clean = summary[summary["perturbation"] != "clean"].copy()
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    x = np.arange(len(non_clean))
    ax.plot(
        x,
        non_clean["relative_object_f1_drop"],
        marker="s",
        markersize=5.8,
        markerfacecolor="white",
        markeredgewidth=1.4,
        color=METHOD_PALETTE["otsu_watershed"],
        linewidth=2.0,
        zorder=3,
    )
    for xpos, value in zip(x, non_clean["relative_object_f1_drop"]):
        if abs(value) >= 0.05:
            ax.text(xpos, value, f"{value:.0%}", va="bottom", ha="center", fontsize=8, fontweight="bold", color="#374151")
    ax.axhline(0, color="#4b5563", linewidth=0.9)
    ax.set_xticks(x, labels=non_clean["label"])
    ax.set_ylabel("Relative F1 drop from clean")
    ax.set_title("Otsu smoke-test performance loss")
    style_axis(ax, grid_axis="y")
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

    if figure_prefix == "robustness_pow_full_train":
        drop = (
            summary[summary["perturbation"] != "clean"]
            .pivot(index="perturbation", columns="method", values="relative_object_f1_drop")
            .reindex([item for item in PERTURBATION_ORDER if item != "clean"])
        )
        drop = drop[[method for method in methods if method in drop.columns]].dropna(how="all")

        fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))
        for method in pivot.columns:
            axes[0].plot(
                x,
                pivot[method].to_numpy(),
                marker=method_marker(method),
                markersize=5.8,
                markerfacecolor="white",
                markeredgewidth=1.4,
                linewidth=2.2,
                label=METHOD_LABELS.get(method, method),
                color=method_color(method),
            )
        axes[0].set_xticks(x, labels=labels)
        axes[0].set_ylim(0, 1.02)
        axes[0].set_ylabel("Mean object F1")
        axes[0].set_title("Mean F1 by perturbation")
        style_axis(axes[0], grid_axis="y")

        drop_x = np.arange(len(drop))
        for method in drop.columns:
            values = drop[method].to_numpy()
            axes[1].plot(
                drop_x,
                values,
                marker=method_marker(method),
                markersize=5.8,
                markerfacecolor="white",
                markeredgewidth=1.4,
                linewidth=2.2,
                label=METHOD_LABELS.get(method, method),
                color=method_color(method),
            )
            for xpos, value in enumerate(values):
                if value >= 0.10:
                    axes[1].text(xpos, value, f"{value:.0%}", va="bottom", ha="center", fontsize=8, fontweight="bold", color="#374151")
        axes[1].axhline(0, color="#4b5563", linewidth=0.9)
        axes[1].set_xticks(drop_x, labels=[PERTURBATION_LABELS[item] for item in drop.index])
        axes[1].set_ylabel("Relative F1 drop from clean")
        axes[1].set_title("Performance loss from clean")
        style_axis(axes[1], grid_axis="y")

        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", ncols=len(handles), bbox_to_anchor=(0.5, 1.04), frameon=False)
        fig.suptitle("Full-train robustness summary", y=1.14)
        fig.tight_layout()
        save_png(fig, FIGURES_DIR / "robustness_pow_full_train_summary.png")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    for method in pivot.columns:
        ax.plot(
            x,
            pivot[method].to_numpy(),
            marker=method_marker(method),
            markersize=5.5,
            markerfacecolor="white",
            markeredgewidth=1.3,
            linewidth=2.0,
            label=METHOD_LABELS.get(method, method),
            color=method_color(method),
        )
    ax.set_xticks(x, labels=labels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Mean object F1")
    ax.set_title("Robustness to image perturbations")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncols=min(len(pivot.columns), 3))
    style_axis(ax, grid_axis="y")
    save_png(fig, FIGURES_DIR / f"{figure_prefix}_mean_f1.png")
    plt.close(fig)

    drop = (
        summary[summary["perturbation"] != "clean"]
        .pivot(index="perturbation", columns="method", values="relative_object_f1_drop")
        .reindex([item for item in PERTURBATION_ORDER if item != "clean"])
    )
    drop = drop[[method for method in methods if method in drop.columns]]
    drop = drop.dropna(how="all")
    x = np.arange(len(drop))

    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    for method in drop.columns:
        values = drop[method].to_numpy()
        ax.plot(
            x,
            values,
            marker=method_marker(method),
            markersize=5.5,
            markerfacecolor="white",
            markeredgewidth=1.3,
            linewidth=2.0,
            label=METHOD_LABELS.get(method, method),
            color=method_color(method),
        )
        if np.isfinite(values).any():
            max_index = int(np.nanargmax(values))
            max_value = values[max_index]
            if max_value >= 0.15:
                ax.annotate(
                    f"{max_value:.0%}",
                    xy=(x[max_index], max_value),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                    color="#374151",
                    arrowprops={"arrowstyle": "-", "color": "#6b7280", "lw": 0.7},
                )
    ax.axhline(0, color="#4b5563", linewidth=0.8)
    ax.set_xticks(x, labels=[PERTURBATION_LABELS[item] for item in drop.index])
    ax.set_ylabel("Relative F1 drop")
    ax.set_title("Performance loss from clean")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncols=min(len(drop.columns), 3))
    style_axis(ax, grid_axis="y")
    save_png(fig, FIGURES_DIR / f"{figure_prefix}_relative_f1_drop.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 2.5 + 0.45 * len(pivot.columns)))
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
    ax.grid(False)
    save_png(fig, FIGURES_DIR / f"{figure_prefix}_method_condition_heatmap.png")
    plt.close(fig)


def redraw_yolo_threshold() -> None:
    summary = pd.read_csv(RESULT_SUBDIRS["supervised"] / "yolo_threshold_diagnostic_summary.csv")
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax2 = ax.twinx()
    ax2.bar(
        summary["conf"],
        summary["mean_absolute_count_error"],
        width=0.045,
        color="#dbeafe",
        edgecolor="#9ca3af",
        linewidth=0.8,
        alpha=0.75,
        label="Count error",
        zorder=1,
    )
    ax.plot(
        summary["conf"],
        summary["mean_object_f1"],
        marker="o",
        markersize=5.5,
        markerfacecolor="white",
        markeredgewidth=1.3,
        color=METHOD_PALETTE["yolo"],
        linewidth=2.2,
        label="Object F1",
        zorder=3,
    )
    best = summary.loc[summary["mean_object_f1"].idxmax()]
    ax.annotate(
        f"best F1 {best['mean_object_f1']:.3f}",
        xy=(best["conf"], best["mean_object_f1"]),
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=8,
        fontweight="bold",
        color="#374151",
        arrowprops={"arrowstyle": "-", "color": "#6b7280", "lw": 0.8},
    )
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Mean object F1")
    ax2.set_ylabel("Mean absolute count error")
    ax.set_xticks(summary["conf"], labels=[f"{value:.2f}" for value in summary["conf"]])
    ax.set_ylim(0.70, 0.90)
    ax.set_title("YOLO F1 by confidence threshold")
    style_axis(ax, grid_axis="y")
    ax2.grid(False)
    handles = [
        Line2D([0], [0], color=METHOD_PALETTE["yolo"], marker="o", markerfacecolor="white", linewidth=2, label="Object F1"),
        Patch(facecolor="#dbeafe", edgecolor="#9ca3af", label="Count error"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncols=2)
    save_png(fig, FIGURES_DIR / "supervised_yolo_threshold_diagnostic_f1.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(
        summary["conf"],
        summary["mean_absolute_count_error"],
        marker="s",
        markersize=5.5,
        markerfacecolor="white",
        markeredgewidth=1.3,
        color=METHOD_PALETTE["warning"],
        linewidth=2.0,
        label="Count error",
    )
    if {"mean_missed_object_rate", "mean_fp_per_true_instance"}.issubset(summary.columns):
        ax2 = ax.twinx()
        ax2.plot(
            summary["conf"],
            summary["mean_missed_object_rate"],
            marker="o",
            markersize=4.8,
            markerfacecolor="white",
            markeredgewidth=1.1,
            color=METHOD_PALETTE["cellpose_cpsam"],
            linewidth=1.8,
            label="Missed rate",
        )
        ax2.plot(
            summary["conf"],
            summary["mean_fp_per_true_instance"],
            marker="^",
            markersize=4.8,
            markerfacecolor="white",
            markeredgewidth=1.1,
            color=METHOD_PALETTE["sam2_amg"],
            linewidth=1.8,
            label="FP / true",
        )
        ax2.set_ylabel("Failure-rate diagnostic")
        ax2.grid(False)
        handles = ax.get_lines() + ax2.get_lines()
        ax.legend(handles=handles, labels=[line.get_label() for line in handles], loc="upper right")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Mean absolute count error")
    ax.set_xticks(summary["conf"], labels=[f"{value:.2f}" for value in summary["conf"]])
    ax.set_title("YOLO count error by confidence threshold")
    style_axis(ax, grid_axis="y")
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
    ax.grid(False)
    save_png(fig, FIGURES_DIR / "robustness_sam2_amg_sensitivity_clean20_mean_f1.png")
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
    compact_labels = {
        "rgb_baseline": "RGB",
        "gray_mean": "Gray",
        "gray_max": "Gray max",
        "rgb_invert": "Invert",
        "rgb_cellprob_-1": "CP -1",
        "rgb_cellprob_-2": "CP -2",
        "rgb_flow_0": "Flow 0",
        "rgb_diameter_15": "D15",
        "gray_diameter_15": "Gray+D15",
        "rgb_diameter_30": "D30",
        "rgb_diameter_60": "D60",
    }

    def draw_summary(frame: pd.DataFrame, title: str, output_path: Path) -> None:
        frame = frame.copy()
        frame["label"] = frame["config_id"].map(short_labels).fillna(frame["config_id"])
        frame["compact_label"] = frame["config_id"].map(compact_labels).fillna(frame["label"])
        ordered = frame.sort_values("mean_object_f1", ascending=False).reset_index(drop=True)
        x = np.arange(len(ordered))
        colors = gradient_colors(len(ordered))

        fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), gridspec_kw={"width_ratios": [1.35, 1]})
        ax = axes[0]
        ax2 = ax.twinx()
        ax2.bar(
            x,
            ordered["mean_absolute_count_error"],
            width=0.62,
            color="#dbeafe",
            edgecolor="#9ca3af",
            linewidth=0.8,
            alpha=0.42,
            label="Count error",
            zorder=0,
        )
        ax.plot(
            x,
            ordered["mean_object_f1"],
            color=METHOD_PALETTE["cellpose_cpsam"],
            marker="o",
            markersize=5.8,
            markerfacecolor="white",
            markeredgewidth=1.3,
            linewidth=2.2,
            label="Object F1",
            zorder=3,
        )
        best = ordered.iloc[0]
        ax.annotate(
            f"best F1 {best['mean_object_f1']:.3f}",
            xy=(0, best["mean_object_f1"]),
            xytext=(12, 10),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="#374151",
            arrowprops={"arrowstyle": "-", "color": "#6b7280", "lw": 0.8},
        )
        ax.set_xticks(x, labels=ordered["compact_label"])
        f1_span = max(ordered["mean_object_f1"].max() - ordered["mean_object_f1"].min(), 0.01)
        ax.set_ylim(
            max(0, ordered["mean_object_f1"].min() - max(0.006, f1_span * 1.4)),
            min(1.0, ordered["mean_object_f1"].max() + max(0.006, f1_span * 1.4)),
        )
        ax.set_ylabel("Mean object F1")
        ax2.set_ylabel("Mean absolute count error")
        ax.set_title("Quality and count error")
        style_axis(ax, grid_axis="y")
        ax2.grid(False)
        ax.legend(
            handles=[
                Line2D([0], [0], color=METHOD_PALETTE["cellpose_cpsam"], marker="o", markerfacecolor="white", linewidth=2, label="Object F1"),
                Patch(facecolor="#dbeafe", edgecolor="#9ca3af", label="Count error"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.22),
            ncols=2,
        )

        ax = axes[1]
        sizes = 60 + 18 * ordered["mean_absolute_count_error"].to_numpy()
        ax.scatter(
            ordered["median_latency_ms"],
            ordered["mean_object_f1"],
            s=sizes,
            c=colors,
            alpha=0.72,
            edgecolors="white",
            linewidth=0.8,
        )
        if len(ordered) > 5:
            label_indices = {
                0,
                int(ordered["median_latency_ms"].idxmin()),
                int(ordered["median_latency_ms"].idxmax()),
                int(ordered["mean_object_f1"].idxmin()),
            }
        else:
            label_indices = set(range(len(ordered)))
        for index, row in ordered.iterrows():
            if index not in label_indices:
                continue
            ax.annotate(
                row["compact_label"],
                xy=(row["median_latency_ms"], row["mean_object_f1"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7,
                color="#374151",
            )
        ax.set_xscale("log")
        ax.set_xlabel("Median latency (ms/image, log)")
        ax.set_ylabel("Mean object F1")
        ax.set_title("Quality-latency trade-off")
        style_axis(ax, grid_axis="both")

        fig.suptitle(title, y=1.08)
        fig.tight_layout()
        save_png(fig, output_path)
        plt.close(fig)

    def draw_line_bar_panel(ax, frame: pd.DataFrame, title: str) -> None:
        ordered = frame.copy()
        ordered["label"] = ordered["config_id"].map(short_labels).fillna(ordered["config_id"])
        ordered["compact_label"] = ordered["config_id"].map(compact_labels).fillna(ordered["label"])
        ordered = ordered.sort_values("mean_object_f1", ascending=False).reset_index(drop=True)
        x = np.arange(len(ordered))
        ax2 = ax.twinx()
        ax2.bar(
            x,
            ordered["mean_absolute_count_error"],
            width=0.58,
            color="#dbeafe",
            edgecolor="#9ca3af",
            linewidth=0.8,
            alpha=0.42,
            zorder=0,
        )
        ax.plot(
            x,
            ordered["mean_object_f1"],
            color=METHOD_PALETTE["cellpose_cpsam"],
            marker="o",
            markersize=5.4,
            markerfacecolor="white",
            markeredgewidth=1.2,
            linewidth=2.0,
            zorder=3,
        )
        best = ordered.iloc[0]
        ax.annotate(
            f"{best['mean_object_f1']:.3f}",
            xy=(0, best["mean_object_f1"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="#374151",
            arrowprops={"arrowstyle": "-", "color": "#6b7280", "lw": 0.7},
        )
        f1_span = max(ordered["mean_object_f1"].max() - ordered["mean_object_f1"].min(), 0.01)
        ax.set_ylim(
            max(0, ordered["mean_object_f1"].min() - max(0.006, f1_span * 1.4)),
            min(1.0, ordered["mean_object_f1"].max() + max(0.006, f1_span * 1.4)),
        )
        ax.set_xticks(x, labels=ordered["compact_label"])
        ax.set_title(title)
        ax.set_ylabel("Mean object F1")
        ax2.set_ylabel("Count error")
        style_axis(ax, grid_axis="y")
        ax2.grid(False)

    def draw_heldout_decision(heldout: pd.DataFrame, input_lock: pd.DataFrame, output_path: Path) -> None:
        heldout = heldout.copy()
        input_lock = input_lock.copy()
        for frame, source in [(heldout, "Parameter check"), (input_lock, "Input lock")]:
            frame["source"] = source
            frame["label"] = frame["config_id"].map(short_labels).fillna(frame["config_id"])

        fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9), gridspec_kw={"width_ratios": [1.2, 1.0, 1.1]})
        draw_line_bar_panel(axes[0], heldout, "Held-out parameter check")
        draw_line_bar_panel(axes[1], input_lock, "Input-mode lock")

        combined = pd.concat([heldout, input_lock], ignore_index=True)
        combined["plot_label"] = combined.apply(
            lambda row: (
                "Gray+d15 lock"
                if row["source"] == "Input lock" and row["config_id"] == "gray_diameter_15"
                else "RGB d15 lock"
                if row["source"] == "Input lock" and row["config_id"] == "rgb_diameter_15"
                else row["label"]
            ),
            axis=1,
        )
        marker_map = {"Parameter check": "o", "Input lock": "s"}
        color_map = {"Parameter check": METHOD_PALETTE["cellpose_cpsam"], "Input lock": METHOD_PALETTE["sam2_amg"]}
        label_offsets = [(5, 5), (5, -10), (5, 14), (-58, 8), (-56, -12)]
        offset_index = 0
        for source, frame in combined.groupby("source"):
            axes[2].scatter(
                frame["median_latency_ms"],
                frame["mean_object_f1"],
                s=80 + 12 * frame["mean_absolute_count_error"],
                marker=marker_map[source],
                color=color_map[source],
                alpha=0.72,
                edgecolors="white",
                linewidth=0.8,
                label=source,
            )
            for _, row in frame.iterrows():
                offset = label_offsets[offset_index % len(label_offsets)]
                offset_index += 1
                axes[2].annotate(
                    row["plot_label"],
                    xy=(row["median_latency_ms"], row["mean_object_f1"]),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=7,
                    color="#374151",
                )
        axes[2].set_xscale("log")
        axes[2].set_xlabel("Median latency (ms/image, log)")
        axes[2].set_ylabel("Mean object F1")
        axes[2].set_title("Combined quality-latency view")
        axes[2].legend(loc="lower right")
        style_axis(axes[2], grid_axis="both")
        fig.legend(
            handles=[
                Line2D([0], [0], color=METHOD_PALETTE["cellpose_cpsam"], marker="o", markerfacecolor="white", linewidth=2, label="Object F1"),
                Patch(facecolor="#dbeafe", edgecolor="#9ca3af", alpha=0.55, label="Count error"),
            ],
            loc="upper center",
            ncols=2,
            bbox_to_anchor=(0.5, 1.08),
        )
        fig.suptitle("Cellpose-SAM held-out parameter decision", y=1.18)
        fig.tight_layout()
        save_png(fig, output_path)
        plt.close(fig)

    summary = pd.read_csv(RESULT_SUBDIRS["baselines"] / "cellpose_cpsam_parameter_diagnostic_summary.csv")
    draw_summary(
        summary,
        "Cellpose-SAM parameter diagnostic",
        FIGURES_DIR / "cellpose_cpsam_parameter_diagnostic_f1.png",
    )

    heldout_path = RESULT_SUBDIRS["baselines"] / "cellpose_cpsam_parameter_diagnostic_heldout_val_summary.csv"
    input_lock_path = RESULT_SUBDIRS["baselines"] / "cellpose_cpsam_input_mode_lock_heldout_val_summary.csv"
    if heldout_path.exists():
        heldout = pd.read_csv(heldout_path)
        if input_lock_path.exists():
            input_lock = pd.read_csv(input_lock_path)
            draw_heldout_decision(
                heldout,
                input_lock,
                FIGURES_DIR / "cellpose_cpsam_parameter_diagnostic_heldout_val_f1.png",
            )
        else:
            draw_summary(
                heldout,
                "Cellpose-SAM held-out parameter check",
                FIGURES_DIR / "cellpose_cpsam_parameter_diagnostic_heldout_val_f1.png",
            )


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

    # Metric comparison: show per-image distributions, not only three means.
    metric_cols = ["mean_object_f1", "mean_matched_iou", "mean_matched_dice"]
    raw_metric_cols = ["object_f1", "mean_matched_iou", "mean_matched_dice"]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.5), sharey=True)
    axes_array = np.atleast_1d(axes)
    method_positions = np.arange(len(METHOD_ORDER))
    for ax, column, label in zip(axes_array, raw_metric_cols, [METRIC_LABELS[col] for col in metric_cols]):
        for index, method in enumerate(METHOD_ORDER):
            values = metrics.loc[metrics["method"] == method, column].to_numpy()
            jitter = ((np.arange(len(values)) % 7) - 3) * 0.018
            ax.scatter(
                np.full(len(values), method_positions[index]) + jitter,
                values,
                s=22,
                alpha=0.55,
                color=method_colors[method],
                linewidths=0,
            )
            mean_value = float(np.mean(values))
            sem_value = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            ax.errorbar(
                method_positions[index],
                mean_value,
                yerr=sem_value,
                color="#111827",
                marker="_",
                markersize=16,
                capsize=3,
                linewidth=1.1,
                zorder=5,
            )
        ax.set_xticks(method_positions, labels=[METHOD_LABELS[method] for method in METHOD_ORDER])
        ax.set_ylim(0, 1.04)
        ax.set_title(label)
        style_axis(ax, grid_axis="y")
    axes_array[0].set_ylabel("Per-image score")
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=method_colors[method], markersize=7, label=BASELINE_METHOD_LABELS[method])
            for method in METHOD_ORDER
        ],
        loc="upper center",
        ncols=3,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.suptitle("Clean subset baseline score distributions", y=1.18)
    fig.tight_layout()
    save_png(fig, FIGURES_DIR / "baseline_clean_subset_metric_comparison.png")
    plt.close(fig)

    # Count error: per-image distribution with mean and standard error.
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for index, method in enumerate(METHOD_ORDER):
        values = metrics.loc[metrics["method"] == method, "absolute_count_error"].to_numpy()
        box = ax.boxplot(
            [values],
            positions=[index],
            widths=0.45,
            showfliers=False,
            patch_artist=True,
        )
        box["boxes"][0].set_facecolor(method_colors[method])
        box["boxes"][0].set_alpha(0.22)
        box["boxes"][0].set_edgecolor(method_colors[method])
        for median in box["medians"]:
            median.set_color("#111827")
        jitter = ((np.arange(len(values)) % 9) - 4) * 0.025
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=22,
            alpha=0.55,
            color=method_colors[method],
            linewidths=0,
        )
        mean_value = float(np.mean(values))
        ax.scatter(index, mean_value, s=64, marker="D", color="white", edgecolor="#111827", linewidth=1.0, zorder=6)
        ax.annotate(
            f"mean {mean_value:.1f}",
            xy=(index, mean_value),
            xytext=(18, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8,
            fontweight="bold",
            color="#374151",
        )
    ax.set_xticks(method_positions, labels=[METHOD_LABELS[method] for method in METHOD_ORDER])
    ax.set_title("Clean subset count-error distribution")
    ax.set_yscale("symlog", linthresh=5)
    ax.set_yticks([0, 1, 5, 10, 50, 100, 500])
    ax.set_yticklabels(["0", "1", "5", "10", "50", "100", "500"])
    ax.set_ylim(0, max(metrics["absolute_count_error"].max() * 1.18, 10))
    ax.set_ylabel("Absolute count error per image")
    style_axis(ax, grid_axis="y")
    save_png(fig, FIGURES_DIR / "baseline_clean_subset_count_error_comparison.png")
    plt.close(fig)

    # Latency: quality/runtime trade-off, with every image shown.
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    for method in METHOD_ORDER:
        method_metrics = metrics[metrics["method"] == method]
        ax.scatter(
            method_metrics["latency_ms"],
            method_metrics["object_f1"],
            s=28,
            alpha=0.42,
            color=method_colors[method],
            linewidths=0,
        )
        median_latency = float(method_metrics["latency_ms"].median())
        mean_f1 = float(method_metrics["object_f1"].mean())
        ax.scatter(
            median_latency,
            mean_f1,
            s=96,
            marker=method_marker(method),
            facecolor="white",
            edgecolor=method_colors[method],
            linewidth=1.8,
            zorder=6,
            label=BASELINE_METHOD_LABELS[method],
        )
        ax.annotate(
            METHOD_LABELS[method],
            xy=(median_latency, mean_f1),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="#374151",
        )
    ax.set_xscale("log")
    ax.set_ylim(0, 1.04)
    ax.set_title("Clean subset quality-latency trade-off")
    ax.set_xlabel("Latency (ms/image, log scale)")
    ax.set_ylabel("Object F1")
    ax.legend(loc="lower right")
    style_axis(ax, grid_axis="both")
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
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    for method in METHOD_ORDER:
        method_metrics = metrics[metrics["method"] == method]
        points = method_metrics[["recall", "precision"]].to_numpy()
        add_point_cloud_region(ax, points, method_colors[method])
        ax.scatter(
            method_metrics["recall"],
            method_metrics["precision"],
            s=28,
            alpha=0.68,
            color=method_colors[method],
            linewidths=0,
            zorder=3,
        )
        ax.scatter(
            method_metrics["recall"].mean(),
            method_metrics["precision"].mean(),
            s=110,
            marker=method_marker(method),
            facecolor="white",
            edgecolor=method_colors[method],
            linewidth=2.0,
            label=BASELINE_METHOD_LABELS[method],
            zorder=5,
        )
        ax.annotate(
            METHOD_LABELS[method],
            xy=(method_metrics["recall"].mean(), method_metrics["precision"].mean()),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="#374151",
            zorder=6,
        )
    ax.plot([0, 1], [0, 1], color="#111827", linewidth=1, linestyle="--", alpha=0.45)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("Clean subset precision-recall regions")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=False, loc="upper left")
    style_axis(ax, grid_axis="both")
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
    ax.grid(False)
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
    frame["decision"] = frame["runnable_as_distinct_method"].map({True: "Use", False: "Defer"})
    frame["short_method"] = frame["method"].str.replace("cellpose_", "", regex=False).str.replace("_", " ")
    status_colors = {
        "available": METHOD_PALETTE["cellpose_cpsam"],
        "alias_to_cpsam": METHOD_PALETTE["otsu_watershed"],
        "error": METHOD_PALETTE["warning"],
    }

    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    ax.axis("off")
    columns = ["Method", "Requested", "Loaded", "Status", "Decision"]
    x_positions = [0.01, 0.31, 0.49, 0.64, 0.84]
    for x, column in zip(x_positions, columns):
        ax.text(x, 0.96, column, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    ax.plot([0.01, 0.98], [0.89, 0.89], transform=ax.transAxes, color="#d1d5db", linewidth=0.9)

    y_start = 0.82
    y_step = 0.125
    for index, row in frame.iterrows():
        y = y_start - index * y_step
        color = status_colors.get(row["status"], METHOD_PALETTE["neutral"])
        values = [
            row["short_method"],
            row["requested_model"],
            row["loaded_model"] if pd.notna(row["loaded_model"]) and row["loaded_model"] else "not loaded",
            row["status"],
            row["decision"],
        ]
        ax.scatter([x_positions[3] - 0.018], [y + 0.002], transform=ax.transAxes, s=44, color=color, edgecolors="white", linewidth=0.5)
        for x, value in zip(x_positions, values):
            weight = "bold" if x == x_positions[-1] and value == "Use" else "normal"
            ax.text(x, y, str(value), transform=ax.transAxes, fontsize=8, va="center", color="#111827", fontweight=weight)
    ax.text(
        0.01,
        0.05,
        "Only cpsam is runnable as a distinct Cellpose-family baseline in the current Cellpose 4.x environment.",
        transform=ax.transAxes,
        fontsize=8,
        color="#4b5563",
    )
    ax.set_title("Cellpose-family method availability audit")
    save_png(fig, FIGURES_DIR / "cellpose_method_availability.png")
    plt.close(fig)


def redraw_clean_subset_count_agreement() -> None:
    specs = {
        "otsu_watershed": ("otsu_watershed_clean_subset_metrics.csv", "Otsu + watershed"),
        "cellpose_cpsam": ("cellpose_cpsam_clean_subset_metrics.csv", "Cellpose-SAM"),
        "sam2_amg": ("sam2_amg_clean_subset_metrics.csv", "SAM2 AMG"),
    }
    method_frames = {
        method: pd.read_csv(RESULT_SUBDIRS["baselines"] / filename)
        for method, (filename, _label) in specs.items()
    }
    x_max = int(np.ceil(max(metrics["true_instances"].max() for metrics in method_frames.values()) / 25) * 25)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.7), sharex=True)
    for ax, method in zip(axes, METHOD_ORDER):
        metrics = method_frames[method].copy()
        label = specs[method][1]
        count_error = metrics["pred_instances"] - metrics["true_instances"]
        mae = count_error.abs().mean()
        under_rate = (count_error < 0).mean()
        over_rate = (count_error > 0).mean()
        color = METHOD_PALETTE.get(method, METHOD_PALETTE["neutral"])
        y_max = int(np.ceil(metrics["pred_instances"].max() / 25) * 25)
        axis_max = max(x_max, y_max)

        ax.fill_between([0, axis_max], [0, axis_max], [axis_max, axis_max], color="#e8a300", alpha=0.035, linewidth=0)
        ax.fill_between([0, axis_max], [0, 0], [0, axis_max], color="#4f7fc4", alpha=0.035, linewidth=0)
        points = metrics[["true_instances", "pred_instances"]].to_numpy()
        add_point_cloud_region(
            ax,
            points,
            color,
            xlim=(0, axis_max),
            ylim=(0, axis_max),
            sigma=max(axis_max * 0.045, 8),
            threshold=0.16,
            alpha=0.14,
        )
        ax.scatter(
            metrics["true_instances"],
            metrics["pred_instances"],
            alpha=0.75,
            color=color,
            s=28,
            linewidths=0,
            zorder=3,
        )
        ax.plot([0, axis_max], [0, axis_max], color="#111827", linewidth=1.1, linestyle="--")
        ax.text(
            0.04,
            0.96,
            f"MAE {mae:.1f}\nunder {under_rate:.0%} / over {over_rate:.0%}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            fontweight="bold",
            color="#374151",
        )
        ax.set_title(label)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, axis_max)
        style_axis(ax, grid_axis="both")
    axes[0].set_ylabel("Predicted instances")
    for ax in axes:
        ax.set_xlabel("Ground-truth instances")
    fig.suptitle("Clean subset count agreement", y=1.03)
    fig.tight_layout()
    save_png(fig, FIGURES_DIR / "baseline_clean_subset_count_agreement.png")
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
        ax.grid(False)
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

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    positions = np.arange(len(non_clean_order))
    offset = 0.16
    for method_index, method in enumerate(methods):
        method_positions = positions + (method_index - (len(methods) - 1) / 2) * offset * 2
        values = [
            non_clean[
                (non_clean["method"] == method) & (non_clean["perturbation"] == perturbation)
            ]["absolute_object_f1_drop"].to_numpy()
            for perturbation in non_clean_order
        ]
        box = axes[0].boxplot(
            values,
            positions=method_positions,
            widths=0.24,
            showfliers=False,
            patch_artist=True,
        )
        for patch in box["boxes"]:
            patch.set_facecolor(method_color(method))
            patch.set_alpha(0.18)
            patch.set_edgecolor(method_color(method))
        for median in box["medians"]:
            median.set_color(method_color(method))
            median.set_linewidth(1.6)
        for index, series in enumerate(values):
            if len(series) == 0:
                continue
            jitter = ((np.arange(len(series)) % 9) - 4) * 0.007
            axes[0].scatter(
                method_positions[index] + jitter,
                series,
                s=7,
                alpha=0.16,
                color=method_color(method),
                linewidths=0,
            )
    axes[0].axhline(0, color="#111827", linewidth=1)
    axes[0].set_xticks(positions, labels=[PERTURBATION_LABELS[label] for label in non_clean_order])
    axes[0].set_ylabel("Absolute object F1 drop from clean")
    axes[0].set_title("Per-image F1 drop distributions")
    style_axis(axes[0], grid_axis="y")

    counts = failure_cases.groupby(["method", "failure_hint"], observed=True).size().unstack(fill_value=0)
    counts = counts.reindex(methods).dropna(how="all").fillna(0)
    hint_order = [column for column in ["NO_PRED", "COLLAPSE", "FN+FP", "FN", "FP/OVER", "COUNT", "MIXED", "NO_DROP"] if column in counts.columns]
    counts = counts[hint_order]
    bottom = np.zeros(len(counts))
    x = np.arange(len(counts))
    colors = gradient_colors(len(hint_order))
    for hint, color in zip(hint_order, colors):
        values = counts[hint].to_numpy()
        axes[1].bar(x, values, bottom=bottom, label=hint, color=color, edgecolor="white")
        bottom += values
    axes[1].set_xticks(x, labels=[METHOD_LABELS[method] for method in counts.index])
    axes[1].set_ylabel("Worst-case rows")
    axes[1].set_title("Worst-case failure hints")
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    style_axis(axes[1], grid_axis="y")
    legend_handles = [
        Line2D([0], [0], color=method_color(method), marker=method_marker(method), markerfacecolor="white", markeredgewidth=1.4, label=METHOD_LABELS[method])
        for method in methods
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncols=len(legend_handles), bbox_to_anchor=(0.36, 1.02), frameon=False)
    fig.suptitle("Full-train robustness failure diagnostics", y=1.12)
    fig.tight_layout()
    save_png(fig, FIGURES_DIR / "robustness_pow_full_train_failure_diagnostics.png")
    plt.close(fig)


def redraw_yolo_comparison(summary_path: Path, figure_path: Path, label_map: dict[str, str]) -> None:
    summary = pd.read_csv(summary_path)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    cellpose = summary[summary["method"] == "cellpose_cpsam"].iloc[0]
    otsu = summary[summary["method"] == "otsu_watershed"].iloc[0]

    if "label_budget" in figure_path.name:
        yolo_order = [
            ("yolo_fixed_budget_100", 100, "100"),
            ("yolo_label_budget_250", 250, "250"),
            ("yolo_label_budget_full_train_pool", 536, "Full"),
        ]
        yolo = pd.concat(
            [
                summary[summary["method"] == method].assign(train_images=train_images, label=label)
                for method, train_images, label in yolo_order
                if method in set(summary["method"])
            ],
            ignore_index=True,
        ).sort_values("train_images")
        x = np.arange(len(yolo))
        xlabels = yolo["label"].tolist()
        title = "YOLO label-budget diagnostic"
        x_label = "Training images"
    else:
        yolo_order = [
            ("yolo11n_full_train_pool", 0, "YOLO11n"),
            ("yolo11m_full_train_pool", 1, "YOLO11m"),
        ]
        yolo = pd.concat(
            [
                summary[summary["method"] == method].assign(train_images=index, label=label)
                for method, index, label in yolo_order
                if method in set(summary["method"])
            ],
            ignore_index=True,
        ).sort_values("train_images")
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
        y_f1 = yolo["mean_object_f1"].to_numpy()
        y_count = yolo["mean_absolute_count_error"].to_numpy()
        labels = yolo["label"].tolist()
        x_positions = np.arange(len(yolo))

        axes[0].plot(
            x_positions,
            y_f1,
            color=METHOD_PALETTE["yolo"],
            marker="o",
            markersize=6.5,
            markerfacecolor="white",
            markeredgewidth=1.7,
            linewidth=2.2,
            label="YOLO",
        )
        axes[0].axhline(cellpose["mean_object_f1"], color=METHOD_PALETTE["cellpose_cpsam"], linewidth=2.0, label="Cellpose-SAM")
        axes[0].axhline(otsu["mean_object_f1"], color=METHOD_PALETTE["otsu_watershed"], linewidth=1.8, linestyle="--", label="Otsu")
        axes[0].scatter(
            x_positions,
            y_f1,
            s=84,
            marker="o",
            facecolor="white",
            edgecolor=METHOD_PALETTE["yolo"],
            linewidth=1.8,
            zorder=4,
        )
        for xpos, value in zip(x_positions, y_f1):
            axes[0].annotate(
                f"{value:.3f}",
                xy=(xpos, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#374151",
            )
        if len(y_f1) == 2:
            delta = y_f1[-1] - y_f1[0]
            gap = cellpose["mean_object_f1"] - y_f1[-1]
            axes[0].text(
                0.5,
                min(y_f1) - 0.035,
                f"+{delta:.3f} F1; {gap:.3f} below Cellpose",
                ha="center",
                fontsize=8,
                fontweight="bold",
                color="#374151",
            )
        axes[0].set_xticks(x_positions, labels=labels)
        axes[0].set_ylim(0.60, 0.95)
        axes[0].set_ylabel("Mean object F1")
        axes[0].set_title("Capacity gain is small")
        style_axis(axes[0], grid_axis="y")

        axes[1].plot(
            x_positions,
            y_count,
            color=METHOD_PALETTE["yolo"],
            marker="s",
            markersize=6.5,
            markerfacecolor="white",
            markeredgewidth=1.7,
            linewidth=2.2,
            label="YOLO",
        )
        axes[1].axhline(cellpose["mean_absolute_count_error"], color=METHOD_PALETTE["cellpose_cpsam"], linewidth=2.0, label="Cellpose-SAM")
        axes[1].axhline(otsu["mean_absolute_count_error"], color=METHOD_PALETTE["otsu_watershed"], linewidth=1.8, linestyle="--", label="Otsu")
        axes[1].scatter(
            x_positions,
            y_count,
            s=84,
            marker="s",
            facecolor="white",
            edgecolor=METHOD_PALETTE["yolo"],
            linewidth=1.8,
            zorder=4,
        )
        for xpos, value in zip(x_positions, y_count):
            axes[1].annotate(
                f"{value:.1f}",
                xy=(xpos, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#374151",
            )
        if len(y_count) == 2:
            delta = y_count[-1] - y_count[0]
            axes[1].text(
                0.5,
                max(y_count) + 1.2,
                f"{delta:+.1f} count error",
                ha="center",
                fontsize=8,
                fontweight="bold",
                color="#374151",
            )
        axes[1].set_xticks(x_positions, labels=labels)
        axes[1].set_ylim(0, max(otsu["mean_absolute_count_error"], y_count.max()) * 1.08)
        axes[1].set_ylabel("Mean absolute count error")
        axes[1].set_title("Counting does not improve")
        style_axis(axes[1], grid_axis="y")

        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.05))
        fig.suptitle("YOLO capacity diagnostic", y=1.14)
        fig.tight_layout()
        save_png(fig, figure_path)
        plt.close(fig)
        return

    axes[0].plot(
        x,
        yolo["mean_object_f1"],
        color=METHOD_PALETTE["yolo"],
        marker="o",
        markersize=5.8,
        markerfacecolor="white",
        markeredgewidth=1.3,
        linewidth=2.2,
        label="YOLO",
    )
    axes[0].axhline(cellpose["mean_object_f1"], color=METHOD_PALETTE["cellpose_cpsam"], linewidth=2.0, label="Cellpose-SAM")
    axes[0].axhline(otsu["mean_object_f1"], color=METHOD_PALETTE["otsu_watershed"], linewidth=1.8, linestyle="--", label="Otsu")
    axes[0].set_xticks(x, labels=xlabels)
    axes[0].set_ylim(0.60, 0.95)
    axes[0].set_ylabel("Mean object F1")
    axes[0].set_xlabel(x_label)
    axes[0].set_title("Instance matching")
    style_axis(axes[0], grid_axis="y")

    axes[1].plot(
        x,
        yolo["mean_absolute_count_error"],
        color=METHOD_PALETTE["yolo"],
        marker="s",
        markersize=5.8,
        markerfacecolor="white",
        markeredgewidth=1.3,
        linewidth=2.2,
        label="YOLO",
    )
    axes[1].axhline(cellpose["mean_absolute_count_error"], color=METHOD_PALETTE["cellpose_cpsam"], linewidth=2.0, label="Cellpose-SAM")
    axes[1].axhline(otsu["mean_absolute_count_error"], color=METHOD_PALETTE["otsu_watershed"], linewidth=1.8, linestyle="--", label="Otsu")
    axes[1].set_xticks(x, labels=xlabels)
    axes[1].set_ylabel("Mean absolute count error")
    axes[1].set_xlabel(x_label)
    axes[1].set_title("Count accuracy")
    style_axis(axes[1], grid_axis="y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(title, y=1.12)
    fig.tight_layout()
    save_png(fig, figure_path)
    plt.close(fig)


def redraw_protocol_ab_heldout_comparison() -> None:
    summary = pd.read_csv(RESULT_SUBDIRS["supervised"] / "yolo_capacity_diagnostic_val_comparison_summary.csv")
    method_order = [
        "cellpose_cpsam",
        "yolo11m_full_train_pool",
        "yolo11n_full_train_pool",
        "otsu_watershed",
    ]
    labels = {
        "cellpose_cpsam": "Cellpose-SAM",
        "yolo11m_full_train_pool": "YOLO11m",
        "yolo11n_full_train_pool": "YOLO11n",
        "otsu_watershed": "Otsu",
    }
    colors = {
        "cellpose_cpsam": METHOD_PALETTE["cellpose_cpsam"],
        "yolo11m_full_train_pool": METHOD_PALETTE["yolo"],
        "yolo11n_full_train_pool": METHOD_PALETTE["yolo"],
        "otsu_watershed": METHOD_PALETTE["otsu_watershed"],
    }
    markers = {
        "cellpose_cpsam": "o",
        "yolo11m_full_train_pool": "D",
        "yolo11n_full_train_pool": "s",
        "otsu_watershed": "s",
    }
    plot_data = (
        summary[summary["method"].isin(method_order)]
        .assign(method=lambda df: pd.Categorical(df["method"], method_order, ordered=True))
        .sort_values("method")
    )
    y = np.arange(len(plot_data))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharey=True)
    for ax, metric, title, xlim, direction in [
        (axes[0], "mean_object_f1", "Instance matching", (0.60, 0.95), "Higher is better"),
        (axes[1], "mean_absolute_count_error", "Count accuracy", (0, 22), "Lower is better"),
    ]:
        values = plot_data[metric].to_numpy()
        ax.hlines(y, 0 if metric != "mean_object_f1" else xlim[0], values, color="#d1d5db", linewidth=1.5, zorder=1)
        for ypos, row in zip(y, plot_data.itertuples(index=False)):
            method = row.method
            value = getattr(row, metric)
            ax.scatter(
                value,
                ypos,
                s=78,
                marker=markers[method],
                facecolor="white",
                edgecolor=colors[method],
                linewidth=1.8,
                zorder=3,
            )
            ax.annotate(
                f"{value:.3f}" if metric == "mean_object_f1" else f"{value:.1f}",
                xy=(value, ypos),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color="#374151",
                fontweight="bold" if method == "cellpose_cpsam" else "normal",
            )
        ax.set_title(f"{title}\n{direction}", fontsize=11)
        ax.set_xlim(*xlim)
        style_axis(ax, grid_axis="x")
    axes[0].set_yticks(y, labels=[labels[method] for method in plot_data["method"]])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean object F1")
    axes[1].set_xlabel("Mean absolute count error")

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=METHOD_PALETTE["cellpose_cpsam"], markeredgewidth=1.8, label="Protocol A zero-shot"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor=METHOD_PALETTE["yolo"], markeredgewidth=1.8, label="Protocol B supervised"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor=METHOD_PALETTE["otsu_watershed"], markeredgewidth=1.8, label="Classical lower bound"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.suptitle("Protocol A/B held-out validation comparison", y=1.14)
    fig.tight_layout()
    save_png(fig, FIGURES_DIR / "protocol_ab_heldout_val_comparison.png")
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
    redraw_clean_subset_count_agreement()
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
    redraw_protocol_ab_heldout_comparison()
    print("Redrew publication-style summary figures from existing CSV outputs.")


if __name__ == "__main__":
    main()
