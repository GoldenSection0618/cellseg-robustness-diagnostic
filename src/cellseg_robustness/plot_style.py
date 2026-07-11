"""Shared publication-style plotting defaults for project figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


METHOD_PALETTE = {
    "otsu_watershed": "#e69f00",      # orange: classical baseline
    "cellpose_cpsam": "#7eb26d",      # green: main cell-specific model
    "sam2_amg": "#6f8fc9",            # blue: general foundation baseline
    "yolo": "#009e73",                # teal: supervised detector family
    "yolo11m": "#009e73",
    "yolo11n": "#56b4a9",
    "warning": "#d55e00",             # directional: loss / error / bad
    "neutral": "#6b7280",
    "positive": "#7eb26d",            # directional: gain / good
}

METRIC_PALETTE = {
    # Same metric family should share hue, vary shade to avoid rainbow effect
    "object_f1": "#2563eb",
    "mean_matched_iou": "#3b82f6",
    "mean_matched_dice": "#60a5fa",
    "precision": "#4b5563",
    "recall": "#9ca3af",
    "absolute_count_error": "#dc2626",
    "latency_ms": "#6b7280",
}

# Sequential blue for score heatmaps (no red-blue diverging semantic)
SCORE_CMAP = LinearSegmentedColormap.from_list(
    "cellseg_score",
    ["#f7fbff", "#dbeafe", "#93c5fd", "#4f83cc", "#1f4e8c"],
)
# Sequential warm for drops (drop is a loss, so red-family is justified)
DROP_CMAP = LinearSegmentedColormap.from_list(
    "cellseg_drop",
    ["#fff7ed", "#fed7aa", "#fb923c", "#d55e00", "#7f1d1d"],
)
# Neutral gradient for generic categorical bars
BAR_CMAP = LinearSegmentedColormap.from_list(
    "cellseg_bar",
    ["#6f8fc9", "#7eb26d", "#e69f00", "#d55e00"],
)


def gradient_colors(count: int, cmap=BAR_CMAP) -> list[tuple[float, float, float, float]]:
    """Return distinct colors for categorical bars without using one flat color."""
    if count <= 0:
        return []
    if count == 1:
        return [cmap(0.5)]
    return [cmap(value) for value in np.linspace(0.12, 0.88, count)]


def apply_figure_style() -> None:
    """Apply a restrained, manuscript-oriented matplotlib style."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
            "grid.color": "#e5e7eb",
            "grid.linestyle": "--",
            "grid.linewidth": 0.65,
            "axes.grid": True,
            "axes.axisbelow": True,
        }
    )


def save_png(fig, path: Path | str) -> None:
    """Save a final PNG only, using project figure export defaults."""
    fig.savefig(path, dpi=600, bbox_inches="tight")


apply_figure_style()
