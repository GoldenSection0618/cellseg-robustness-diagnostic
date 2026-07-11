"""Shared publication-style plotting defaults for project figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


METHOD_PALETTE = {
    "otsu_watershed": "#6b7280",      # neutral: baseline thresholding
    "cellpose_cpsam": "#2563eb",      # signal: primary deep-learning method
    "sam2_amg": "#7c3aed",             # accent: alternative DL method
    "yolo": "#059669",                 # accent: supervised detector family
    "yolo11m": "#059669",
    "yolo11n": "#10b981",
    "warning": "#dc2626",              # directional: loss / error / bad
    "neutral": "#6b7280",
    "positive": "#16a34a",             # directional: gain / good
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
    ["#eff6ff", "#bfdbfe", "#60a5fa", "#3b82f6", "#1e40af"],
)
# Sequential warm for drops (drop is a loss, so red-family is justified)
DROP_CMAP = LinearSegmentedColormap.from_list(
    "cellseg_drop",
    ["#fff1f2", "#fecdd3", "#f87171", "#dc2626", "#991b1b"],
)
# Neutral gradient for generic categorical bars
BAR_CMAP = LinearSegmentedColormap.from_list(
    "cellseg_bar",
    ["#2563eb", "#7c3aed", "#f59e0b", "#dc2626"],
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
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 10,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
            "grid.color": "#e5e7eb",
            "grid.linewidth": 0.45,
        }
    )


def save_png(fig, path: Path | str) -> None:
    """Save a final PNG only, using project figure export defaults."""
    fig.savefig(path, dpi=600, bbox_inches="tight")


apply_figure_style()
