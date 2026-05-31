"""Visualization helpers for segmentation diagnostics."""

from __future__ import annotations

import numpy as np

from .data import image_to_gray_float


def mask_boundaries(mask: np.ndarray) -> np.ndarray:
    """Return a boolean map of instance boundaries."""
    boundaries = np.zeros(mask.shape, dtype=bool)
    boundaries[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundaries[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundaries[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundaries[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return boundaries & (mask > 0)


def overlay_truth_prediction(
    image: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    """Create RGB overlay with truth boundaries in green and prediction in red."""
    gray = image_to_gray_float(image)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    truth_edges = mask_boundaries(truth)
    pred_edges = mask_boundaries(prediction)

    rgb[truth_edges] = np.array([0.0, 1.0, 0.0])
    rgb[pred_edges] = np.array([1.0, 0.0, 0.0])
    both = truth_edges & pred_edges
    rgb[both] = np.array([1.0, 1.0, 0.0])
    return np.clip(rgb, 0.0, 1.0)
