"""Shared summary-table helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


FAILURE_RATE_AGGREGATIONS = {
    "mean_missed_object_rate": ("missed_object_rate", "mean"),
    "mean_fp_per_true_instance": ("fp_per_true_instance", "mean"),
    "mean_count_bias": ("count_error", "mean"),
}


def add_failure_rate_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add per-image failure-rate diagnostics before grouped summaries."""
    output = metrics.copy()
    output["missed_object_rate"] = np.where(
        output["true_instances"] > 0,
        output["false_negatives"] / output["true_instances"],
        0.0,
    )
    output["fp_per_true_instance"] = np.where(
        output["true_instances"] > 0,
        output["false_positives"] / output["true_instances"],
        0.0,
    )
    return output
