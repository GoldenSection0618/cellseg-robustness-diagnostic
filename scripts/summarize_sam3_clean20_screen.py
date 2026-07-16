#!/usr/bin/env python
"""Summarize the predeclared SAM3 clean20 expansion gate from existing metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cellseg_robustness.paths import RESULT_SUBDIRS, ensure_output_dirs


SAM3_METRICS = RESULT_SUBDIRS["baselines"] / "sam3_prompted_concept_clean_subset_metrics.csv"
BASELINE_METRICS = RESULT_SUBDIRS["baselines"] / "clean_subset_baseline_metrics_long.csv"
SUMMARY_PATH = RESULT_SUBDIRS["baselines"] / "sam3_prompted_concept_clean_subset_screen_summary.csv"
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20_260_716


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def paired_bootstrap_interval(deltas: np.ndarray) -> tuple[float, float]:
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    samples = generator.choice(deltas, size=(BOOTSTRAP_REPS, len(deltas)), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(lower), float(upper)


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    if SUMMARY_PATH.exists() and not args.overwrite:
        raise FileExistsError(f"Existing screen summary found. Use --overwrite.\n{SUMMARY_PATH}")

    sam3 = pd.read_csv(SAM3_METRICS)
    otsu = pd.read_csv(BASELINE_METRICS).loc[
        lambda frame: frame["method"].eq("otsu_watershed"), ["image_id", "object_f1"]
    ].rename(columns={"object_f1": "otsu_object_f1"})
    paired = sam3[["image_id", "object_f1"]].merge(otsu, on="image_id", how="left", validate="one_to_one")
    if len(sam3) != 20 or paired["otsu_object_f1"].isna().any():
        raise RuntimeError("SAM3 clean20 and Otsu clean-subset coverage must match exactly.")

    deltas = (paired["object_f1"] - paired["otsu_object_f1"]).to_numpy()
    ci_lower, ci_upper = paired_bootstrap_interval(deltas)
    completed = int(sam3["object_f1"].notna().sum())
    zero_prediction_rate = float(sam3["zero_prediction"].mean())
    failed_reasons: list[str] = []
    if completed < 20:
        failed_reasons.append("fewer_than_20_completed")
    if zero_prediction_rate >= 0.5:
        failed_reasons.append("zero_prediction_rate_at_least_0.5")
    if ci_upper < 0:
        failed_reasons.append("paired_otsu_f1_ci_upper_below_zero")

    summary = pd.DataFrame(
        [
            {
                "stage": "clean20",
                "method": "sam3_prompted_concept",
                "images_expected": 20,
                "images_completed": completed,
                "zero_prediction_images": int(sam3["zero_prediction"].sum()),
                "zero_prediction_rate": zero_prediction_rate,
                "mean_object_f1": float(sam3["object_f1"].mean()),
                "median_object_f1": float(sam3["object_f1"].median()),
                "mean_otsu_object_f1": float(paired["otsu_object_f1"].mean()),
                "mean_paired_f1_delta_vs_otsu": float(deltas.mean()),
                "paired_bootstrap_reps": BOOTSTRAP_REPS,
                "paired_bootstrap_seed": BOOTSTRAP_SEED,
                "paired_f1_delta_ci95_lower": ci_lower,
                "paired_f1_delta_ci95_upper": ci_upper,
                "expansion_gate_passed": not failed_reasons,
                "decision": "stop_before_full_train" if failed_reasons else "eligible_for_full_train",
                "failed_gate_reasons": ";".join(failed_reasons),
            }
        ]
    ).round(4)
    temporary = SUMMARY_PATH.with_suffix(".tmp.csv")
    summary.to_csv(temporary, index=False)
    temporary.replace(SUMMARY_PATH)
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
