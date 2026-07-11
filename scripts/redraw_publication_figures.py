#!/usr/bin/env python
"""Redraw publication figures from existing CSV outputs."""

from __future__ import annotations

try:
    from . import _redraw_publication_figures_core as _core
    from ._publication_overrides import (
        redraw_baseline_clean_subset,
        redraw_clean20_diagnostics,
        redraw_full_train_diagnostics,
        redraw_robustness_summary,
        redraw_sam2_sensitivity,
    )
except ImportError:
    import _redraw_publication_figures_core as _core
    from _publication_overrides import (
        redraw_baseline_clean_subset,
        redraw_clean20_diagnostics,
        redraw_full_train_diagnostics,
        redraw_robustness_summary,
        redraw_sam2_sensitivity,
    )

for _name in dir(_core):
    if not _name.startswith("_") and _name not in globals():
        globals()[_name] = getattr(_core, _name)


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
