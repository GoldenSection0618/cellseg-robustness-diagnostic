#!/usr/bin/env python
"""Audit optional cross-version Cellpose-family methods in the current env."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cellseg_robustness.paths import RESULT_SUBDIRS, ensure_output_dirs


def audit_segmentation_models() -> list[dict[str, object]]:
    from cellpose import models

    rows: list[dict[str, object]] = []
    candidates = [
        ("cellpose_default_cyto3", "cyto3"),
        ("cellpose_default_nuclei", "nuclei"),
        ("cellpose_default_transformer_cp3", "transformer_cp3"),
        ("cellpose_cpsam", "cpsam"),
    ]

    for method, requested_model in candidates:
        try:
            model = models.CellposeModel(gpu=False, pretrained_model=requested_model)
            loaded_model = Path(str(model.pretrained_model)).name
            runnable = loaded_model == requested_model
            if requested_model != "cpsam" and loaded_model == "cpsam":
                reason = "requested model resolves to cpsam in cellpose 4.1.1"
            else:
                reason = "available"
            rows.append(
                {
                    "method": method,
                    "api": "cellpose.models.CellposeModel",
                    "requested_model": requested_model,
                    "loaded_model": loaded_model,
                    "runnable_as_distinct_method": runnable,
                    "status": "available" if runnable else "alias_to_cpsam",
                    "reason": reason,
                }
            )
        except Exception as exc:  # pragma: no cover - records local env failures
            rows.append(
                {
                    "method": method,
                    "api": "cellpose.models.CellposeModel",
                    "requested_model": requested_model,
                    "loaded_model": "",
                    "runnable_as_distinct_method": False,
                    "status": "error",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    return rows


def audit_restoration_models() -> list[dict[str, object]]:
    from cellpose import denoise

    rows: list[dict[str, object]] = []
    candidates = [
        ("cellpose_oneclick_cyto3", "oneclick_cyto3"),
        ("cellpose_oneclick_nuclei", "oneclick_nuclei"),
    ]

    for method, restore_type in candidates:
        try:
            denoise.CellposeDenoiseModel(
                gpu=False,
                pretrained_model="cpsam",
                restore_type=restore_type,
                nchan=3,
            )
            rows.append(
                {
                    "method": method,
                    "api": "cellpose.denoise.CellposeDenoiseModel",
                    "requested_model": restore_type,
                    "loaded_model": restore_type,
                    "runnable_as_distinct_method": True,
                    "status": "available",
                    "reason": "available",
                }
            )
        except Exception as exc:  # pragma: no cover - records local env failures
            rows.append(
                {
                    "method": method,
                    "api": "cellpose.denoise.CellposeDenoiseModel",
                    "requested_model": restore_type,
                    "loaded_model": "",
                    "runnable_as_distinct_method": False,
                    "status": "error",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    return rows


def main() -> None:
    ensure_output_dirs()
    rows = audit_segmentation_models() + audit_restoration_models()
    audit = pd.DataFrame(rows)
    output_path = RESULT_SUBDIRS["baselines"] / "cellpose_method_availability.csv"
    audit.to_csv(output_path, index=False)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
