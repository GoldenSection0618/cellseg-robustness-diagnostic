#!/usr/bin/env python
"""Run a small Cellpose-SAM input/parameter diagnostic on DSB2018 nuclei."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellseg-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from cellpose import models

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cellseg_robustness.data import image_to_gray_float, load_train_example, stage1_train_image_dirs
from cellseg_robustness.metrics import compute_instance_metrics, relabel_sequential
from cellseg_robustness.paths import FIGURES_DIR, RESULT_SUBDIRS, ensure_output_dirs


METHOD = "cellpose_cpsam"
IOU_THRESHOLD = 0.5
EXISTING_METRICS = RESULT_SUBDIRS["robustness"] / "pow_baseline_robustness_full_train_metrics.csv"
HELDOUT_MANIFEST = RESULT_SUBDIRS["supervised"] / "yolo_fixed_budget_manifest.csv"


@dataclass(frozen=True)
class CellposeConfig:
    config_id: str
    input_mode: str
    normalize: bool
    invert: bool
    diameter: float | None
    flow_threshold: float
    cellprob_threshold: float
    min_size: int = 15


CONFIGS = [
    CellposeConfig("rgb_baseline", "rgb", True, False, None, 0.4, 0.0),
    CellposeConfig("gray_mean", "gray_mean", True, False, None, 0.4, 0.0),
    CellposeConfig("gray_max", "gray_max", True, False, None, 0.4, 0.0),
    CellposeConfig("rgb_invert", "rgb", True, True, None, 0.4, 0.0),
    CellposeConfig("rgb_cellprob_-1", "rgb", True, False, None, 0.4, -1.0),
    CellposeConfig("rgb_cellprob_-2", "rgb", True, False, None, 0.4, -2.0),
    CellposeConfig("rgb_flow_0", "rgb", True, False, None, 0.0, 0.0),
    CellposeConfig("rgb_diameter_15", "rgb", True, False, 15.0, 0.4, 0.0),
    CellposeConfig("gray_diameter_15", "gray_mean", True, False, 15.0, 0.4, 0.0),
    CellposeConfig("rgb_diameter_30", "rgb", True, False, 30.0, 0.4, 0.0),
    CellposeConfig("rgb_diameter_60", "rgb", True, False, 60.0, 0.4, 0.0),
]
CONFIG_BY_ID = {config.config_id: config for config in CONFIGS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument(
        "--selection",
        choices=["diagnostic", "deterministic", "heldout_val"],
        default="diagnostic",
    )
    parser.add_argument("--configs", nargs="+", choices=sorted(CONFIG_BY_ID), default=[config.config_id for config in CONFIGS])
    parser.add_argument("--output-prefix", default="cellpose_cpsam_parameter_diagnostic")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def output_paths(prefix: str) -> tuple[Path, Path, Path]:
    metrics_path = RESULT_SUBDIRS["baselines"] / f"{prefix}_metrics.csv"
    summary_path = RESULT_SUBDIRS["baselines"] / f"{prefix}_summary.csv"
    figure_path = FIGURES_DIR / f"{prefix}_f1.png"
    return metrics_path, summary_path, figure_path


def prepare_outputs(overwrite: bool, paths: tuple[Path, Path, Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Existing Cellpose diagnostic outputs found. Use --overwrite.\n{joined}")
    if overwrite:
        for path in existing:
            path.unlink()


def deterministic_image_dirs(image_dirs: list[Path], limit: int) -> list[Path]:
    selected: list[Path] = []
    indices = np.linspace(0, len(image_dirs) - 1, num=limit, dtype=int)
    for index in indices:
        path = image_dirs[int(index)]
        if path not in selected:
            selected.append(path)
    for path in image_dirs:
        if len(selected) >= limit:
            break
        if path not in selected:
            selected.append(path)
    return selected[:limit]


def selected_image_dirs(limit: int, selection: str) -> list[Path]:
    image_dirs = stage1_train_image_dirs()
    by_id = {path.name: path for path in image_dirs}
    if selection == "deterministic":
        return deterministic_image_dirs(image_dirs, limit)
    if selection == "heldout_val":
        if not HELDOUT_MANIFEST.exists():
            raise FileNotFoundError(f"Missing held-out manifest: {HELDOUT_MANIFEST}")
        manifest = pd.read_csv(HELDOUT_MANIFEST)
        val_ids = manifest.loc[manifest["split"] == "val", "image_id"].tolist()
        return [by_id[image_id] for image_id in val_ids[:limit]]

    selected: list[Path] = []

    if EXISTING_METRICS.exists():
        metrics = pd.read_csv(EXISTING_METRICS)
        poor_cases = metrics[
            (metrics["method"] == METHOD)
            & (metrics["perturbation"] == "clean")
            & (metrics["recall"] < 0.8)
        ].sort_values(["recall", "object_f1", "false_negatives"])
        for image_id in poor_cases["image_id"].head(max(5, limit // 3)):
            path = by_id.get(image_id)
            if path is not None and path not in selected:
                selected.append(path)

    for path in deterministic_image_dirs(image_dirs, limit):
        if len(selected) >= limit:
            break
        if path not in selected:
            selected.append(path)
    for path in image_dirs:
        if len(selected) >= limit:
            break
        if path not in selected:
            selected.append(path)

    return selected[:limit]


def to_input(image: np.ndarray, mode: str) -> tuple[np.ndarray, int | None]:
    rgb = image[..., :3] if image.ndim == 3 else image
    if mode == "rgb":
        return rgb, -1 if rgb.ndim == 3 else None
    if mode == "gray_mean":
        return (image_to_gray_float(rgb) * 255).astype(np.uint8), None
    if mode == "gray_max":
        if rgb.ndim == 2:
            return rgb, None
        return rgb.max(axis=2).astype(rgb.dtype), None
    raise ValueError(f"Unknown input mode: {mode}")


def predict(model: models.CellposeModel, image: np.ndarray, config: CellposeConfig) -> np.ndarray:
    model_input, channel_axis = to_input(image, config.input_mode)
    prediction, *_ = model.eval(
        model_input,
        channel_axis=channel_axis,
        normalize=config.normalize,
        invert=config.invert,
        diameter=config.diameter,
        flow_threshold=config.flow_threshold,
        cellprob_threshold=config.cellprob_threshold,
        min_size=config.min_size,
    )
    return relabel_sequential(prediction.astype(np.int32))


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics.groupby("config_id", as_index=False)
        .agg(
            images=("image_id", "count"),
            mean_object_f1=("object_f1", "mean"),
            median_object_f1=("object_f1", "median"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_matched_iou=("mean_matched_iou", "mean"),
            mean_matched_dice=("mean_matched_dice", "mean"),
            mean_absolute_count_error=("absolute_count_error", "mean"),
            median_absolute_count_error=("absolute_count_error", "median"),
            mean_true_instances=("true_instances", "mean"),
            mean_pred_instances=("pred_instances", "mean"),
            no_prediction_rate=("pred_instances", lambda values: float((values == 0).mean())),
            median_latency_ms=("latency_ms", "median"),
            mean_latency_ms=("latency_ms", "mean"),
        )
        .reset_index(drop=True)
    )
    config_frame = pd.DataFrame([asdict(config) for config in CONFIGS])
    summary = summary.merge(config_frame, on="config_id", how="left")
    return summary.sort_values("mean_object_f1", ascending=False).round(4)


def save_figure(summary: pd.DataFrame, figure_path: Path) -> None:
    ordered = summary.sort_values("mean_object_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.barh(ordered["config_id"], ordered["mean_object_f1"], color="#0891b2")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean object F1")
    ax.set_title("Cellpose-SAM Parameter Diagnostic")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    metrics_path, summary_path, figure_path = output_paths(args.output_prefix)
    prepare_outputs(args.overwrite, (metrics_path, summary_path, figure_path))

    use_gpu = torch.cuda.is_available()
    model = models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam")
    rows: list[dict[str, object]] = []
    configs = [CONFIG_BY_ID[config_id] for config_id in args.configs]
    image_dirs = selected_image_dirs(args.limit, args.selection)

    for config in configs:
        for image_dir in image_dirs:
            image_id, image, truth = load_train_example(image_dir)
            start = time.perf_counter()
            prediction = predict(model, image, config)
            latency_ms = (time.perf_counter() - start) * 1000
            metrics = compute_instance_metrics(truth, prediction, iou_threshold=IOU_THRESHOLD)
            rows.append(
                {
                    "image_id": image_id,
                    "method": METHOD,
                    "config_id": config.config_id,
                    "selection": args.selection,
                    "iou_threshold": IOU_THRESHOLD,
                    "gpu": use_gpu,
                    "latency_ms": round(latency_ms, 3),
                    **asdict(config),
                    **asdict(metrics),
                }
            )
        print(f"Finished {config.config_id}", flush=True)

    metrics = pd.DataFrame(rows)
    summary = summarize(metrics)
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    save_figure(summary, figure_path)

    print(f"Wrote {metrics_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
