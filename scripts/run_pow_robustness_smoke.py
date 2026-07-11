#!/usr/bin/env python
"""Run targeted robustness tests across completed PoW baselines."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellseg-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from cellpose import models
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cellseg_robustness.data import load_train_example, stage1_train_image_dirs
from cellseg_robustness.metrics import compute_instance_metrics
from cellseg_robustness.paths import FIGURES_DIR, RESULT_SUBDIRS, ensure_output_dirs
from cellseg_robustness.plot_style import save_png
from cellseg_robustness.perturbations import Perturbation, apply_perturbation, smoke_test_perturbations
from cellseg_robustness.summary import FAILURE_RATE_AGGREGATIONS, add_failure_rate_columns
from cellseg_robustness.visualization import overlay_truth_prediction
from run_cellpose_cpsam_baseline import predict_cellpose
from run_otsu_watershed_baseline import otsu_watershed_predict
from run_sam2_amg_baseline import (
    SAM2_CHECKPOINT,
    SAM2_CONFIG,
    predict_sam2,
)


IOU_THRESHOLD = 0.5
DEFAULT_LIMIT = 5
DEFAULT_OUTPUT_TAG = "smoke"
METHOD_ORDER = ["otsu_watershed", "cellpose_cpsam", "sam2_amg"]
METHOD_LABELS = {
    "otsu_watershed": "Otsu + watershed",
    "cellpose_cpsam": "Cellpose-SAM",
    "sam2_amg": "SAM2 AMG",
}
METHOD_COLORS = {
    "otsu_watershed": "#f97316",
    "cellpose_cpsam": "#0891b2",
    "sam2_amg": "#6366f1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of deterministic stage1_train images to evaluate.",
    )
    parser.add_argument(
        "--output-tag",
        default=DEFAULT_OUTPUT_TAG,
        choices=["smoke", "clean20", "full_train"],
        help="Output tag used in result and figure filenames.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=METHOD_ORDER,
        choices=METHOD_ORDER,
        help="Methods to run. Use this for staged full_train execution.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing metrics CSV and skip completed rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing outputs for this output tag before running.",
    )
    args = parser.parse_args()
    if args.output_tag == "clean20" and args.limit != 20:
        parser.error("--output-tag clean20 requires --limit 20")
    if args.output_tag == "full_train" and args.limit != DEFAULT_LIMIT:
        parser.error("--output-tag full_train uses all stage1_train images; omit --limit")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite cannot be used together")
    return args


def selected_image_dirs(limit: int, output_tag: str) -> list[Path]:
    """Use the same deterministic spread as the clean subset scripts."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    image_dirs = stage1_train_image_dirs()
    if output_tag == "full_train":
        return image_dirs
    if limit >= len(image_dirs):
        return image_dirs
    indices = np.linspace(0, len(image_dirs) - 1, num=limit, dtype=int)
    return [image_dirs[int(index)] for index in indices]


def build_predictors(methods: list[str]) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    """Initialize each completed PoW baseline once."""
    use_gpu = torch.cuda.is_available()
    predictors: dict[str, Callable[[np.ndarray], np.ndarray]] = {}

    def predict_otsu(image: np.ndarray) -> np.ndarray:
        return otsu_watershed_predict(image)

    if "otsu_watershed" in methods:
        predictors["otsu_watershed"] = predict_otsu

    if "cellpose_cpsam" in methods:
        cellpose_model = models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam")

        def predict_cpsam(image: np.ndarray) -> np.ndarray:
            return predict_cellpose(cellpose_model, image)

        predictors["cellpose_cpsam"] = predict_cpsam

    if "sam2_amg" in methods:
        if not SAM2_CHECKPOINT.exists():
            raise FileNotFoundError(f"Missing SAM2 checkpoint: {SAM2_CHECKPOINT}")
        sam2_device = "cuda" if use_gpu else "cpu"
        sam2_model = build_sam2(SAM2_CONFIG, str(SAM2_CHECKPOINT), device=sam2_device)
        sam2_generator = SAM2AutomaticMaskGenerator(
            sam2_model,
            points_per_side=24,
            points_per_batch=64,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.9,
            box_nms_thresh=0.7,
            crop_n_layers=0,
            min_mask_region_area=15,
            output_mode="binary_mask",
        )

        def predict_sam2_amg(image: np.ndarray) -> np.ndarray:
            return predict_sam2(sam2_generator, image)

        predictors["sam2_amg"] = predict_sam2_amg

    return predictors


def summarize(metrics: pd.DataFrame, perturbations: list[Perturbation]) -> pd.DataFrame:
    metrics = add_failure_rate_columns(metrics)
    summary = (
        metrics.groupby(["method", "perturbation"], as_index=False)
        .agg(
            images=("image_id", "count"),
            mean_object_f1=("object_f1", "mean"),
            mean_matched_iou=("mean_matched_iou", "mean"),
            mean_matched_dice=("mean_matched_dice", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_absolute_count_error=("absolute_count_error", "mean"),
            **FAILURE_RATE_AGGREGATIONS,
            median_latency_ms=("latency_ms", "median"),
            mean_latency_ms=("latency_ms", "mean"),
        )
        .round(4)
    )

    clean_f1 = (
        summary[summary["perturbation"] == "clean"]
        .set_index("method")["mean_object_f1"]
        .to_dict()
    )
    summary["clean_mean_object_f1"] = summary["method"].map(clean_f1)
    summary["absolute_object_f1_drop"] = (
        summary["clean_mean_object_f1"] - summary["mean_object_f1"]
    ).round(4)
    summary["relative_object_f1_drop"] = np.where(
        summary["clean_mean_object_f1"] > 0,
        summary["absolute_object_f1_drop"] / summary["clean_mean_object_f1"],
        0.0,
    ).round(4)
    summary["method_label"] = summary["method"].map(METHOD_LABELS)
    summary["method"] = pd.Categorical(summary["method"], categories=METHOD_ORDER, ordered=True)
    summary["perturbation"] = pd.Categorical(
        summary["perturbation"],
        categories=[perturbation.name for perturbation in perturbations],
        ordered=True,
    )
    return summary.sort_values(["method", "perturbation"]).reset_index(drop=True)


def present_methods(frame: pd.DataFrame) -> list[str]:
    available = set(frame["method"].astype(str).unique())
    return [method for method in METHOD_ORDER if method in available]


def output_stem(output_tag: str) -> str:
    return f"pow_baseline_robustness_{output_tag}"


def figure_stem(output_tag: str) -> str:
    return f"robustness_pow_{output_tag}"


def display_title(output_tag: str) -> str:
    if output_tag == "full_train":
        return "PoW Robustness Full Train"
    if output_tag == "clean20":
        return "PoW Robustness Clean20"
    return "PoW Robustness Smoke"


def split_label(output_tag: str) -> str:
    if output_tag == "full_train":
        return "stage1_train_full"
    if output_tag == "clean20":
        return "stage1_train_clean20_subset"
    return "stage1_train_tiny_subset"


def save_overlay_examples(
    examples: list[tuple[str, str, str, np.ndarray, np.ndarray, np.ndarray]],
    output_tag: str,
) -> None:
    columns = 3
    rows = int(np.ceil(len(examples) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, 3.8 * rows))
    axes_array = np.atleast_1d(axes).ravel()

    for ax, (method, image_id, perturbation, image, truth, prediction) in zip(axes_array, examples):
        ax.imshow(overlay_truth_prediction(image, truth, prediction))
        ax.set_title(f"{METHOD_LABELS[method]}\n{image_id[:8]}... {perturbation}")
        ax.axis("off")

    for ax in axes_array[len(examples) :]:
        ax.axis("off")

    fig.tight_layout()
    save_png(fig, FIGURES_DIR / f"{figure_stem(output_tag)}_overlay_examples.png")
    plt.close(fig)


def output_paths(output_tag: str) -> tuple[Path, Path]:
    stem = output_stem(output_tag)
    metrics_path = RESULT_SUBDIRS["robustness"] / f"{stem}_metrics.csv"
    summary_path = RESULT_SUBDIRS["robustness"] / f"{stem}_summary.csv"
    return metrics_path, summary_path


def figure_paths(output_tag: str) -> list[Path]:
    stem = figure_stem(output_tag)
    return [FIGURES_DIR / f"{stem}_overlay_examples.png"]


def prepare_existing_outputs(args: argparse.Namespace) -> pd.DataFrame:
    metrics_path, summary_path = output_paths(args.output_tag)
    existing_paths = [metrics_path, summary_path, *figure_paths(args.output_tag)]
    existing_paths = [path for path in existing_paths if path.exists()]

    if args.overwrite:
        for path in existing_paths:
            path.unlink()
        return pd.DataFrame()

    if existing_paths and not args.resume:
        joined = "\n".join(str(path) for path in existing_paths)
        raise FileExistsError(
            f"Existing outputs found for tag {args.output_tag}. Use --resume or --overwrite.\n{joined}"
        )

    if args.resume and metrics_path.exists():
        return pd.read_csv(metrics_path)
    return pd.DataFrame()


def completed_keys(existing_metrics: pd.DataFrame) -> set[tuple[str, str, str]]:
    if existing_metrics.empty:
        return set()
    return set(
        zip(
            existing_metrics["image_id"].astype(str),
            existing_metrics["method"].astype(str),
            existing_metrics["perturbation"].astype(str),
        )
    )


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    perturbations = smoke_test_perturbations()
    image_dirs = selected_image_dirs(args.limit, args.output_tag)
    existing_metrics = prepare_existing_outputs(args)
    done = completed_keys(existing_metrics)
    predictors = build_predictors(args.methods)
    autocast_enabled = torch.cuda.is_available()
    split = split_label(args.output_tag)

    rows: list[dict[str, object]] = []
    overlay_examples: list[tuple[str, str, str, np.ndarray, np.ndarray, np.ndarray]] = []

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        for image_dir in image_dirs:
            image_id, image, truth = load_train_example(image_dir)
            for perturbation in perturbations:
                perturbed = apply_perturbation(image, perturbation)
                for method in args.methods:
                    if (image_id, method, perturbation.name) in done:
                        continue
                    start = time.perf_counter()
                    prediction = predictors[method](perturbed)
                    latency_ms = (time.perf_counter() - start) * 1000
                    metrics = compute_instance_metrics(truth, prediction, iou_threshold=IOU_THRESHOLD)
                    rows.append(
                        {
                            "split": split,
                            "image_id": image_id,
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "perturbation": perturbation.name,
                            "perturbation_params": perturbation.params,
                            "iou_threshold": IOU_THRESHOLD,
                            "latency_ms": round(latency_ms, 3),
                            **asdict(metrics),
                        }
                    )

                    if image_id == image_dirs[0].name and len(overlay_examples) < 15:
                        overlay_examples.append((method, image_id, perturbation.name, perturbed, truth, prediction))

    metrics_df = pd.DataFrame(rows)
    if not existing_metrics.empty:
        metrics_df = pd.concat([existing_metrics, metrics_df], ignore_index=True)
    if metrics_df.empty:
        raise RuntimeError("No metrics were produced or loaded")
    summary_df = summarize(metrics_df, perturbations)

    metrics_path, summary_path = output_paths(args.output_tag)
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    save_overlay_examples(overlay_examples, args.output_tag)

    print(f"Wrote {metrics_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {FIGURES_DIR / f'{figure_stem(args.output_tag)}_overlay_examples.png'}")


if __name__ == "__main__":
    main()
