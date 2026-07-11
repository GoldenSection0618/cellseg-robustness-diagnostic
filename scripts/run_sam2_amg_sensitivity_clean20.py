#!/usr/bin/env python
"""Run SAM2 AMG parameter sensitivity on the deterministic clean20 subset."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cellseg_robustness.data import load_train_example, stage1_train_image_dirs
from cellseg_robustness.metrics import compute_instance_metrics
from cellseg_robustness.paths import RESULT_SUBDIRS, ensure_output_dirs
from cellseg_robustness.perturbations import Perturbation, apply_perturbation, smoke_test_perturbations
from run_sam2_amg_baseline import (
    IOU_THRESHOLD,
    SAM2_CHECKPOINT,
    SAM2_CONFIG,
    masks_to_instance_labels,
    sam2_rgb_input,
)


DEFAULT_LIMIT = 20
DEFAULT_CONFIG_ID = "default_current"
METHOD = "sam2_amg"
METHOD_LABEL = "SAM2 AMG"
OUTPUT_PREFIX = "sam2_amg_sensitivity_clean20"


@dataclass(frozen=True)
class Sam2AmgConfig:
    config_id: str
    sweep_group: str
    points_per_side: int
    pred_iou_thresh: float
    stability_score_thresh: float
    min_mask_region_area: int
    crop_n_layers: int
    box_nms_thresh: float

    def generator_kwargs(self) -> dict[str, Any]:
        return {
            "points_per_side": self.points_per_side,
            "points_per_batch": 16,
            "pred_iou_thresh": self.pred_iou_thresh,
            "stability_score_thresh": self.stability_score_thresh,
            "box_nms_thresh": self.box_nms_thresh,
            "crop_n_layers": self.crop_n_layers,
            "min_mask_region_area": self.min_mask_region_area,
            "output_mode": "binary_mask",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["clean_screen", "validation"],
        required=True,
        help="clean_screen runs single-parameter clean-only sweep; validation runs top configs across all perturbations.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of clean-screen configs to carry into validation, with the current default always included.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing outputs for the selected stage before running.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing metrics CSV and skip completed configurations.",
    )
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite cannot be used together")
    return args


def selected_image_dirs(limit: int = DEFAULT_LIMIT) -> list[Path]:
    image_dirs = stage1_train_image_dirs()
    indices = np.linspace(0, len(image_dirs) - 1, num=limit, dtype=int)
    return [image_dirs[int(index)] for index in indices]


def sensitivity_configs() -> list[Sam2AmgConfig]:
    base = Sam2AmgConfig(
        config_id=DEFAULT_CONFIG_ID,
        sweep_group="default",
        points_per_side=24,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.9,
        min_mask_region_area=15,
        crop_n_layers=0,
        box_nms_thresh=0.7,
    )
    configs = [base]

    for value in [16, 32, 64]:
        configs.append(
            Sam2AmgConfig(
                config_id=f"points_per_side_{value}",
                sweep_group="points_per_side",
                points_per_side=value,
                pred_iou_thresh=base.pred_iou_thresh,
                stability_score_thresh=base.stability_score_thresh,
                min_mask_region_area=base.min_mask_region_area,
                crop_n_layers=base.crop_n_layers,
                box_nms_thresh=base.box_nms_thresh,
            )
        )

    for value in [0.5, 0.7, 0.88]:
        configs.append(
            Sam2AmgConfig(
                config_id=f"pred_iou_thresh_{value:g}",
                sweep_group="pred_iou_thresh",
                points_per_side=base.points_per_side,
                pred_iou_thresh=value,
                stability_score_thresh=base.stability_score_thresh,
                min_mask_region_area=base.min_mask_region_area,
                crop_n_layers=base.crop_n_layers,
                box_nms_thresh=base.box_nms_thresh,
            )
        )

    for value in [0.5, 0.75, 0.95]:
        configs.append(
            Sam2AmgConfig(
                config_id=f"stability_score_thresh_{value:g}",
                sweep_group="stability_score_thresh",
                points_per_side=base.points_per_side,
                pred_iou_thresh=base.pred_iou_thresh,
                stability_score_thresh=value,
                min_mask_region_area=base.min_mask_region_area,
                crop_n_layers=base.crop_n_layers,
                box_nms_thresh=base.box_nms_thresh,
            )
        )

    for value in [0, 25, 100]:
        configs.append(
            Sam2AmgConfig(
                config_id=f"min_mask_region_area_{value}",
                sweep_group="min_mask_region_area",
                points_per_side=base.points_per_side,
                pred_iou_thresh=base.pred_iou_thresh,
                stability_score_thresh=base.stability_score_thresh,
                min_mask_region_area=value,
                crop_n_layers=base.crop_n_layers,
                box_nms_thresh=base.box_nms_thresh,
            )
        )

    configs.append(
        Sam2AmgConfig(
            config_id="crop_n_layers_1",
            sweep_group="crop_n_layers",
            points_per_side=base.points_per_side,
            pred_iou_thresh=base.pred_iou_thresh,
            stability_score_thresh=base.stability_score_thresh,
            min_mask_region_area=base.min_mask_region_area,
            crop_n_layers=1,
            box_nms_thresh=base.box_nms_thresh,
        )
    )

    for value in [0.5, 0.9]:
        configs.append(
            Sam2AmgConfig(
                config_id=f"box_nms_thresh_{value:g}",
                sweep_group="box_nms_thresh",
                points_per_side=base.points_per_side,
                pred_iou_thresh=base.pred_iou_thresh,
                stability_score_thresh=base.stability_score_thresh,
                min_mask_region_area=base.min_mask_region_area,
                crop_n_layers=base.crop_n_layers,
                box_nms_thresh=value,
            )
        )

    return configs


def configs_by_id() -> dict[str, Sam2AmgConfig]:
    return {config.config_id: config for config in sensitivity_configs()}


def stage_paths(stage: str) -> tuple[Path, Path]:
    metrics_path = RESULT_SUBDIRS["robustness"] / f"{OUTPUT_PREFIX}_{stage}_metrics.csv"
    summary_path = RESULT_SUBDIRS["robustness"] / f"{OUTPUT_PREFIX}_{stage}_summary.csv"
    return metrics_path, summary_path


def validation_failure_path() -> Path:
    return RESULT_SUBDIRS["robustness"] / f"{OUTPUT_PREFIX}_failure_cases.csv"


def failed_config_path(stage: str) -> Path:
    return RESULT_SUBDIRS["robustness"] / f"{OUTPUT_PREFIX}_{stage}_failed_configs.csv"


def prepare_outputs(stage: str, overwrite: bool, resume: bool) -> pd.DataFrame:
    metrics_path, summary_path = stage_paths(stage)
    paths = [summary_path, failed_config_path(stage)]
    if stage == "validation":
        paths.append(validation_failure_path())
    existing = [path for path in paths if path.exists()]
    if overwrite:
        for path in [metrics_path, *existing]:
            if path.exists():
                path.unlink()
        return pd.DataFrame()
    if resume:
        return pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    if existing or metrics_path.exists():
        all_existing = [path for path in [metrics_path, *existing] if path.exists()]
        joined = "\n".join(str(path) for path in all_existing)
        raise FileExistsError(f"Existing outputs found. Use --resume or --overwrite.\n{joined}")
    return pd.DataFrame()


def append_failed_config(stage: str, config: Sam2AmgConfig, error: BaseException) -> None:
    row = {
        "stage": stage,
        "config_id": config.config_id,
        "sweep_group": config.sweep_group,
        "points_per_side": config.points_per_side,
        "pred_iou_thresh": config.pred_iou_thresh,
        "stability_score_thresh": config.stability_score_thresh,
        "min_mask_region_area": config.min_mask_region_area,
        "crop_n_layers": config.crop_n_layers,
        "box_nms_thresh": config.box_nms_thresh,
        "error_type": type(error).__name__,
        "error_message": str(error).splitlines()[0],
    }
    path = failed_config_path(stage)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def failed_config_ids(stage: str) -> set[str]:
    path = failed_config_path(stage)
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["config_id"].astype(str).unique())


def is_cuda_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cudaerrormemoryallocation" in message


def try_empty_cuda_cache() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception as error:
        print(f"Skipping CUDA cache cleanup after error: {type(error).__name__}", flush=True)


def build_generator(config: Sam2AmgConfig, device: str) -> SAM2AutomaticMaskGenerator:
    if not SAM2_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing SAM2 checkpoint: {SAM2_CHECKPOINT}")
    model = build_sam2(SAM2_CONFIG, str(SAM2_CHECKPOINT), device=device)
    return SAM2AutomaticMaskGenerator(model, **config.generator_kwargs())


def predict_with_raw_count(
    generator: SAM2AutomaticMaskGenerator,
    image: np.ndarray,
) -> tuple[np.ndarray, int]:
    rgb = sam2_rgb_input(image)
    masks = generator.generate(rgb)
    prediction = masks_to_instance_labels(masks, image.shape[:2])
    return prediction, len(masks)


def row_for_prediction(
    *,
    image_id: str,
    config: Sam2AmgConfig,
    perturbation: Perturbation,
    perturbation_index: int,
    latency_ms: float,
    raw_amg_masks: int,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, object]:
    metrics = compute_instance_metrics(truth, prediction, iou_threshold=IOU_THRESHOLD)
    return {
        "split": "stage1_train_clean20_subset",
        "image_id": image_id,
        "method": METHOD,
        "method_label": METHOD_LABEL,
        "config_id": config.config_id,
        "sweep_group": config.sweep_group,
        "perturbation": perturbation.name,
        "perturbation_index": perturbation_index,
        "perturbation_params": perturbation.params,
        "iou_threshold": IOU_THRESHOLD,
        "sam2_config": SAM2_CONFIG,
        "checkpoint": SAM2_CHECKPOINT.relative_to(REPO_ROOT).as_posix(),
        "points_per_side": config.points_per_side,
        "pred_iou_thresh": config.pred_iou_thresh,
        "stability_score_thresh": config.stability_score_thresh,
        "min_mask_region_area": config.min_mask_region_area,
        "crop_n_layers": config.crop_n_layers,
        "box_nms_thresh": config.box_nms_thresh,
        "raw_amg_masks": raw_amg_masks,
        "zero_prediction": metrics.pred_instances == 0,
        "latency_ms": round(latency_ms, 3),
        **asdict(metrics),
    }


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics.groupby(["config_id", "sweep_group", "perturbation"], as_index=False)
        .agg(
            images=("image_id", "count"),
            mean_object_f1=("object_f1", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_matched_iou=("mean_matched_iou", "mean"),
            mean_matched_dice=("mean_matched_dice", "mean"),
            mean_true_instances=("true_instances", "mean"),
            mean_pred_instances=("pred_instances", "mean"),
            mean_raw_amg_masks=("raw_amg_masks", "mean"),
            mean_absolute_count_error=("absolute_count_error", "mean"),
            zero_prediction_rate=("zero_prediction", "mean"),
            median_latency_ms=("latency_ms", "median"),
            mean_latency_ms=("latency_ms", "mean"),
        )
        .round(4)
    )
    clean_f1 = (
        summary[summary["perturbation"] == "clean"]
        .set_index("config_id")["mean_object_f1"]
        .to_dict()
    )
    summary["clean_mean_object_f1"] = summary["config_id"].map(clean_f1)
    summary["absolute_object_f1_drop"] = (
        summary["clean_mean_object_f1"] - summary["mean_object_f1"]
    ).round(4)
    summary["relative_object_f1_drop"] = np.where(
        summary["clean_mean_object_f1"] > 0,
        summary["absolute_object_f1_drop"] / summary["clean_mean_object_f1"],
        0.0,
    ).round(4)
    return summary.sort_values(["config_id", "perturbation"]).reset_index(drop=True)


def clean_screen_configs() -> list[Sam2AmgConfig]:
    return sensitivity_configs()


def validation_configs(top_k: int) -> list[Sam2AmgConfig]:
    _, summary_path = stage_paths("clean_screen")
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing clean-screen summary: {summary_path}. Run --stage clean_screen first."
        )
    summary = pd.read_csv(summary_path)
    clean_summary = summary[summary["perturbation"] == "clean"].copy()
    clean_summary = clean_summary.sort_values(
        [
            "mean_object_f1",
            "mean_recall",
            "zero_prediction_rate",
            "mean_absolute_count_error",
        ],
        ascending=[False, False, True, True],
    )
    selected_ids = clean_summary["config_id"].head(top_k).astype(str).tolist()
    if DEFAULT_CONFIG_ID not in selected_ids:
        selected_ids.append(DEFAULT_CONFIG_ID)
    by_id = configs_by_id()
    return [by_id[config_id] for config_id in selected_ids]


def stage_perturbations(stage: str) -> list[Perturbation]:
    if stage == "clean_screen":
        return [Perturbation("clean", {})]
    return smoke_test_perturbations()


def save_failure_cases(metrics: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    clean = metrics[metrics["perturbation"] == "clean"][
        ["config_id", "image_id", "object_f1", "pred_instances", "absolute_count_error"]
    ].rename(
        columns={
            "object_f1": "clean_object_f1",
            "pred_instances": "clean_pred_instances",
            "absolute_count_error": "clean_absolute_count_error",
        }
    )
    deltas = metrics.merge(clean, on=["config_id", "image_id"], how="left")
    deltas["absolute_object_f1_drop"] = deltas["clean_object_f1"] - deltas["object_f1"]
    deltas["absolute_count_error_delta"] = (
        deltas["absolute_count_error"] - deltas["clean_absolute_count_error"]
    )

    non_clean = deltas[deltas["perturbation"] != "clean"].copy()
    non_clean = non_clean.sort_values(
        ["absolute_object_f1_drop", "zero_prediction", "absolute_count_error_delta"],
        ascending=[False, False, False],
    )
    selected = []
    for config_id in summary["config_id"].drop_duplicates():
        frame = non_clean[non_clean["config_id"] == config_id].head(8)
        selected.append(frame)
    cases = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    columns = [
        "config_id",
        "sweep_group",
        "image_id",
        "perturbation",
        "clean_object_f1",
        "object_f1",
        "absolute_object_f1_drop",
        "relative_object_f1_drop",
        "precision",
        "recall",
        "true_instances",
        "clean_pred_instances",
        "pred_instances",
        "raw_amg_masks",
        "zero_prediction",
        "clean_absolute_count_error",
        "absolute_count_error",
        "absolute_count_error_delta",
    ]
    cases["relative_object_f1_drop"] = np.where(
        cases["clean_object_f1"] > 0,
        cases["absolute_object_f1_drop"] / cases["clean_object_f1"],
        0.0,
    )
    cases[columns].round(4).to_csv(validation_failure_path(), index=False)
    return cases[columns]


def completed_config_ids(existing_metrics: pd.DataFrame) -> set[str]:
    if existing_metrics.empty:
        return set()
    return set(existing_metrics["config_id"].astype(str).unique())


def run_stage(
    stage: str,
    configs: list[Sam2AmgConfig],
    existing_metrics: pd.DataFrame,
    metrics_path: Path,
) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    perturbations = stage_perturbations(stage)
    image_dirs = selected_image_dirs()
    frames: list[pd.DataFrame] = []
    if not existing_metrics.empty:
        frames.append(existing_metrics)
    done_configs = completed_config_ids(existing_metrics)
    skipped_failed_configs = failed_config_ids(stage)
    autocast_enabled = device == "cuda"

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        for config in configs:
            if config.config_id in done_configs:
                print(f"Skipping completed config {config.config_id}", flush=True)
                continue
            if config.config_id in skipped_failed_configs:
                print(f"Skipping previously failed config {config.config_id}", flush=True)
                continue

            rows: list[dict[str, object]] = []
            try:
                generator = build_generator(config, device)
                for image_dir in image_dirs:
                    image_id, image, truth = load_train_example(image_dir)
                    for perturbation_index, perturbation in enumerate(perturbations):
                        perturbed = apply_perturbation(image, perturbation)
                        start = time.perf_counter()
                        prediction, raw_amg_masks = predict_with_raw_count(generator, perturbed)
                        latency_ms = (time.perf_counter() - start) * 1000
                        rows.append(
                            row_for_prediction(
                                image_id=image_id,
                                config=config,
                                perturbation=perturbation,
                                perturbation_index=perturbation_index,
                                latency_ms=latency_ms,
                                raw_amg_masks=raw_amg_masks,
                                truth=truth,
                                prediction=prediction,
                            )
                        )
            except Exception as error:
                if not is_cuda_oom(error):
                    raise
                append_failed_config(stage, config, error)
                print(f"Recorded CUDA OOM for config {config.config_id}", flush=True)
                try_empty_cuda_cache()
                continue

            config_frame = pd.DataFrame(rows)
            frames.append(config_frame)
            pd.concat(frames, ignore_index=True).to_csv(metrics_path, index=False)
            print(f"Wrote partial metrics through {config.config_id}", flush=True)
            try_empty_cuda_cache()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    existing_metrics = prepare_outputs(args.stage, args.overwrite, args.resume)

    if args.stage == "clean_screen":
        configs = clean_screen_configs()
    else:
        configs = validation_configs(args.top_k)

    metrics_path, summary_path = stage_paths(args.stage)
    metrics = run_stage(args.stage, configs, existing_metrics, metrics_path)
    if metrics.empty:
        raise RuntimeError("No metrics were produced or loaded")
    summary = summarize(metrics)
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)

    if args.stage == "validation":
        cases = save_failure_cases(metrics, summary)
        print(f"Wrote {validation_failure_path()} ({len(cases)} rows)")

    print(f"Wrote {metrics_path} ({len(metrics)} rows)")
    print(f"Wrote {summary_path} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
