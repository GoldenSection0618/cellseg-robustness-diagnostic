#!/usr/bin/env python
"""Evaluate fixed-budget YOLO checkpoint sensitivity to confidence thresholds."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cellseg_robustness.data import load_instance_mask, mask_dir_from_dir
from cellseg_robustness.metrics import compute_instance_metrics
from cellseg_robustness.paths import RESULT_SUBDIRS, ensure_output_dirs

from evaluate_yolo_tiny_train_smoke import image_dir_from_source_image, yolo_result_to_instances


IOU_THRESHOLD = 0.5
DEFAULT_IMGSZ = 512
DEFAULT_CONFS = (0.05, 0.10, 0.25, 0.40, 0.60)
METRICS_PATH = RESULT_SUBDIRS["supervised"] / "yolo_threshold_diagnostic_metrics.csv"
SUMMARY_PATH = RESULT_SUBDIRS["supervised"] / "yolo_threshold_diagnostic_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=RESULT_SUBDIRS["supervised"] / "yolo_fixed_budget_train_metadata.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=RESULT_SUBDIRS["supervised"] / "yolo_fixed_budget_manifest.csv",
    )
    parser.add_argument("--confs", type=float, nargs="+", default=list(DEFAULT_CONFS))
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if any(conf < 0 for conf in args.confs):
        parser.error("--confs values must be non-negative")
    if len(set(args.confs)) != len(args.confs):
        parser.error("--confs values must be unique")
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")
    args.confs = sorted(args.confs)
    return args


def prepare_outputs(overwrite: bool) -> None:
    existing = [path for path in [METRICS_PATH, SUMMARY_PATH] if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Existing YOLO threshold diagnostic outputs found. Use --overwrite.\n{joined}")
    if overwrite:
        for path in existing:
            path.unlink()


def summarize(metrics: pd.DataFrame, metadata: pd.Series, imgsz: int) -> pd.DataFrame:
    summary = (
        metrics.groupby("conf", as_index=False)
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
    summary.insert(0, "protocol", "supervised_yolo_threshold_diagnostic")
    summary.insert(2, "train_images", int(metadata["train_images"]))
    summary.insert(3, "val_images", int(metadata["val_images"]))
    summary.insert(4, "model_weights", metadata["model_weights"])
    summary.insert(5, "checkpoint", metadata["best_checkpoint"])
    summary.insert(6, "train_epochs", int(metadata["epochs"]))
    summary.insert(7, "train_imgsz", int(metadata["imgsz"]))
    summary.insert(8, "eval_imgsz", imgsz)
    summary["is_v1_conf"] = np.isclose(summary["conf"], 0.25)
    return summary.round(4)


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    prepare_outputs(args.overwrite)

    metadata = pd.read_csv(args.metadata).iloc[0]
    manifest = pd.read_csv(args.manifest)
    val_manifest = manifest[manifest["split"] == "val"].copy()
    checkpoint = REPO_ROOT / str(metadata["best_checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing YOLO checkpoint: {checkpoint}")

    model = YOLO(str(checkpoint))
    device = 0 if torch.cuda.is_available() else "cpu"
    rows: list[dict[str, object]] = []

    for conf in args.confs:
        for row in val_manifest.itertuples(index=False):
            source_image_path = Path(row.source_image_path)
            image = iio.imread(source_image_path)
            image_dir = image_dir_from_source_image(source_image_path)
            truth = load_instance_mask(mask_dir_from_dir(image_dir), image.shape[:2])
            start = time.perf_counter()
            result = model.predict(
                source=str(source_image_path),
                imgsz=args.imgsz,
                conf=conf,
                device=device,
                verbose=False,
                retina_masks=True,
            )[0]
            latency_ms = (time.perf_counter() - start) * 1000
            prediction = yolo_result_to_instances(result, image.shape[:2])
            metrics = compute_instance_metrics(truth, prediction, iou_threshold=IOU_THRESHOLD)
            rows.append(
                {
                    "split": "val",
                    "image_id": row.image_id,
                    "source_image_path": source_image_path.as_posix(),
                    "checkpoint": checkpoint.relative_to(REPO_ROOT).as_posix(),
                    "iou_threshold": IOU_THRESHOLD,
                    "conf": conf,
                    "imgsz": args.imgsz,
                    "latency_ms": round(latency_ms, 3),
                    **asdict(metrics),
                }
            )
        print(f"Finished conf={conf:g}", flush=True)

    metrics_df = pd.DataFrame(rows)
    summary_df = summarize(metrics_df, metadata, imgsz=args.imgsz)
    metrics_df.to_csv(METRICS_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print(f"Wrote {METRICS_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
