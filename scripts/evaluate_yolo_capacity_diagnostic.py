#!/usr/bin/env python
"""Evaluate the YOLO capacity diagnostic checkpoint with repository metrics."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellseg-matplotlib")

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cellseg_robustness.data import load_instance_mask, mask_dir_from_dir
from cellseg_robustness.metrics import compute_instance_metrics
from cellseg_robustness.paths import FIGURES_DIR, RESULT_SUBDIRS, ensure_output_dirs
from cellseg_robustness.visualization import overlay_truth_prediction

from evaluate_yolo_tiny_train_smoke import image_dir_from_source_image, yolo_result_to_instances


IOU_THRESHOLD = 0.5
DEFAULT_CONF = 0.25
DEFAULT_IMGSZ = 512
MODEL_KEY = "yolo11m"
METADATA_PATH = RESULT_SUBDIRS["supervised"] / f"yolo_capacity_diagnostic_{MODEL_KEY}_train_metadata.csv"
METRICS_PATH = RESULT_SUBDIRS["supervised"] / f"yolo_capacity_diagnostic_{MODEL_KEY}_metrics.csv"
SUMMARY_PATH = RESULT_SUBDIRS["supervised"] / f"yolo_capacity_diagnostic_{MODEL_KEY}_eval_summary.csv"
FIGURE_PATH = FIGURES_DIR / f"supervised_yolo_capacity_diagnostic_{MODEL_KEY}_eval_overlays.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--overlay-count", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.conf < 0:
        parser.error("--conf must be non-negative")
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")
    if args.overlay_count <= 0:
        parser.error("--overlay-count must be positive")
    return args


def prepare_outputs(overwrite: bool) -> None:
    existing = [path for path in [METRICS_PATH, SUMMARY_PATH, FIGURE_PATH] if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Existing YOLO capacity evaluation outputs found. Use --overwrite.\n{joined}")
    if overwrite:
        for path in existing:
            path.unlink()


def select_overlay_examples(metrics_df: pd.DataFrame, max_count: int) -> list[str]:
    if len(metrics_df) <= max_count:
        return metrics_df["image_id"].tolist()
    ranked = metrics_df.sort_values("object_f1")
    positions = np.linspace(0, len(ranked) - 1, num=max_count, dtype=int)
    return ranked.iloc[positions]["image_id"].tolist()


def save_overlay_figure(examples: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]) -> None:
    columns = 4
    rows = int(np.ceil(len(examples) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(14, 3.4 * rows))
    axes_array = np.atleast_1d(axes).ravel()
    for ax, (image_id, image, truth, prediction) in zip(axes_array, examples):
        ax.imshow(overlay_truth_prediction(image, truth, prediction))
        ax.set_title(f"{image_id[:10]}...")
        ax.axis("off")
    for ax in axes_array[len(examples) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=160)
    plt.close(fig)


def summarize(metrics: pd.DataFrame, metadata: pd.Series, conf: float, imgsz: int) -> pd.DataFrame:
    no_prediction_rate = float((metrics["pred_instances"] == 0).mean())
    summary = {
        "protocol": "supervised_yolo_capacity_eval",
        "model_key": metadata["model_key"],
        "images": len(metrics),
        "train_images": int(metadata["train_images"]),
        "val_images": int(metadata["val_images"]),
        "model_weights": metadata["model_weights"],
        "checkpoint": metadata["best_checkpoint"],
        "epochs": int(metadata["epochs"]),
        "imgsz": int(metadata["imgsz"]),
        "batch": int(metadata["batch"]),
        "eval_imgsz": imgsz,
        "eval_conf": conf,
        "mean_object_f1": metrics["object_f1"].mean(),
        "median_object_f1": metrics["object_f1"].median(),
        "mean_precision": metrics["precision"].mean(),
        "mean_recall": metrics["recall"].mean(),
        "mean_matched_iou": metrics["mean_matched_iou"].mean(),
        "mean_matched_dice": metrics["mean_matched_dice"].mean(),
        "mean_absolute_count_error": metrics["absolute_count_error"].mean(),
        "median_absolute_count_error": metrics["absolute_count_error"].median(),
        "mean_true_instances": metrics["true_instances"].mean(),
        "mean_pred_instances": metrics["pred_instances"].mean(),
        "no_prediction_rate": no_prediction_rate,
        "median_latency_ms": metrics["latency_ms"].median(),
        "mean_latency_ms": metrics["latency_ms"].mean(),
    }
    return pd.DataFrame([summary]).round(4)


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    prepare_outputs(args.overwrite)
    metadata = pd.read_csv(METADATA_PATH).iloc[0]
    manifest = pd.read_csv(RESULT_SUBDIRS["supervised"] / "yolo_label_budget_diagnostic_manifest.csv")
    val_manifest = manifest[(manifest["budget"] == "full_train_pool") & (manifest["split"] == "val")].copy()
    checkpoint = REPO_ROOT / str(metadata["best_checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing YOLO checkpoint: {checkpoint}")

    model = YOLO(str(checkpoint))
    device = 0 if torch.cuda.is_available() else "cpu"
    rows: list[dict[str, object]] = []
    rendered: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for row in val_manifest.itertuples(index=False):
        source_image_path = Path(row.source_image_path)
        image = iio.imread(source_image_path)
        image_dir = image_dir_from_source_image(source_image_path)
        truth = load_instance_mask(mask_dir_from_dir(image_dir), image.shape[:2])
        start = time.perf_counter()
        result = model.predict(
            source=str(source_image_path),
            imgsz=args.imgsz,
            conf=args.conf,
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
                "model_key": MODEL_KEY,
                "image_id": row.image_id,
                "source_image_path": source_image_path.as_posix(),
                "checkpoint": checkpoint.relative_to(REPO_ROOT).as_posix(),
                "iou_threshold": IOU_THRESHOLD,
                "conf": args.conf,
                "imgsz": args.imgsz,
                "latency_ms": round(latency_ms, 3),
                **asdict(metrics),
            }
        )
        rendered[row.image_id] = (image, truth, prediction)

    metrics_df = pd.DataFrame(rows)
    summary_df = summarize(metrics_df, metadata, conf=args.conf, imgsz=args.imgsz)
    metrics_df.to_csv(METRICS_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    overlay_ids = select_overlay_examples(metrics_df, args.overlay_count)
    overlay_examples = [(image_id, *rendered[image_id]) for image_id in overlay_ids]
    save_overlay_figure(overlay_examples)

    print(f"Wrote {METRICS_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
