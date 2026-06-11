#!/usr/bin/env python
"""Evaluate the tiny YOLO training smoke with repository instance metrics."""

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

from cellseg_robustness.data import load_instance_mask, mask_dir_from_dir
from cellseg_robustness.metrics import compute_instance_metrics, relabel_sequential
from cellseg_robustness.paths import FIGURES_DIR, RESULT_SUBDIRS, ensure_output_dirs
from cellseg_robustness.visualization import overlay_truth_prediction


IOU_THRESHOLD = 0.5
DEFAULT_CONF = 0.001
DEFAULT_IMGSZ = 256
METRICS_PATH = RESULT_SUBDIRS["supervised"] / "yolo_tiny_train_smoke_metrics.csv"
SUMMARY_PATH = RESULT_SUBDIRS["supervised"] / "yolo_tiny_train_smoke_eval_summary.csv"
FIGURE_PATH = FIGURES_DIR / "supervised_yolo_tiny_train_smoke_overlays.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=RESULT_SUBDIRS["supervised"] / "yolo_tiny_train_smoke_metadata.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=RESULT_SUBDIRS["supervised"] / "yolo_label_smoke_manifest.csv",
    )
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_outputs(overwrite: bool) -> None:
    existing = [path for path in [METRICS_PATH, SUMMARY_PATH, FIGURE_PATH] if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Existing YOLO evaluation outputs found. Use --overwrite.\n{joined}")
    if overwrite:
        for path in existing:
            path.unlink()


def image_dir_from_source_image(source_image_path: Path) -> Path:
    return source_image_path.parents[1]


def yolo_result_to_instances(result: object, shape: tuple[int, int]) -> np.ndarray:
    labeled = np.zeros(shape, dtype=np.int32)
    occupied = np.zeros(shape, dtype=bool)
    masks = getattr(result, "masks", None)
    if masks is None or masks.data is None:
        return labeled

    mask_array = masks.data.detach().cpu().numpy()
    boxes = getattr(result, "boxes", None)
    if boxes is not None and boxes.conf is not None:
        confidences = boxes.conf.detach().cpu().numpy()
    else:
        confidences = np.ones(mask_array.shape[0], dtype=np.float32)
    order = np.argsort(-confidences)

    instance_id = 1
    for index in order:
        mask = mask_array[index] > 0.5
        if mask.shape != shape:
            # Ultralytics should return original image size here; fail loudly if that changes.
            raise ValueError(f"Unexpected YOLO mask shape {mask.shape}, expected {shape}")
        remaining = mask & ~occupied
        if int(remaining.sum()) == 0:
            continue
        labeled[remaining] = instance_id
        occupied |= remaining
        instance_id += 1
    return relabel_sequential(labeled)


def save_overlay_figure(examples: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]) -> None:
    columns = 2
    rows = int(np.ceil(len(examples) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(9, 4.2 * rows))
    axes_array = np.atleast_1d(axes).ravel()
    for ax, (image_id, image, truth, prediction) in zip(axes_array, examples):
        ax.imshow(overlay_truth_prediction(image, truth, prediction))
        ax.set_title(f"{image_id[:10]}...  truth=green pred=red")
        ax.axis("off")
    for ax in axes_array[len(examples) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=160)
    plt.close(fig)


def summarize(metrics: pd.DataFrame, metadata: pd.Series) -> pd.DataFrame:
    summary = {
        "protocol": "supervised_yolo_tiny_train_smoke_eval",
        "images": len(metrics),
        "train_images": int(metadata["train_images"]),
        "val_images": int(metadata["val_images"]),
        "model_weights": metadata["model_weights"],
        "checkpoint": metadata["best_checkpoint"],
        "epochs": int(metadata["epochs"]),
        "imgsz": int(metadata["imgsz"]),
        "batch": int(metadata["batch"]),
        "eval_conf": DEFAULT_CONF,
        "mean_object_f1": metrics["object_f1"].mean(),
        "mean_precision": metrics["precision"].mean(),
        "mean_recall": metrics["recall"].mean(),
        "mean_matched_iou": metrics["mean_matched_iou"].mean(),
        "mean_matched_dice": metrics["mean_matched_dice"].mean(),
        "mean_absolute_count_error": metrics["absolute_count_error"].mean(),
        "mean_true_instances": metrics["true_instances"].mean(),
        "mean_pred_instances": metrics["pred_instances"].mean(),
        "median_latency_ms": metrics["latency_ms"].median(),
        "mean_latency_ms": metrics["latency_ms"].mean(),
    }
    return pd.DataFrame([summary]).round(4)


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
    overlay_examples: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []

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
        overlay_examples.append((row.image_id, image, truth, prediction))

    metrics_df = pd.DataFrame(rows)
    summary_df = summarize(metrics_df, metadata)
    metrics_df.to_csv(METRICS_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    save_overlay_figure(overlay_examples)

    print(f"Wrote {METRICS_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
