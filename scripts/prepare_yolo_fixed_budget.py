#!/usr/bin/env python
"""Prepare the fixed-budget DSB2018 YOLO segmentation dataset."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellseg-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cellseg_robustness.data import image_path_from_dir, load_train_example, stage1_train_image_dirs
from cellseg_robustness.metrics import instance_ids
from cellseg_robustness.paths import FIGURES_DIR, RESULT_SUBDIRS, ensure_output_dirs

from prepare_yolo_label_smoke import (
    labels_for_mask,
    link_image,
    polygon_overlay,
    save_data_yaml,
    save_image_list,
)


OUTPUT_NAME = "yolo_fixed_budget"
DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_TRAIN_BUDGET = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--train-budget", type=int, default=DEFAULT_TRAIN_BUDGET)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        parser.error("--train-fraction must be between 0 and 1")
    if args.train_budget <= 0:
        parser.error("--train-budget must be positive")
    return args


def output_dir() -> Path:
    return RESULT_SUBDIRS["supervised"] / OUTPUT_NAME


def prepare_output_dir(overwrite: bool) -> Path:
    root = output_dir()
    existing = [
        root / "images.txt",
        root / "train.txt",
        root / "val.txt",
        root / "data.yaml",
        RESULT_SUBDIRS["supervised"] / f"{OUTPUT_NAME}_manifest.csv",
        RESULT_SUBDIRS["supervised"] / f"{OUTPUT_NAME}_split.csv",
        RESULT_SUBDIRS["supervised"] / f"{OUTPUT_NAME}_summary.csv",
        FIGURES_DIR / "supervised_yolo_fixed_budget_overlays.png",
    ]
    images_dir = root / "images"
    labels_dir = root / "labels"
    existing.extend(images_dir.glob("*.png") if images_dir.exists() else [])
    existing.extend(labels_dir.glob("*.txt") if labels_dir.exists() else [])
    existing = [path for path in existing if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing[:12])
        raise FileExistsError(f"Existing YOLO fixed-budget outputs found. Use --overwrite.\n{joined}")
    if overwrite:
        for path in existing:
            path.unlink()
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return root


def fixed_split(
    image_dirs: list[Path],
    train_fraction: float,
    train_budget: int,
) -> tuple[list[Path], list[Path], pd.DataFrame]:
    train_pool_count = int(np.floor(len(image_dirs) * train_fraction))
    train_pool_count = max(1, min(len(image_dirs) - 1, train_pool_count))
    train_pool = image_dirs[:train_pool_count]
    val_pool = image_dirs[train_pool_count:]
    if train_budget > len(train_pool):
        raise ValueError(f"train budget {train_budget} exceeds train pool size {len(train_pool)}")

    train_indices = np.linspace(0, len(train_pool) - 1, num=train_budget, dtype=int)
    train_index_set = {int(index) for index in train_indices}
    selected_train = [train_pool[index] for index in train_indices]

    split_rows: list[dict[str, object]] = []
    for index, image_dir in enumerate(image_dirs):
        if index < train_pool_count:
            pool = "train_pool"
            selected_split = "train" if index in train_index_set else "unused_train_pool"
            train_pool_index = index
            selection_rank = int(np.where(train_indices == index)[0][0]) if index in train_index_set else -1
        else:
            pool = "validation_pool"
            selected_split = "val"
            train_pool_index = -1
            selection_rank = -1
        split_rows.append(
            {
                "stable_index": index,
                "image_id": image_dir.name,
                "pool": pool,
                "selected_split": selected_split,
                "train_pool_index": train_pool_index,
                "selection_rank": selection_rank,
            }
        )

    return selected_train, val_pool, pd.DataFrame(split_rows)


def save_overlay_figure(examples: list[tuple[str, str, np.ndarray, np.ndarray]]) -> None:
    columns = 3
    rows = int(np.ceil(len(examples) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, 3.8 * rows))
    axes_array = np.atleast_1d(axes).ravel()
    for ax, (split, image_id, image, mask) in zip(axes_array, examples):
        ax.imshow(polygon_overlay(image, mask))
        ax.set_title(f"{split} {image_id[:10]}...")
        ax.axis("off")
    for ax in axes_array[len(examples) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "supervised_yolo_fixed_budget_overlays.png", dpi=160)
    plt.close(fig)


def build_summary(manifest: pd.DataFrame, split_table: pd.DataFrame) -> pd.DataFrame:
    split_counts = split_table["selected_split"].value_counts().to_dict()
    summary = (
        manifest.groupby("split", as_index=False)
        .agg(
            images=("image_id", "count"),
            true_instances=("true_instances", "sum"),
            converted_polygons=("converted_polygons", "sum"),
            dropped_instances=("dropped_instances", "sum"),
            fallback_polygons=("fallback_polygons", "sum"),
            min_polygon_points=("min_polygon_points", "min"),
            median_polygon_points=("median_polygon_points", "median"),
            max_polygon_points=("max_polygon_points", "max"),
        )
        .round(3)
    )
    total = pd.DataFrame(
        [
            {
                "split": "all_selected",
                "images": len(manifest),
                "true_instances": int(manifest["true_instances"].sum()),
                "converted_polygons": int(manifest["converted_polygons"].sum()),
                "dropped_instances": int(manifest["dropped_instances"].sum()),
                "fallback_polygons": int(manifest["fallback_polygons"].sum()),
                "min_polygon_points": int(manifest["min_polygon_points"].min()),
                "median_polygon_points": float(manifest["median_polygon_points"].median()),
                "max_polygon_points": int(manifest["max_polygon_points"].max()),
            },
        ]
    )
    summary = pd.concat([summary, total], ignore_index=True)
    summary["source_images"] = len(split_table)
    summary["train_pool_images"] = int((split_table["pool"] == "train_pool").sum())
    summary["validation_pool_images"] = int((split_table["pool"] == "validation_pool").sum())
    summary["unused_train_pool_images"] = int(split_counts.get("unused_train_pool", 0))
    return summary


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    root = prepare_output_dir(args.overwrite)
    images_dir = root / "images"
    labels_dir = root / "labels"

    all_image_dirs = stage1_train_image_dirs()
    train_dirs, val_dirs, split_table = fixed_split(
        all_image_dirs,
        train_fraction=args.train_fraction,
        train_budget=args.train_budget,
    )
    selected_dirs = [("train", image_dir) for image_dir in train_dirs] + [
        ("val", image_dir) for image_dir in val_dirs
    ]

    manifest_rows: list[dict[str, object]] = []
    image_paths: list[Path] = []
    train_image_paths: list[Path] = []
    val_image_paths: list[Path] = []
    overlay_examples: list[tuple[str, str, np.ndarray, np.ndarray]] = []

    for split, image_dir in selected_dirs:
        image_id, image, mask = load_train_example(image_dir)
        rows, polygon_stats = labels_for_mask(mask)
        label_path = labels_dir / f"{image_id}.txt"
        label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

        source_image_path = image_path_from_dir(image_dir)
        yolo_image_path = images_dir / f"{image_id}.png"
        link_image(source_image_path, yolo_image_path)
        image_paths.append(yolo_image_path)
        if split == "train":
            train_image_paths.append(yolo_image_path)
        else:
            val_image_paths.append(yolo_image_path)

        if len([example for example in overlay_examples if example[0] == split]) < 3:
            overlay_examples.append((split, image_id, image, mask))

        converted_count = sum(1 for stat in polygon_stats if stat["converted"])
        fallback_count = sum(1 for stat in polygon_stats if stat["fallback"])
        point_counts = [int(stat["points"]) for stat in polygon_stats if stat["converted"]]
        manifest_rows.append(
            {
                "split": split,
                "image_id": image_id,
                "source_image_path": source_image_path.as_posix(),
                "image_path": yolo_image_path.relative_to(REPO_ROOT).as_posix(),
                "label_path": label_path.relative_to(REPO_ROOT).as_posix(),
                "image_height": int(mask.shape[0]),
                "image_width": int(mask.shape[1]),
                "true_instances": int(len(instance_ids(mask))),
                "converted_polygons": int(converted_count),
                "dropped_instances": int(len(polygon_stats) - converted_count),
                "fallback_polygons": int(fallback_count),
                "min_polygon_points": min(point_counts) if point_counts else 0,
                "median_polygon_points": float(np.median(point_counts)) if point_counts else 0.0,
                "max_polygon_points": max(point_counts) if point_counts else 0,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    summary = build_summary(manifest, split_table)

    save_image_list(root / "images.txt", image_paths)
    save_image_list(root / "train.txt", train_image_paths)
    save_image_list(root / "val.txt", val_image_paths)
    save_data_yaml(root)
    save_overlay_figure(overlay_examples)
    manifest.to_csv(RESULT_SUBDIRS["supervised"] / f"{OUTPUT_NAME}_manifest.csv", index=False)
    split_table.to_csv(RESULT_SUBDIRS["supervised"] / f"{OUTPUT_NAME}_split.csv", index=False)
    summary.to_csv(RESULT_SUBDIRS["supervised"] / f"{OUTPUT_NAME}_summary.csv", index=False)

    if int(manifest["dropped_instances"].sum()) > 0:
        raise RuntimeError("Some instances were not converted to polygons")
    if int((manifest["converted_polygons"] == 0).sum()) > 0:
        raise RuntimeError("Some selected images have empty YOLO labels")
    if int((manifest["min_polygon_points"] < 3).sum()) > 0:
        raise RuntimeError("Some YOLO polygons have fewer than three points")
    if len(train_image_paths) != args.train_budget:
        raise RuntimeError("Unexpected train image count")
    if len(split_table) != len(all_image_dirs):
        raise RuntimeError("Split table does not cover all source images")

    print(f"Wrote {labels_dir}")
    print(f"Wrote {root / 'images.txt'}")
    print(f"Wrote {root / 'train.txt'}")
    print(f"Wrote {root / 'val.txt'}")
    print(f"Wrote {root / 'data.yaml'}")
    print(f"Wrote {RESULT_SUBDIRS['supervised'] / f'{OUTPUT_NAME}_manifest.csv'}")
    print(f"Wrote {RESULT_SUBDIRS['supervised'] / f'{OUTPUT_NAME}_split.csv'}")
    print(f"Wrote {RESULT_SUBDIRS['supervised'] / f'{OUTPUT_NAME}_summary.csv'}")
    print(f"Wrote {FIGURES_DIR / 'supervised_yolo_fixed_budget_overlays.png'}")


if __name__ == "__main__":
    main()
