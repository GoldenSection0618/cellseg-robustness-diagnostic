#!/usr/bin/env python
"""Prepare nested YOLO label-budget diagnostic datasets."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellseg-matplotlib")

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cellseg_robustness.data import image_path_from_dir, load_train_example, stage1_train_image_dirs
from cellseg_robustness.metrics import instance_ids
from cellseg_robustness.paths import RESULT_SUBDIRS, ensure_output_dirs

from prepare_yolo_fixed_budget import DEFAULT_TRAIN_FRACTION
from prepare_yolo_label_smoke import labels_for_mask, link_image, save_data_yaml, save_image_list


OUTPUT_NAME = "yolo_label_budget_diagnostic"
FIXED_BUDGET_MANIFEST = RESULT_SUBDIRS["supervised"] / "yolo_fixed_budget_manifest.csv"
FIXED_BUDGET_SPLIT = RESULT_SUBDIRS["supervised"] / "yolo_fixed_budget_split.csv"
MANIFEST_PATH = RESULT_SUBDIRS["supervised"] / f"{OUTPUT_NAME}_manifest.csv"
SPLIT_PATH = RESULT_SUBDIRS["supervised"] / f"{OUTPUT_NAME}_split.csv"
SUMMARY_PATH = RESULT_SUBDIRS["supervised"] / f"{OUTPUT_NAME}_summary.csv"
NEW_BUDGETS = (250, "full_train_pool")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        parser.error("--train-fraction must be between 0 and 1")
    return args


def output_root() -> Path:
    return RESULT_SUBDIRS["supervised"] / OUTPUT_NAME


def prepare_output_dir(overwrite: bool) -> Path:
    root = output_root()
    existing = [MANIFEST_PATH, SPLIT_PATH, SUMMARY_PATH]
    existing.extend(root.rglob("*") if root.exists() else [])
    existing = [path for path in existing if path.exists() and path.is_file()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing[:12])
        raise FileExistsError(f"Existing YOLO label-budget outputs found. Use --overwrite.\n{joined}")
    if overwrite:
        for path in existing:
            path.unlink()
    root.mkdir(parents=True, exist_ok=True)
    return root


def train_pool_count(total_images: int, train_fraction: float) -> int:
    count = int(np.floor(total_images * train_fraction))
    return max(1, min(total_images - 1, count))


def budget_key(budget: int | str) -> str:
    return f"budget_{budget}" if isinstance(budget, int) else str(budget)


def load_fixed_budget_ids() -> tuple[list[str], list[str]]:
    fixed_manifest = pd.read_csv(FIXED_BUDGET_MANIFEST)
    fixed_split = pd.read_csv(FIXED_BUDGET_SPLIT)
    base_train_ids = fixed_manifest.loc[fixed_manifest["split"] == "train", "image_id"].tolist()
    val_ids = fixed_manifest.loc[fixed_manifest["split"] == "val", "image_id"].tolist()
    split_train_ids = fixed_split.loc[fixed_split["selected_split"] == "train", "image_id"].tolist()
    split_val_ids = fixed_split.loc[fixed_split["selected_split"] == "val", "image_id"].tolist()
    if base_train_ids != split_train_ids:
        raise RuntimeError("Fixed-budget train ids disagree between manifest and split table")
    if val_ids != split_val_ids:
        raise RuntimeError("Fixed-budget val ids disagree between manifest and split table")
    if len(base_train_ids) != 100:
        raise RuntimeError(f"Expected 100 fixed-budget train ids, found {len(base_train_ids)}")
    return base_train_ids, val_ids


def select_nested_budget_ids(train_pool_ids: list[str], base_train_ids: list[str], budget: int | str) -> list[str]:
    if budget == "full_train_pool":
        return list(train_pool_ids)
    if not isinstance(budget, int):
        raise TypeError(f"Unsupported budget {budget!r}")
    if budget <= len(base_train_ids):
        raise ValueError("New label-budget diagnostic budgets must exceed the fixed 100-image baseline")
    if budget > len(train_pool_ids):
        raise ValueError(f"Budget {budget} exceeds train pool size {len(train_pool_ids)}")

    selected = list(base_train_ids)
    selected_set = set(selected)
    remaining_ids = [image_id for image_id in train_pool_ids if image_id not in selected_set]
    needed = budget - len(selected)
    extra_indices = np.linspace(0, len(remaining_ids) - 1, num=needed, dtype=int)
    selected.extend(remaining_ids[int(index)] for index in extra_indices)
    if len(selected) != len(set(selected)):
        raise RuntimeError(f"Duplicate image ids selected for budget {budget}")
    return selected


def write_budget_dataset(
    root: Path,
    budget: int | str,
    image_dirs_by_id: dict[str, Path],
    train_ids: list[str],
    val_ids: list[str],
) -> pd.DataFrame:
    key = budget_key(budget)
    budget_dir = root / key
    images_dir = budget_dir / "images"
    labels_dir = budget_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []
    train_image_paths: list[Path] = []
    val_image_paths: list[Path] = []
    manifest_rows: list[dict[str, object]] = []

    selected = [("train", image_id) for image_id in train_ids] + [("val", image_id) for image_id in val_ids]
    for split, image_id in selected:
        image_dir = image_dirs_by_id[image_id]
        loaded_id, _image, mask = load_train_example(image_dir)
        if loaded_id != image_id:
            raise RuntimeError(f"Loaded unexpected image id {loaded_id} for {image_id}")
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

        converted_count = sum(1 for stat in polygon_stats if stat["converted"])
        fallback_count = sum(1 for stat in polygon_stats if stat["fallback"])
        point_counts = [int(stat["points"]) for stat in polygon_stats if stat["converted"]]
        manifest_rows.append(
            {
                "budget": key,
                "train_budget": len(train_ids),
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

    save_image_list(budget_dir / "images.txt", image_paths)
    save_image_list(budget_dir / "train.txt", train_image_paths)
    save_image_list(budget_dir / "val.txt", val_image_paths)
    save_data_yaml(budget_dir)
    return pd.DataFrame(manifest_rows)


def build_split_table(
    train_pool_ids: list[str],
    val_ids: list[str],
    budget_train_ids: dict[str, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stable_index, image_id in enumerate([*train_pool_ids, *val_ids]):
        pool = "train_pool" if stable_index < len(train_pool_ids) else "validation_pool"
        row: dict[str, object] = {
            "stable_index": stable_index,
            "image_id": image_id,
            "pool": pool,
        }
        for budget, train_ids in budget_train_ids.items():
            if pool == "validation_pool":
                selected_split = "val"
                selection_rank = -1
            elif image_id in train_ids:
                selected_split = "train"
                selection_rank = train_ids.index(image_id)
            else:
                selected_split = "unused_train_pool"
                selection_rank = -1
            row[f"{budget}_selected_split"] = selected_split
            row[f"{budget}_selection_rank"] = selection_rank
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(manifest: pd.DataFrame, split_table: pd.DataFrame) -> pd.DataFrame:
    summary = (
        manifest.groupby(["budget", "train_budget", "split"], as_index=False)
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
    totals = (
        manifest.groupby(["budget", "train_budget"], as_index=False)
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
    totals["split"] = "all_selected"
    summary = pd.concat([summary, totals[summary.columns]], ignore_index=True)
    summary["source_images"] = len(split_table)
    summary["train_pool_images"] = int((split_table["pool"] == "train_pool").sum())
    summary["validation_pool_images"] = int((split_table["pool"] == "validation_pool").sum())
    return summary.sort_values(["train_budget", "split"]).reset_index(drop=True)


def validate_outputs(
    manifest: pd.DataFrame,
    split_table: pd.DataFrame,
    base_train_ids: list[str],
    val_ids: list[str],
    budget_train_ids: dict[str, list[str]],
) -> None:
    if int(manifest["dropped_instances"].sum()) > 0:
        raise RuntimeError("Some instances were not converted to polygons")
    if int((manifest["converted_polygons"] == 0).sum()) > 0:
        raise RuntimeError("Some selected images have empty YOLO labels")
    if int((manifest["min_polygon_points"] < 3).sum()) > 0:
        raise RuntimeError("Some YOLO polygons have fewer than three points")

    previous_ids = set(base_train_ids)
    for budget, train_ids in budget_train_ids.items():
        train_set = set(train_ids)
        if not previous_ids.issubset(train_set):
            raise RuntimeError(f"{budget} does not contain the previous nested budget")
        previous_ids = train_set
        frame = manifest[manifest["budget"] == budget]
        if frame.loc[frame["split"] == "train", "image_id"].tolist() != train_ids:
            raise RuntimeError(f"{budget} manifest train ids do not match selected ids")
        if frame.loc[frame["split"] == "val", "image_id"].tolist() != val_ids:
            raise RuntimeError(f"{budget} manifest val ids do not match fixed validation ids")

    if len(split_table) != len(set(split_table["image_id"])):
        raise RuntimeError("Split table contains duplicate image ids")


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    root = prepare_output_dir(args.overwrite)

    all_image_dirs = stage1_train_image_dirs()
    image_dirs_by_id = {path.name: path for path in all_image_dirs}
    count = train_pool_count(len(all_image_dirs), args.train_fraction)
    train_pool_ids = [path.name for path in all_image_dirs[:count]]
    computed_val_ids = [path.name for path in all_image_dirs[count:]]
    base_train_ids, fixed_val_ids = load_fixed_budget_ids()
    if computed_val_ids != fixed_val_ids:
        raise RuntimeError("Computed validation ids do not match fixed-budget validation ids")

    budget_train_ids = {
        budget_key(budget): select_nested_budget_ids(train_pool_ids, base_train_ids, budget)
        for budget in NEW_BUDGETS
    }

    manifest_frames = [
        write_budget_dataset(root, budget, image_dirs_by_id, budget_train_ids[budget_key(budget)], fixed_val_ids)
        for budget in NEW_BUDGETS
    ]
    manifest = pd.concat(manifest_frames, ignore_index=True)
    split_table = build_split_table(train_pool_ids, fixed_val_ids, budget_train_ids)
    summary = build_summary(manifest, split_table)
    validate_outputs(manifest, split_table, base_train_ids, fixed_val_ids, budget_train_ids)

    manifest.to_csv(MANIFEST_PATH, index=False)
    split_table.to_csv(SPLIT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"Wrote {root}")
    print(f"Wrote {MANIFEST_PATH}")
    print(f"Wrote {SPLIT_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
