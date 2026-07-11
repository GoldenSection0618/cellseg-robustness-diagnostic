#!/usr/bin/env python
"""Audit the local DSB2018 dataset and generate first PoW outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v3 as iio
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cellseg_robustness.paths import DATA_ROOT, RESULT_SUBDIRS, ensure_output_dirs


def image_metadata(image_path: Path) -> dict[str, object]:
    image = iio.imread(image_path)
    if image.ndim == 2:
        height, width = image.shape
        channels = 1
    else:
        height, width, channels = image.shape
    return {
        "image_path": str(image_path.relative_to(REPO_ROOT)),
        "height": int(height),
        "width": int(width),
        "channels": int(channels),
        "dtype": str(image.dtype),
    }


def stage1_train_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for image_dir in sorted((DATA_ROOT / "stage1_train").iterdir()):
        if not image_dir.is_dir():
            continue
        image_id = image_dir.name
        image_path = image_dir / "images" / f"{image_id}.png"
        mask_dir = image_dir / "masks"
        masks = sorted(mask_dir.glob("*.png")) if mask_dir.exists() else []
        rows.append(
            {
                "split": "stage1_train",
                "image_id": image_id,
                "has_png_masks": bool(masks),
                "mask_count": len(masks),
                "solution_rle_count": 0,
                **image_metadata(image_path),
            }
        )
    return rows


def image_only_rows(split: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    split_dir = DATA_ROOT / split
    for image_dir in sorted(split_dir.iterdir()):
        if not image_dir.is_dir():
            continue
        image_id = image_dir.name
        image_path = image_dir / "images" / f"{image_id}.png"
        rows.append(
            {
                "split": split,
                "image_id": image_id,
                "has_png_masks": False,
                "mask_count": 0,
                "solution_rle_count": 0,
                **image_metadata(image_path),
            }
        )
    return rows


def add_stage1_solution_counts(inventory: pd.DataFrame) -> pd.DataFrame:
    solution_path = DATA_ROOT / "solutions" / "stage1_solution.csv"
    if not solution_path.exists():
        inventory["has_solution_rle"] = False
        return inventory

    solution = pd.read_csv(solution_path)
    counts = solution.groupby("ImageId").size().rename("solution_rle_count")
    inventory = inventory.drop(columns=["solution_rle_count"]).merge(
        counts,
        how="left",
        left_on="image_id",
        right_index=True,
    )
    inventory["solution_rle_count"] = inventory["solution_rle_count"].fillna(0).astype(int)
    inventory["has_solution_rle"] = inventory["solution_rle_count"] > 0
    return inventory


def make_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    grouped = inventory.groupby("split", dropna=False)
    summary = grouped.agg(
        image_count=("image_id", "count"),
        total_png_masks=("mask_count", "sum"),
        total_solution_rles=("solution_rle_count", "sum"),
        min_height=("height", "min"),
        max_height=("height", "max"),
        min_width=("width", "min"),
        max_width=("width", "max"),
        min_channels=("channels", "min"),
        max_channels=("channels", "max"),
    )
    summary["mean_png_masks_per_image"] = grouped["mask_count"].mean().round(3)
    summary["mean_solution_rles_per_image"] = grouped["solution_rle_count"].mean().round(3)
    return summary.reset_index()


def main() -> None:
    ensure_output_dirs()
    rows = stage1_train_rows()
    rows.extend(image_only_rows("stage1_test"))
    rows.extend(image_only_rows("stage2_test_final"))

    inventory = pd.DataFrame(rows)
    inventory = add_stage1_solution_counts(inventory)
    inventory = inventory[
        [
            "split",
            "image_id",
            "image_path",
            "height",
            "width",
            "channels",
            "dtype",
            "has_png_masks",
            "mask_count",
            "has_solution_rle",
            "solution_rle_count",
        ]
    ]
    summary = make_summary(inventory)

    dataset_dir = RESULT_SUBDIRS["dataset"]
    inventory.to_csv(dataset_dir / "dataset_inventory.csv", index=False)
    summary.to_csv(dataset_dir / "dataset_summary.csv", index=False)

    print(f"Wrote {dataset_dir / 'dataset_inventory.csv'}")
    print(f"Wrote {dataset_dir / 'dataset_summary.csv'}")


if __name__ == "__main__":
    main()
