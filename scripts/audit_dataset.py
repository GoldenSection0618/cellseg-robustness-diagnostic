#!/usr/bin/env python
"""Audit the local DSB2018 dataset and generate first PoW outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cellseg_robustness.paths import DATA_ROOT, FIGURES_DIR, RESULT_SUBDIRS, ensure_output_dirs


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


def save_split_counts(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary["split"], summary["image_count"], color="#3b82f6")
    ax.set_title("DSB2018 Image Counts by Split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Images")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dataset_split_counts.png", dpi=160)
    plt.close(fig)


def save_instance_hist(inventory: pd.DataFrame) -> None:
    train = inventory[inventory["split"] == "stage1_train"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(train["mask_count"], bins=30, color="#10b981", edgecolor="white")
    ax.set_title("DSB2018 Stage 1 Train Instance Counts")
    ax.set_xlabel("PNG instance masks per image")
    ax.set_ylabel("Images")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dataset_train_instance_count_hist.png", dpi=160)
    plt.close(fig)


def save_size_scatter(inventory: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for split, frame in inventory.groupby("split"):
        ax.scatter(frame["width"], frame["height"], s=18, alpha=0.65, label=split)
    ax.set_title("DSB2018 Image Size Distribution")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dataset_image_size_scatter.png", dpi=160)
    plt.close(fig)


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

    save_split_counts(summary)
    save_instance_hist(inventory)
    save_size_scatter(inventory)

    print(f"Wrote {dataset_dir / 'dataset_inventory.csv'}")
    print(f"Wrote {dataset_dir / 'dataset_summary.csv'}")
    print(f"Wrote {FIGURES_DIR / 'dataset_split_counts.png'}")
    print(f"Wrote {FIGURES_DIR / 'dataset_train_instance_count_hist.png'}")
    print(f"Wrote {FIGURES_DIR / 'dataset_image_size_scatter.png'}")


if __name__ == "__main__":
    main()
