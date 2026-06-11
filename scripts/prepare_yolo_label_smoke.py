#!/usr/bin/env python
"""Convert a deterministic DSB2018 subset to YOLO segmentation labels."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellseg-matplotlib")

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import measure

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cellseg_robustness.data import image_path_from_dir, load_train_example, stage1_train_image_dirs
from cellseg_robustness.metrics import instance_ids
from cellseg_robustness.paths import FIGURES_DIR, RESULT_SUBDIRS, ensure_output_dirs
from cellseg_robustness.visualization import mask_boundaries


OUTPUT_NAME = "yolo_label_smoke"
DEFAULT_LIMIT = 20
DEFAULT_TRAIN_FRACTION = 0.8
CLASS_ID = 0
CLASS_NAME = "cell"


@dataclass(frozen=True)
class Polygon:
    points: np.ndarray
    source_area: int
    fallback: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if not 0.0 < args.train_fraction < 1.0:
        parser.error("--train-fraction must be between 0 and 1")
    return args


def selected_image_dirs(limit: int) -> list[Path]:
    image_dirs = stage1_train_image_dirs()
    if limit >= len(image_dirs):
        return image_dirs
    indices = np.linspace(0, len(image_dirs) - 1, num=limit, dtype=int)
    return [image_dirs[int(index)] for index in indices]


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
        RESULT_SUBDIRS["supervised"] / f"{OUTPUT_NAME}_summary.csv",
        FIGURES_DIR / "supervised_yolo_label_smoke_overlays.png",
    ]
    images_dir = root / "images"
    labels_dir = root / "labels"
    existing.extend(images_dir.glob("*.png") if images_dir.exists() else [])
    existing.extend(labels_dir.glob("*.txt") if labels_dir.exists() else [])
    existing = [path for path in existing if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing[:12])
        raise FileExistsError(f"Existing YOLO smoke outputs found. Use --overwrite.\n{joined}")
    if overwrite:
        for path in existing:
            path.unlink()
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return root


def split_names(image_dirs: list[Path], train_fraction: float) -> dict[str, str]:
    train_count = max(1, min(len(image_dirs) - 1, int(round(len(image_dirs) * train_fraction))))
    return {
        image_dir.name: "train" if index < train_count else "val"
        for index, image_dir in enumerate(image_dirs)
    }


def mask_to_polygon(mask: np.ndarray, tolerance: float = 1.0) -> Polygon | None:
    contours = measure.find_contours(mask.astype(np.uint8), level=0.5)
    if contours:
        contour = max(contours, key=len)
        # find_contours returns row, col; YOLO expects x, y.
        points = np.column_stack([contour[:, 1], contour[:, 0]])
        points = measure.approximate_polygon(points, tolerance=tolerance)
        if len(points) >= 3:
            return Polygon(points=points, source_area=int(mask.sum()))

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0 = max(float(xs.min()) - 0.5, 0.0)
    x1 = min(float(xs.max()) + 0.5, float(mask.shape[1] - 1))
    y0 = max(float(ys.min()) - 0.5, 0.0)
    y1 = min(float(ys.max()) + 0.5, float(mask.shape[0] - 1))
    if x0 == x1:
        x1 = min(x0 + 1.0, float(mask.shape[1] - 1))
    if y0 == y1:
        y1 = min(y0 + 1.0, float(mask.shape[0] - 1))
    points = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    return Polygon(points=points, source_area=int(mask.sum()), fallback=True)


def polygon_to_yolo_row(polygon: Polygon, width: int, height: int) -> str | None:
    points = polygon.points.copy()
    points[:, 0] = np.clip(points[:, 0] / width, 0.0, 1.0)
    points[:, 1] = np.clip(points[:, 1] / height, 0.0, 1.0)
    if len(points) < 3:
        return None
    coords = " ".join(f"{value:.6f}" for value in points.reshape(-1))
    return f"{CLASS_ID} {coords}"


def labels_for_mask(mask: np.ndarray) -> tuple[list[str], list[dict[str, object]]]:
    height, width = mask.shape
    rows: list[str] = []
    polygon_stats: list[dict[str, object]] = []
    for instance_id in instance_ids(mask):
        binary = mask == instance_id
        polygon = mask_to_polygon(binary)
        if polygon is None:
            polygon_stats.append(
                {
                    "instance_id": int(instance_id),
                    "source_area": int(binary.sum()),
                    "points": 0,
                    "converted": False,
                    "fallback": False,
                }
            )
            continue
        row = polygon_to_yolo_row(polygon, width=width, height=height)
        converted = row is not None
        if row is not None:
            rows.append(row)
        polygon_stats.append(
            {
                "instance_id": int(instance_id),
                "source_area": polygon.source_area,
                "points": int(len(polygon.points)),
                "converted": converted,
                "fallback": polygon.fallback,
            }
        )
    return rows, polygon_stats


def save_image_list(path: Path, image_paths: list[Path]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for path in image_paths:
            handle.write(f"{path.absolute().as_posix()}\n")


def save_data_yaml(root: Path) -> None:
    text = "\n".join(
        [
            f"path: {root.resolve()}",
            "train: train.txt",
            "val: val.txt",
            "names:",
            f"  0: {CLASS_NAME}",
            "",
        ]
    )
    (root / "data.yaml").write_text(text, encoding="utf-8")


def link_image(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    relative_source = os.path.relpath(source.resolve(), start=destination.parent.resolve())
    destination.symlink_to(relative_source)


def polygon_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image.astype(np.float32)
    else:
        gray = image[..., :3].mean(axis=2).astype(np.float32)
    max_value = float(gray.max()) if gray.size else 0.0
    if max_value > 1.0:
        gray /= 255.0 if max_value <= 255.0 else max_value
    rgb = np.repeat(np.clip(gray, 0.0, 1.0)[..., None], 3, axis=2)
    truth_edges = mask_boundaries(mask)
    rgb[truth_edges] = np.array([0.0, 1.0, 0.0])
    for instance_id in instance_ids(mask):
        polygon = mask_to_polygon(mask == instance_id)
        if polygon is None:
            continue
        coords = np.rint(polygon.points).astype(int)
        coords[:, 0] = np.clip(coords[:, 0], 0, mask.shape[1] - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, mask.shape[0] - 1)
        rgb[coords[:, 1], coords[:, 0]] = np.array([1.0, 0.0, 0.0])
    return rgb


def save_overlay_figure(examples: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    columns = 2
    rows = int(np.ceil(len(examples) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(9, 4.2 * rows))
    axes_array = np.atleast_1d(axes).ravel()
    for ax, (image_id, image, mask) in zip(axes_array, examples):
        ax.imshow(polygon_overlay(image, mask))
        ax.set_title(f"{image_id[:10]}...  mask=green polygon=red")
        ax.axis("off")
    for ax in axes_array[len(examples) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "supervised_yolo_label_smoke_overlays.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    root = prepare_output_dir(args.overwrite)
    images_dir = root / "images"
    labels_dir = root / "labels"
    image_dirs = selected_image_dirs(args.limit)
    splits = split_names(image_dirs, args.train_fraction)

    manifest_rows: list[dict[str, object]] = []
    image_paths: list[Path] = []
    train_image_paths: list[Path] = []
    val_image_paths: list[Path] = []
    overlay_examples: list[tuple[str, np.ndarray, np.ndarray]] = []

    for image_dir in image_dirs:
        image_id, image, mask = load_train_example(image_dir)
        rows, polygon_stats = labels_for_mask(mask)
        label_path = labels_dir / f"{image_id}.txt"
        label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        source_image_path = image_path_from_dir(image_dir)
        yolo_image_path = images_dir / f"{image_id}.png"
        link_image(source_image_path, yolo_image_path)
        image_paths.append(yolo_image_path)
        if splits[image_id] == "train":
            train_image_paths.append(yolo_image_path)
        else:
            val_image_paths.append(yolo_image_path)
        if len(overlay_examples) < 6:
            overlay_examples.append((image_id, image, mask))
        converted_count = sum(1 for stat in polygon_stats if stat["converted"])
        fallback_count = sum(1 for stat in polygon_stats if stat["fallback"])
        point_counts = [int(stat["points"]) for stat in polygon_stats if stat["converted"]]
        manifest_rows.append(
            {
                "split": splits[image_id],
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
                "split": "all",
                "images": len(manifest),
                "true_instances": int(manifest["true_instances"].sum()),
                "converted_polygons": int(manifest["converted_polygons"].sum()),
                "dropped_instances": int(manifest["dropped_instances"].sum()),
                "fallback_polygons": int(manifest["fallback_polygons"].sum()),
                "min_polygon_points": int(manifest["min_polygon_points"].min()),
                "median_polygon_points": float(manifest["median_polygon_points"].median()),
                "max_polygon_points": int(manifest["max_polygon_points"].max()),
            }
        ]
    )
    summary = pd.concat([summary, total], ignore_index=True)

    save_image_list(root / "images.txt", image_paths)
    save_image_list(root / "train.txt", train_image_paths)
    save_image_list(root / "val.txt", val_image_paths)
    save_data_yaml(root)
    save_overlay_figure(overlay_examples)
    manifest.to_csv(RESULT_SUBDIRS["supervised"] / f"{OUTPUT_NAME}_manifest.csv", index=False)
    summary.to_csv(RESULT_SUBDIRS["supervised"] / f"{OUTPUT_NAME}_summary.csv", index=False)

    if int(manifest["dropped_instances"].sum()) > 0:
        raise RuntimeError("Some instances were not converted to polygons")
    if int((manifest["converted_polygons"] == 0).sum()) > 0:
        raise RuntimeError("Some selected images have empty YOLO labels")
    if int((manifest["min_polygon_points"] < 3).sum()) > 0:
        raise RuntimeError("Some YOLO polygons have fewer than three points")

    print(f"Wrote {labels_dir}")
    print(f"Wrote {root / 'images.txt'}")
    print(f"Wrote {root / 'train.txt'}")
    print(f"Wrote {root / 'val.txt'}")
    print(f"Wrote {root / 'data.yaml'}")
    print(f"Wrote {RESULT_SUBDIRS['supervised'] / f'{OUTPUT_NAME}_manifest.csv'}")
    print(f"Wrote {RESULT_SUBDIRS['supervised'] / f'{OUTPUT_NAME}_summary.csv'}")
    print(f"Wrote {FIGURES_DIR / 'supervised_yolo_label_smoke_overlays.png'}")


if __name__ == "__main__":
    main()
