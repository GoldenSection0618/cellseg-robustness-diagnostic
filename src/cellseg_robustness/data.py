"""Dataset loading helpers for DSB2018-style instance masks."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from .paths import DATA_ROOT


def stage1_train_image_dirs() -> list[Path]:
    """Return stage 1 train image directories in stable order."""
    return sorted(path for path in (DATA_ROOT / "stage1_train").iterdir() if path.is_dir())


def image_id_from_dir(image_dir: Path) -> str:
    return image_dir.name


def image_path_from_dir(image_dir: Path) -> Path:
    image_id = image_id_from_dir(image_dir)
    return image_dir / "images" / f"{image_id}.png"


def mask_dir_from_dir(image_dir: Path) -> Path:
    return image_dir / "masks"


def load_image(image_path: Path) -> np.ndarray:
    """Load an image as an ndarray."""
    return iio.imread(image_path)


def image_to_gray_float(image: np.ndarray) -> np.ndarray:
    """Convert grayscale/RGB/RGBA microscopy image to float grayscale in [0, 1]."""
    if image.ndim == 2:
        gray = image
    else:
        rgb = image[..., :3]
        gray = rgb.mean(axis=2)

    gray = gray.astype(np.float32)
    max_value = float(gray.max()) if gray.size else 0.0
    if max_value > 1.0:
        gray /= 255.0 if max_value <= 255.0 else max_value
    return gray


def load_instance_mask(mask_dir: Path, shape: tuple[int, int]) -> np.ndarray:
    """Load per-instance binary PNG masks into a labeled instance mask."""
    instance_mask = np.zeros(shape, dtype=np.int32)
    for instance_id, mask_path in enumerate(sorted(mask_dir.glob("*.png")), start=1):
        mask = iio.imread(mask_path)
        if mask.ndim > 2:
            mask = mask[..., 0]
        instance_mask[mask > 0] = instance_id
    return instance_mask


def load_train_example(image_dir: Path) -> tuple[str, np.ndarray, np.ndarray]:
    """Load a DSB2018 train image and its labeled ground-truth instance mask."""
    image_id = image_id_from_dir(image_dir)
    image = load_image(image_path_from_dir(image_dir))
    gt_mask = load_instance_mask(mask_dir_from_dir(image_dir), image.shape[:2])
    return image_id, image, gt_mask
