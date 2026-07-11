"""Controlled image perturbations for robustness smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import filters, transform, util


@dataclass(frozen=True)
class Perturbation:
    name: str
    params: dict[str, float | int | str]


def clip_like_float(image: np.ndarray) -> np.ndarray:
    """Return a float image clipped to [0, 1]."""
    image_float = util.img_as_float32(image)
    return np.clip(image_float, 0.0, 1.0)


def apply_perturbation(image: np.ndarray, perturbation: Perturbation) -> np.ndarray:
    """Apply a named perturbation to an image."""
    if perturbation.name == "clean":
        return image.copy()
    if perturbation.name == "gaussian_noise":
        return gaussian_noise(image, sigma=float(perturbation.params["sigma"]))
    if perturbation.name == "poisson_noise":
        return poisson_noise(image, peak=float(perturbation.params["peak"]))
    if perturbation.name == "gaussian_blur":
        return gaussian_blur(image, sigma=float(perturbation.params["sigma"]))
    if perturbation.name == "downsample_upsample":
        return downsample_upsample(image, scale=float(perturbation.params["scale"]))
    if perturbation.name == "intensity_scale":
        return intensity_scale(image, scale=float(perturbation.params["scale"]))
    if perturbation.name == "contrast_inversion":
        return contrast_inversion(image)
    raise ValueError(f"Unknown perturbation: {perturbation.name}")


def gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    image_float = clip_like_float(image)
    noisy = image_float + rng.normal(0.0, sigma, size=image_float.shape)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def poisson_noise(image: np.ndarray, peak: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    image_float = clip_like_float(image)
    noisy = rng.poisson(image_float * peak) / peak
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    image_float = clip_like_float(image)
    channel_axis = -1 if image_float.ndim == 3 else None
    blurred = filters.gaussian(image_float, sigma=sigma, channel_axis=channel_axis)
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


def downsample_upsample(image: np.ndarray, scale: float) -> np.ndarray:
    image_float = clip_like_float(image)
    height, width = image_float.shape[:2]
    small_shape = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))
    if image_float.ndim == 3:
        small_shape = (*small_shape, image_float.shape[2])
    small = transform.resize(
        image_float,
        small_shape,
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    )
    restored = transform.resize(
        small,
        image_float.shape,
        order=1,
        mode="reflect",
        anti_aliasing=False,
        preserve_range=True,
    )
    return np.clip(restored, 0.0, 1.0).astype(np.float32)


def intensity_scale(image: np.ndarray, scale: float) -> np.ndarray:
    image_float = clip_like_float(image)
    return np.clip(image_float * scale, 0.0, 1.0).astype(np.float32)


def contrast_inversion(image: np.ndarray) -> np.ndarray:
    image_float = clip_like_float(image)
    return (1.0 - image_float).astype(np.float32)


def smoke_test_perturbations() -> list[Perturbation]:
    """Small perturbation set for the first robustness smoke test."""
    return [
        Perturbation("clean", {}),
        Perturbation("gaussian_noise", {"sigma": 0.08}),
        Perturbation("poisson_noise", {"peak": 30.0}),
        Perturbation("gaussian_blur", {"sigma": 1.5}),
        Perturbation("downsample_upsample", {"scale": 0.5}),
        Perturbation("intensity_scale", {"scale": 0.6}),
        Perturbation("contrast_inversion", {}),
    ]
