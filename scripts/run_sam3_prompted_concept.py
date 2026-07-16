#!/usr/bin/env python
"""Run the fixed-concept SAM3 zero-shot Protocol A extension."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cellseg_robustness.data import image_to_gray_float, load_train_example, stage1_train_image_dirs
from cellseg_robustness.metrics import compute_instance_metrics, relabel_sequential
from cellseg_robustness.paths import FIGURES_DIR, RESULT_SUBDIRS, ensure_output_dirs
from cellseg_robustness.perturbations import Perturbation, apply_perturbation, smoke_test_perturbations
from cellseg_robustness.plot_style import save_png
from cellseg_robustness.visualization import overlay_truth_prediction


METHOD = "sam3_prompted_concept"
METHOD_LABEL = 'SAM3 ("nucleus")'
PROMPT = "nucleus"
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
RESOLUTION = 1008
CODE_COMMIT = "46957e47805eaa273f4aa7bbbd25a88bca9108ce"
CHECKPOINT = REPO_ROOT / "data" / "checkpoints" / "sam3.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "clean20", "full_train"], required=True)
    parser.add_argument(
        "--conditions",
        choices=["clean", "all"],
        default="clean",
        help="Use all existing robustness perturbations only after the full-clean gate.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume cannot be used together")
    if args.stage in {"smoke", "clean20"} and args.conditions != "clean":
        parser.error(f"--stage {args.stage} only supports --conditions clean")
    return args


def selected_image_dirs(stage: str) -> list[Path]:
    image_dirs = stage1_train_image_dirs()
    if stage == "full_train":
        return image_dirs
    indices = np.linspace(0, len(image_dirs) - 1, num=20, dtype=int)
    clean20 = [image_dirs[int(index)] for index in indices]
    return clean20[:1] if stage == "smoke" else clean20


def output_paths(stage: str) -> tuple[Path, Path]:
    stems = {
        "smoke": "sam3_prompted_concept_smoke",
        "clean20": "sam3_prompted_concept_clean_subset",
        "full_train": "sam3_prompted_concept_full_train",
    }
    stem = stems[stage]
    return (
        RESULT_SUBDIRS["baselines"] / f"{stem}_metrics.csv"
        if stage != "full_train"
        else RESULT_SUBDIRS["robustness"] / f"{stem}_metrics.csv",
        FIGURES_DIR / f"{stem}_overlay_examples.png",
    )


def split_label(stage: str) -> str:
    return {
        "smoke": "stage1_train_clean20_smoke",
        "clean20": "stage1_train_clean20_subset",
        "full_train": "stage1_train_full",
    }[stage]


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Preserve RGB inputs and replicate grayscale inputs for SAM3's RGB processor."""
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        gray = image_to_gray_float(image[..., 0] if image.ndim == 3 else image)
        return np.repeat((gray * 255).round().astype(np.uint8)[..., None], 3, axis=2)

    rgb = image[..., :3]
    if rgb.dtype == np.uint8:
        return rgb
    rgb = rgb.astype(np.float32)
    maximum = float(rgb.max()) if rgb.size else 0.0
    if maximum > 1.0:
        rgb /= 255.0 if maximum <= 255.0 else maximum
    return (np.clip(rgb, 0.0, 1.0) * 255).round().astype(np.uint8)


def masks_to_instance_labels(
    masks: torch.Tensor,
    probabilities: torch.Tensor,
    shape: tuple[int, int],
) -> tuple[np.ndarray, int, float]:
    """Resolve overlapping SAM3 query masks without target-domain post-processing."""
    if masks.numel() == 0 or masks.shape[0] == 0:
        return np.zeros(shape, dtype=np.int32), 0, 0.0

    active = masks.detach().to(device="cpu", dtype=torch.bool).numpy()
    scores = probabilities.detach().to(device="cpu", dtype=torch.float32).numpy()
    raw_count = int(active.shape[0])
    coverage = active.sum(axis=0)
    overlap_fraction = float((coverage > 1).mean())
    valid = coverage > 0
    labels = np.zeros(shape, dtype=np.int32)
    masked_scores = np.where(active, scores, -np.inf)
    winners = masked_scores.argmax(axis=0)
    labels[valid] = winners[valid] + 1
    return relabel_sequential(labels), raw_count, overlap_fraction


def build_processor() -> Sam3Processor:
    if not torch.cuda.is_available():
        raise RuntimeError("SAM3 Protocol A requires a CUDA GPU")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing SAM3 checkpoint: {CHECKPOINT}. Download approved facebook/sam3 sam3.pt first."
        )
    model = build_sam3_image_model(
        device="cuda",
        checkpoint_path=str(CHECKPOINT),
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )
    return Sam3Processor(
        model,
        resolution=RESOLUTION,
        device="cuda",
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )


def predict(processor: Sam3Processor, image: np.ndarray) -> tuple[np.ndarray, int, float, float, float]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    pil_image = Image.fromarray(as_rgb_uint8(image), mode="RGB")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(pil_image)
        state = processor.set_text_prompt(PROMPT, state)
    prediction, raw_count, overlap_fraction = masks_to_instance_labels(
        state["masks"], state["masks_logits"], image.shape[:2]
    )
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    return prediction, raw_count, overlap_fraction, latency_ms, peak_memory_mb


def selected_perturbations(conditions: str) -> list[Perturbation]:
    perturbations = smoke_test_perturbations()
    return perturbations if conditions == "all" else [perturbations[0]]


def prepare_outputs(metrics_path: Path, overlay_path: Path, args: argparse.Namespace) -> pd.DataFrame:
    existing_paths = [path for path in [metrics_path, overlay_path] if path.exists()]
    if args.overwrite:
        for path in existing_paths:
            path.unlink()
        return pd.DataFrame()
    if existing_paths and not args.resume:
        joined = "\n".join(str(path) for path in existing_paths)
        raise FileExistsError(f"Existing SAM3 outputs found. Use --resume or --overwrite.\n{joined}")
    return pd.read_csv(metrics_path) if args.resume and metrics_path.exists() else pd.DataFrame()


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def save_overlays(
    examples: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]], overlay_path: Path
) -> None:
    if not examples:
        return
    columns = 2
    rows = int(np.ceil(len(examples) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(9, 4.2 * rows))
    axes_array = np.atleast_1d(axes).ravel()
    for axis, (image_id, perturbation, image, truth, prediction) in zip(axes_array, examples):
        axis.imshow(overlay_truth_prediction(image, truth, prediction))
        axis.set_title(f"{image_id[:10]}... {perturbation}; truth=green pred=red")
        axis.axis("off")
    for axis in axes_array[len(examples) :]:
        axis.axis("off")
    fig.tight_layout()
    save_png(fig, overlay_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    metrics_path, overlay_path = output_paths(args.stage)
    existing = prepare_outputs(metrics_path, overlay_path, args)
    completed = set()
    if not existing.empty:
        completed = set(zip(existing["image_id"].astype(str), existing["perturbation"].astype(str)))

    processor = build_processor()
    checkpoint_hash = checkpoint_sha256(CHECKPOINT)
    perturbations = selected_perturbations(args.conditions)
    rows: list[dict[str, object]] = []
    examples: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []

    for image_dir in selected_image_dirs(args.stage):
        image_id, image, truth = load_train_example(image_dir)
        for perturbation in perturbations:
            if (image_id, perturbation.name) in completed:
                continue
            perturbed = apply_perturbation(image, perturbation)
            prediction, raw_count, overlap_fraction, latency_ms, peak_memory_mb = predict(processor, perturbed)
            metrics = compute_instance_metrics(truth, prediction, iou_threshold=IOU_THRESHOLD)
            rows.append(
                {
                    "split": split_label(args.stage),
                    "image_id": image_id,
                    "method": METHOD,
                    "method_label": METHOD_LABEL,
                    "perturbation": perturbation.name,
                    "perturbation_params": perturbation.params,
                    "iou_threshold": IOU_THRESHOLD,
                    "model_repo": "facebook/sam3",
                    "model_code_commit": CODE_COMMIT,
                    "checkpoint": CHECKPOINT.relative_to(REPO_ROOT).as_posix(),
                    "checkpoint_sha256": checkpoint_hash,
                    "prompt": PROMPT,
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "mask_threshold": MASK_THRESHOLD,
                    "processor_resolution": RESOLUTION,
                    "raw_mask_count": raw_count,
                    "overlap_pixel_fraction": round(overlap_fraction, 8),
                    "zero_prediction": metrics.pred_instances == 0,
                    "gpu": torch.cuda.get_device_name(0),
                    "peak_memory_mb": round(peak_memory_mb, 3),
                    "latency_ms": round(latency_ms, 3),
                    **asdict(metrics),
                }
            )
            if len(examples) < 6:
                examples.append((image_id, perturbation.name, perturbed, truth, prediction))
            print(f"Completed {image_id} {perturbation.name}", flush=True)

    metrics = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    if metrics.empty:
        raise RuntimeError("No SAM3 metrics were produced or loaded")
    write_csv_atomic(metrics, metrics_path)
    save_overlays(examples, overlay_path)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {overlay_path}")


if __name__ == "__main__":
    main()
