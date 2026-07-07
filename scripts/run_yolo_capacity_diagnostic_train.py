#!/usr/bin/env python
"""Train a YOLO capacity diagnostic model on the full train pool."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cellseg_robustness.paths import RESULT_SUBDIRS, ensure_output_dirs


DEFAULT_MODEL = REPO_ROOT / "model_assets" / "yolo" / "yolo11m-seg.pt"
DATA_YAML = RESULT_SUBDIRS["supervised"] / "yolo_label_budget_diagnostic" / "full_train_pool" / "data.yaml"
RUN_ROOT = RESULT_SUBDIRS["supervised"] / "yolo_capacity_diagnostic_train_runs"
MODEL_KEY = "yolo11m"
RUN_NAME = MODEL_KEY
METADATA_PATH = RESULT_SUBDIRS["supervised"] / f"yolo_capacity_diagnostic_{MODEL_KEY}_train_metadata.csv"
SUMMARY_PATH = RESULT_SUBDIRS["supervised"] / f"yolo_capacity_diagnostic_{MODEL_KEY}_train_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DATA_YAML)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")
    if args.batch <= 0:
        parser.error("--batch must be positive")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if args.patience < 0:
        parser.error("--patience must be non-negative")
    return args


def count_list_rows(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def prepare_outputs(overwrite: bool) -> Path:
    run_dir = RUN_ROOT / RUN_NAME
    existing = [path for path in [run_dir, METADATA_PATH, SUMMARY_PATH] if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Existing YOLO capacity training outputs found. Use --overwrite.\n{joined}")
    if overwrite:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        for path in [METADATA_PATH, SUMMARY_PATH]:
            if path.exists():
                path.unlink()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    return run_dir


def gpu_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)


def peak_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated() / (1024**2), 2)


def read_last_metrics(results_csv: Path) -> dict[str, object]:
    if not results_csv.exists():
        return {}
    frame = pd.read_csv(results_csv)
    if frame.empty:
        return {}
    last = frame.tail(1).copy()
    last.columns = [column.strip() for column in last.columns]
    return last.iloc[0].to_dict()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    if not args.model.exists():
        raise FileNotFoundError(f"Missing YOLO weights: {args.model}")
    if not args.data.exists():
        raise FileNotFoundError(f"Missing YOLO data YAML: {args.data}")

    prepare_outputs(args.overwrite)
    data_dir = args.data.parent
    train_images = count_list_rows(data_dir / "train.txt")
    val_images = count_list_rows(data_dir / "val.txt")
    device = 0 if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = YOLO(str(args.model))
    start = time.perf_counter()
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=str(RUN_ROOT),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        amp=False,
        plots=False,
        verbose=False,
        patience=args.patience,
    )
    train_seconds = round(time.perf_counter() - start, 3)
    save_dir = Path(getattr(results, "save_dir", RUN_ROOT / RUN_NAME))
    results_csv = save_dir / "results.csv"
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"

    metadata = {
        "protocol": "supervised_yolo_capacity_train",
        "model_key": MODEL_KEY,
        "data_yaml": args.data.relative_to(REPO_ROOT).as_posix(),
        "model_weights": args.model.relative_to(REPO_ROOT).as_posix(),
        "model_name": args.model.name,
        "pretrained": True,
        "train_images": train_images,
        "val_images": val_images,
        "annotation_type": "YOLO segmentation polygons converted from DSB2018 instance masks",
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "amp": False,
        "patience": args.patience,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "gpu": gpu_name(),
        "peak_memory_mb": peak_memory_mb(),
        "train_seconds": train_seconds,
        "run_dir": save_dir.relative_to(REPO_ROOT).as_posix(),
        "results_csv": results_csv.relative_to(REPO_ROOT).as_posix() if results_csv.exists() else "",
        "best_checkpoint": best_pt.relative_to(REPO_ROOT).as_posix() if best_pt.exists() else "",
        "last_checkpoint": last_pt.relative_to(REPO_ROOT).as_posix() if last_pt.exists() else "",
    }
    pd.DataFrame([metadata]).to_csv(METADATA_PATH, index=False)

    metrics = read_last_metrics(results_csv)
    summary = {**metadata, **{f"yolo_{key}": value for key, value in metrics.items()}}
    pd.DataFrame([summary]).to_csv(SUMMARY_PATH, index=False)

    print(f"Wrote {METADATA_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"YOLO run dir: {save_dir}")
    print(f"Training seconds: {train_seconds}")


if __name__ == "__main__":
    main()
