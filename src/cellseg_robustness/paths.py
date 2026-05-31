"""Repository path helpers."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "raw" / "dsb2018"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

RESULT_SUBDIRS = {
    "dataset": RESULTS_DIR / "dataset",
    "baselines": RESULTS_DIR / "baselines",
    "robustness": RESULTS_DIR / "robustness",
    "supervised": RESULTS_DIR / "supervised",
    "vlm": RESULTS_DIR / "vlm",
}


def ensure_output_dirs() -> None:
    """Create standard generated-output directories."""
    for path in RESULT_SUBDIRS.values():
        path.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
