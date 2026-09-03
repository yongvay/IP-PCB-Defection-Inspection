"""Shared plumbing for the Module 1 experiments.

Owner: Chan Xing Szen.

Kept separate from src/ deliberately: the package under src/ is the system, and
these scripts are the measurements taken of it. Mixing the two makes the
pipeline import pandas and matplotlib to inspect one board, and makes it
unclear which code the marker is being asked to assess.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs"
FIGURE_DIR = REPO_ROOT / "outputs" / "figures"

# Every experiment seeds from this so that a rerun reproduces the reported
# numbers exactly. An unseeded noise study cannot be checked by anyone else.
RANDOM_SEED = 20260901


def rng() -> np.random.Generator:
    return np.random.default_rng(RANDOM_SEED)


def save_table(frame: pd.DataFrame, name: str) -> Path:
    """Write a metrics table to outputs/ and echo it, so a run is self-documenting."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    frame.to_csv(path, index=False)
    print(f"\n=== {name} ===")
    print(frame.to_string(index=False))
    print(f"written: {path}")
    return path


def centre_crop(image: np.ndarray, size: int) -> np.ndarray:
    """Take a square crop from the centre of an image.

    Used so that a filter study runs in minutes rather than hours. Cropping
    rather than downscaling is the important part: resizing would resample the
    image and change the very noise characteristics the study is measuring,
    whereas a crop keeps every pixel at its captured resolution.
    """
    height, width = image.shape[:2]
    if height <= size and width <= size:
        return image
    top = max(0, (height - size) // 2)
    left = max(0, (width - size) // 2)
    return image[top:top + size, left:left + size]
