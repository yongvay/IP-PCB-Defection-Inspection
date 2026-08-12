"""Module 1 stub — Data ingestion, preprocessing and calibration.

Owner: Chan Xing Szen (tasks 1.3 to 1.8).

This file exists so that Modules 2 and 3 can be written, run and tested before
Module 1 is finished. Every function below returns a valid PreprocessResult
built with placeholder algorithms, which the owner replaces one at a time.
The signatures are fixed by the contract and must not change; the bodies are
free to change completely.

DeepPCB images are already binarised by the dataset authors, so the fallback
threshold below is genuinely usable and the integration spine can run
end-to-end today. That is the point of the stub.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.contracts import PreprocessResult

# DeepPCB was captured at approximately 48 pixels per millimetre (dataset
# README). Task 1.7 replaces this constant with a factor derived from a known
# board feature rather than a quoted figure.
DEEPPCB_PIXELS_PER_MM = 48.0


def load_pair(template_path: str | Path,
              test_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one template-test image pair as greyscale arrays. (Task 1.3)"""
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    test = cv2.imread(str(test_path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"Could not read template image: {template_path}")
    if test is None:
        raise FileNotFoundError(f"Could not read test image: {test_path}")
    return template, test


def denoise(image: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
    """TODO (task 1.4): Gaussian, median and bilateral filtering, benchmarked.

    Median is used as the placeholder because it removes the salt-and-pepper
    speckle typical of scanned boards without blurring trace edges.
    """
    params = params or {}
    kernel = params.get("median_kernel", 3)
    return cv2.medianBlur(image, kernel)


def enhance(image: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
    """TODO (task 1.5): histogram equalisation, CLAHE, linear stretching."""
    return image


def binarise(image: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
    """TODO (task 1.6): Otsu against adaptive mean and adaptive Gaussian.

    Otsu is the placeholder: it chooses the threshold that best separates the
    two intensity populations, which suits the bimodal copper-against-substrate
    histogram of a bare board.
    """
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(np.uint8)


def calibrate(image: np.ndarray, params: dict[str, Any] | None = None) -> float:
    """TODO (task 1.7): derive mm per pixel from a known board feature."""
    return 1.0 / DEEPPCB_PIXELS_PER_MM


def preprocess_pair(template_path: str | Path,
                    test_path: str | Path,
                    params: dict[str, Any] | None = None) -> PreprocessResult:
    """Run the full Module 1 pipeline for one pair. Called by the orchestrator."""
    params = params or {}
    template, test = load_pair(template_path, test_path)

    template_bin = binarise(enhance(denoise(template, params), params), params)
    test_bin = binarise(enhance(denoise(test, params), params), params)

    result = PreprocessResult(
        template_bin=template_bin,
        test_bin=test_bin,
        mm_per_px=calibrate(template, params),
        params={"stub": True, **params},
    )
    result.validate()
    return result
