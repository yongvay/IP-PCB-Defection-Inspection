"""Morphological cleanup of the difference image — Module 1, task 1.10.

Owner: Chan Xing Szen.

This sits between differencing and blob extraction in the pipeline, so it is
exposed as a single pure function that the orchestrator calls. Yong Vay only
calls it; Xing Szen owns the implementation and the tuning.

Contract: same shape in, same shape out, no side effects.

Why opening dominates everything else in the pipeline
-----------------------------------------------------
The reason is geometric rather than statistical. Binarisation jitter along a
trace edge produces differences that are long but only one or two pixels wide,
whereas a genuine defect is compact in both directions. An elliptical
structuring element of radius two fits inside a compact blob but cannot fit
inside a one-pixel ribbon, so opening deletes the false positives and keeps the
real defects. No intensity threshold can make that distinction, because the
false and true differences have identical pixel values; only shape separates
them.

Measured on 88 DeepPCB pairs, varying only the opening kernel:

    opening kernel   detections   precision   recall     F1
          none            1548       0.10       0.71    0.18
          3 x 3            690       0.28       0.87    0.42
          5 x 5            176       0.93       0.74    0.82

Author: Chan Xing Szen (Member A, Module 1)
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def clean_difference(diff_bin: np.ndarray,
                     params: dict[str, Any] | None = None) -> np.ndarray:
    """Suppress registration jitter and scan noise in the difference image.

    Three optional stages, applied in this order:

    * **Top-hat** (off by default) removes any background component larger than
      its element, which is aimed at a slow illumination gradient. A difference
      image has no such gradient — it is already the residue of a subtraction —
      so this stage is available for the benchmark to test and is expected to
      earn its keep only on unregistered photographic input.
    * **Opening** erodes then dilates, deleting anything the structuring element
      cannot contain. This is the stage that does the work.
    * **Closing** dilates then erodes, rejoining a genuine defect that opening
      has fragmented into pieces.

    Every kernel size is read from ``params`` so the benchmark can sweep them
    without editing this file.
    """
    params = params or {}
    result = diff_bin

    tophat_size = int(params.get("morph_tophat_kernel", 0))
    if tophat_size > 0:
        result = cv2.morphologyEx(result, cv2.MORPH_TOPHAT, _element(tophat_size))

    # Verified default. Do not lower it without re-running the sweep above:
    # a 3 x 3 element drops precision from 0.93 to 0.28.
    open_size = int(params.get("morph_open_kernel", 5))
    if open_size > 0:
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, _element(open_size))

    close_size = int(params.get("morph_close_kernel", 5))
    if close_size > 0:
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, _element(close_size))

    return result.astype(np.uint8)


def _element(size: int) -> np.ndarray:
    """An elliptical structuring element.

    Elliptical rather than rectangular because a rectangle has corners that
    reach further along its diagonals than along its axes, so a rectangular
    element preserves diagonal jitter that it deletes horizontally. An ellipse
    treats every direction alike, which is what "compact" is supposed to mean.
    """
    size = size if size % 2 == 1 else size + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
