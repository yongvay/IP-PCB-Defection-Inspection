"""Rectification benchmark — Module 1, task 1.8.

Owner: Chan Xing Szen.

HRIPCB records the angle it turned every board through, so unlike almost
everything else in this project the rectification stage has a directly
checkable answer. This experiment reports the error in degrees.

Two things are measured, and the difference between them is the finding.

**Absolute angle.** The angle estimated from the rotated image alone. This
carries not only the estimator's error but also whatever tilt the board already
had when it was first photographed, since no board is laid perfectly square.

**Template-relative angle.** The same estimate, minus the estimate made on that
board's own reference image. Rectification exists to bring a test image into
agreement with its template, not with the image axes, so the quantity that
actually matters to the pipeline is the difference between the two — and
subtracting cancels the board's inherent tilt, which is a constant common to
both.

Estimation is also run at full resolution and at quarter resolution, because an
angle is scale-invariant in principle and the saving is large in practice. If
the two agree, the cheaper one is justified by measurement rather than by
assumption.

Run:  python -m experiments.benchmark_rectify
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd

from experiments.common import save_table
from src.module1 import ingest, rectify

SAMPLE_SIZE = 120
SCALES = (0.25, 1.0)


def measure(scale: float) -> pd.DataFrame:
    pairs = ingest.index_hripcb(rotated=True)
    step = max(1, len(pairs) // SAMPLE_SIZE)
    sample = pairs[::step][:SAMPLE_SIZE]

    template_angles: dict[str, float] = {}
    rows = []

    for pair in sample:
        if pair.angle_deg is None:
            continue
        board = pair.name.split("_")[0]

        if board not in template_angles:
            template = _scaled(ingest.load_grey(pair.template_path), scale)
            template_angles[board] = rectify.estimate_angle(template) or 0.0

        test = _scaled(ingest.load_grey(pair.test_path), scale)
        started = time.perf_counter()
        estimate, method = rectify.estimate_angle_detailed(test)
        elapsed = time.perf_counter() - started

        # The threshold fallback returns the whole frame, so its angle is always
        # zero. Counting that as a detection would report a confident wrong
        # answer as an accurate one on boards that happen to be near square.
        if method != "flood":
            estimate = None

        if estimate is None:
            rows.append({"name": pair.name, "board": board, "scale": scale,
                         "true_deg": pair.angle_deg, "estimated_deg": np.nan,
                         "abs_error_deg": np.nan, "relative_error_deg": np.nan,
                         "ms": round(elapsed * 1000, 1), "detected": False,
                         "method": method})
            continue

        relative = estimate - template_angles[board]
        rows.append({
            "name": pair.name, "board": board, "scale": scale,
            "true_deg": pair.angle_deg,
            "estimated_deg": round(estimate, 3),
            "abs_error_deg": round(abs(estimate - pair.angle_deg), 3),
            "relative_error_deg": round(abs(relative - pair.angle_deg), 3),
            "ms": round(elapsed * 1000, 1),
            "detected": True,
            "method": method,
        })

    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scale, group in frame.groupby("scale"):
        found = group[group["detected"]]
        rows.append({
            "scale": scale,
            "boards": len(group),
            "detected": len(found),
            "detection_rate": round(len(found) / len(group), 4) if len(group) else 0.0,
            "mae_absolute_deg": round(float(found["abs_error_deg"].mean()), 3),
            "mae_relative_deg": round(float(found["relative_error_deg"].mean()), 3),
            "median_relative_deg": round(float(found["relative_error_deg"].median()), 3),
            "within_1deg_pct": round(100.0 * float((found["relative_error_deg"] <= 1.0).mean()), 1),
            "within_2deg_pct": round(100.0 * float((found["relative_error_deg"] <= 2.0).mean()), 1),
            "ms_per_board": round(float(found["ms"].mean()), 1),
        })
    return pd.DataFrame(rows)


def _scaled(image: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def main() -> None:
    print(f"Rectification benchmark: up to {SAMPLE_SIZE} rotated HRIPCB boards "
          f"at scales {SCALES}.")
    per_board = pd.concat([measure(scale) for scale in SCALES], ignore_index=True)
    save_table(per_board, "rectify_per_board.csv")
    save_table(summarise(per_board), "rectify_benchmark.csv")


if __name__ == "__main__":
    main()
