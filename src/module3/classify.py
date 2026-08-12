"""Module 3 stub — Defect classification, measurement and verdict.

Owner: Ng Zhi Xuan (tasks 3.1 to 3.4, 3.7).

Stage one of the classifier is already solved for free: the polarity carried
by each Blob comes from the sign of the template-test difference, and it
partitions the six classes into two groups of three. Stage two only has to
separate three classes within a group, using region descriptors.

The stub below implements stage one honestly and returns the most common
class within each polarity group as a placeholder for stage two, so the
pipeline produces a well-formed InspectionReport today.
"""

import time
from typing import Any

from src.contracts import Blob, Defect, InspectionReport, LocalisationResult

# Placeholder mapping used until the descriptor rules of task 3.2 exist.
_PLACEHOLDER_CLASS = {"removed": "open_circuit", "added": "short"}

# Board fails if more than this many defects are found. Task 3.4 replaces this
# with a configurable severity-weighted rule.
DEFAULT_MAX_DEFECTS = 0


def describe(blob: Blob) -> dict[str, float]:
    """TODO (task 3.1): area, perimeter, aspect ratio, solidity, extent,
    eccentricity and Hu moments for one blob."""
    x, y, w, h = blob.bbox
    return {
        "area_px": float(blob.area_px),
        "aspect_ratio": w / h if h else 0.0,
        "extent": blob.area_px / (w * h) if w * h else 0.0,
    }


def classify(blob: Blob) -> tuple[str, float]:
    """TODO (task 3.2): stage-two rule-based discrimination into six classes.

    Stage one is the polarity already on the blob and is not guesswork.
    """
    return _PLACEHOLDER_CLASS[blob.polarity], 0.5


def measure(blob: Blob, mm_per_px: float) -> float:
    """Convert pixel area to mm^2 using Module 1's calibration factor. (Task 3.3)

    Area scales with the square of a linear factor, hence mm_per_px squared.
    """
    return blob.area_px * (mm_per_px ** 2)


def decide_verdict(defects: list[Defect],
                   params: dict[str, Any] | None = None) -> str:
    """TODO (task 3.4): configurable defect-count and severity tolerance."""
    params = params or {}
    limit = params.get("max_defects", DEFAULT_MAX_DEFECTS)
    return "FAIL" if len(defects) > limit else "PASS"


def build_report(localisation: LocalisationResult,
                 mm_per_px: float,
                 started_at: float,
                 params: dict[str, Any] | None = None) -> InspectionReport:
    """Turn candidate blobs into the final inspection report."""
    defects = []
    for blob in localisation.blobs:
        defect_class, confidence = classify(blob)
        defects.append(Defect(
            id=blob.id,
            bbox=blob.bbox,
            defect_class=defect_class,
            area_mm2=measure(blob, mm_per_px),
            confidence=confidence,
        ))

    return InspectionReport(
        defects=defects,
        verdict=decide_verdict(defects, params),
        runtime_s=time.perf_counter() - started_at,
        align_residual=localisation.align_residual,
    )
