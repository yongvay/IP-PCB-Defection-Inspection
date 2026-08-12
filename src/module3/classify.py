"""Module 3 contract adapter — classification, measurement and verdict.

Owner: Ng Zhi Xuan (tasks 3.1 to 3.4).

This file is deliberately thin. The real classification work lives in
``descriptors.py`` and is Zhi Xuan's own code, moved here unchanged from the
repository root. What this module adds is the translation between that code
and the frozen interface in ``src/contracts.py``:

  * Module 2 emits ``Blob`` dataclasses; ``descriptors.py`` was prototyped
    against plain dictionaries, so each blob is presented as a mapping.
  * ``descriptors.py`` returns human-readable labels such as "Open circuit"
    for display in the dashboard; the contract, the ground-truth parser and
    the evaluation harness all use the machine labels in ``DEFECT_CLASSES``.
    Scoring cannot compare the two vocabularies, so the mapping is made once,
    here, rather than being duplicated wherever a comparison happens.

Keeping the translation separate from the rules means the rules stay in one
file with one author, which matters because understanding of code is assessed
individually and live.
"""

import time
from typing import Any

from src.contracts import (
    DEFECT_CLASSES,
    Blob,
    Defect,
    InspectionReport,
    LocalisationResult,
)
from src.module3 import descriptors

# Display label (descriptors.py, dashboard, PDF report) -> contract label
# (contracts.py, ground_truth.py, evaluate.py). Every value on the right must
# appear in DEFECT_CLASSES; the assertion below enforces that at import time
# rather than letting a typo surface as a silent scoring failure in Week 6.
DISPLAY_TO_CONTRACT = {
    "Open circuit": "open_circuit",
    "Mouse bite": "mouse_bite",
    "Missing hole/pin-hole": "pin_hole",
    "Short": "short",
    "Spur": "spur",
    "Spurious copper": "spurious_copper",
}

assert set(DISPLAY_TO_CONTRACT.values()) == set(DEFECT_CLASSES), (
    "Display labels and DEFECT_CLASSES have drifted apart"
)

# Board fails if more than this many defects are found. Task 3.4 replaces this
# with a configurable severity-weighted rule.
DEFAULT_MAX_DEFECTS = 0

# The stage-two rules are hard thresholds on aspect ratio and solidity, so they
# produce a decision but no graded score. A fixed value is used rather than an
# invented one, and it is not reported as a probability anywhere.
#
# TODO (task 3.2): derive a real confidence, for example the margin by which a
# descriptor clears its threshold, so that borderline blobs can be flagged.
PLACEHOLDER_CONFIDENCE = 0.5


def as_mapping(blob: Blob) -> dict[str, Any]:
    """Present a Blob in the dictionary form ``descriptors.py`` expects.

    The prototype in ``notebooks/prototyping.ipynb`` was written against a mock
    dictionary before Module 2 existed. Adapting here rather than rewriting the
    rules keeps that prototype and this pipeline in agreement.
    """
    return {
        "id": blob.id,
        "bbox": blob.bbox,
        "contour": blob.contour,
        "area_px": blob.area_px,
        "polarity": blob.polarity,
    }


def describe(blob: Blob) -> dict[str, Any]:
    """Region descriptors for one blob. (Task 3.1)

    Delegates to ``descriptors.extract_descriptors``: rotated bounding box,
    aspect ratio, extent, solidity and Hu moments.
    """
    return descriptors.extract_descriptors(as_mapping(blob))


def classify(blob: Blob) -> tuple[str, float]:
    """Two-stage rule-based classification into one of the six classes. (Task 3.2)

    Stage one is the polarity already carried by the blob, which comes from the
    sign of the template-test difference in Module 2 and is not guesswork.
    Stage two is the descriptor rule set in ``descriptors.classify_defect``.

    The contract label is returned, not the display label.
    """
    measurements = describe(blob)
    display_label = descriptors.classify_defect(
        blob.polarity,
        measurements["aspect_ratio"],
        measurements["solidity"],
    )
    return DISPLAY_TO_CONTRACT[display_label], PLACEHOLDER_CONFIDENCE


def measure(blob: Blob, mm_per_px: float) -> float:
    """Convert pixel area to mm^2 using Module 1's calibration factor. (Task 3.3)

    Area scales with the square of a linear factor, hence mm_per_px squared.
    """
    return blob.area_px * (mm_per_px ** 2)


def decide_verdict(defects: list[Defect],
                   params: dict[str, Any] | None = None) -> str:
    """Pass or fail the board against a defect-count tolerance. (Task 3.4)

    TODO (task 3.4): weight the count by severity, so that one short is not
    treated as equivalent to one cosmetic spur.
    """
    params = params or {}
    limit = params.get("max_defects", DEFAULT_MAX_DEFECTS)
    return "FAIL" if len(defects) > limit else "PASS"


def build_report(localisation: LocalisationResult,
                 mm_per_px: float,
                 started_at: float,
                 params: dict[str, Any] | None = None) -> InspectionReport:
    """Turn candidate blobs into the final inspection report.

    Called by the orchestrator in ``src/pipeline.py``. Note that noise removal
    has already happened twice by this point: morphological opening in Module 1
    and the minimum-area threshold in Module 2. No further area filter is
    applied here, because filtering in three places makes the effective
    threshold impossible to reason about when tuning.
    """
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
