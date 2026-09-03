"""Module 3 entry point — classification, measurement and verdict.

Owner: Ng Zhi Xuan (tasks 3.2, 3.3, 3.4).

This file is the boundary between the frozen contract in ``src/contracts.py``
and the two classifiers that decide a defect's class. It does three things and
delegates everything else:

1. Presents each ``Blob`` to the rule sets in the dictionary form they expect.
2. Chooses between the connectivity classifier and the descriptor baseline,
   and records which one actually decided each defect.
3. Converts pixel measurements to millimetres and applies the board verdict.

Two vocabularies, one translation
---------------------------------
``descriptors.py`` and ``connectivity.py`` return human-readable labels such as
"Open circuit" for the dashboard and the PDF report. The contract, the
ground-truth parser and the evaluation harness all use the machine labels in
``DEFECT_CLASSES``. Scoring cannot compare the two vocabularies, so the mapping
is made once, here, rather than being duplicated wherever a comparison happens.
"""

from __future__ import annotations

import time
from typing import Any

from src.contracts import (
    DEFECT_CLASSES,
    Blob,
    Defect,
    InspectionReport,
    LocalisationResult,
)
from src.module3 import connectivity, descriptors
from src.module3.connectivity import BoardContext

# Display label (classifiers, dashboard, PDF report) -> contract label
# (contracts.py, ground_truth.py, evaluate.py). The assertion below enforces
# agreement at import time rather than letting a typo surface as a silent
# scoring failure in Week 6.
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

CONTRACT_TO_DISPLAY = {value: key for key, value in DISPLAY_TO_CONTRACT.items()}

# Which classifier runs by default. "connectivity" reads how the region sits
# against the surrounding copper and falls back to the descriptor rules when
# that reading is unusable; "descriptor" is the shape-only baseline retained
# for the Chapter 4 comparison. Overridden per run via params["classifier"].
DEFAULT_CLASSIFIER = "connectivity"

# ---------------------------------------------------------------------------
# Task 3.4 — severity weighting
# ---------------------------------------------------------------------------
# A defect count alone treats a bridged pair of traces as equivalent to a
# cosmetic nick on a trace edge, which no inspection operator would accept.
# The weights below follow the electrical consequence of each class rather than
# its visual prominence:
#
#   3  the board cannot work — the net is broken or two nets are joined
#   2  the board works now but a stray island is a latent bridge risk
#   1  the board works and the trace is narrowed but continuous
#
# The scale is ordinal, not a probability of failure, and Chapter 4 should
# present it as an engineering convention adopted for the demonstration rather
# than as a measured reliability model.
SEVERITY = {
    "open_circuit": 3,
    "short": 3,
    "spurious_copper": 2,
    "mouse_bite": 1,
    "spur": 1,
    "pin_hole": 1,
}

# Any defect at or above this weight fails the board on its own, however
# generous the count tolerance is.
CRITICAL_SEVERITY = 3

# Default count tolerance. Zero means a board fails on any defect at all, which
# is the correct default for bare-board inspection.
DEFAULT_MAX_DEFECTS = 0


# ---------------------------------------------------------------------------
# Task 3.1 to 3.2 — describing and classifying one blob
# ---------------------------------------------------------------------------
def as_mapping(blob: Blob) -> dict[str, Any]:
    """Present a ``Blob`` in the dictionary form the rule sets expect.

    The rules were prototyped against plain dictionaries before Module 2
    existed. Adapting here rather than rewriting them keeps the notebook
    prototype and this pipeline in agreement, and keeps the rule sets testable
    without constructing contract objects.
    """
    return {
        "id": blob.id,
        "bbox": blob.bbox,
        "contour": blob.contour,
        "centroid": blob.centroid,
        "area_px": blob.area_px,
        "polarity": blob.polarity,
    }


def describe(blob: Blob) -> dict[str, Any]:
    """Region descriptors for one blob. (Task 3.1)"""
    return descriptors.extract_descriptors(as_mapping(blob))


def classify(blob: Blob,
             context: BoardContext | None = None,
             method: str = DEFAULT_CLASSIFIER) -> tuple[str, float, str]:
    """Assign one of the six classes to a candidate defect. (Task 3.2)

    Stage one is the polarity carried by the blob, which comes from the sign of
    the template-test difference in Module 2 and is measured rather than
    inferred. Stage two is either rule set.

    Returns the contract label, a confidence between 0.5 and 1.0, and the name
    of the rule set that actually decided. The third value matters: the
    connectivity classifier declines to rule on a region whose copper context
    cannot be read, and how often that happens is a result Chapter 4 has to
    report rather than hide behind a silent fallback.
    """
    features = describe(blob)

    if method == "connectivity" and context is not None:
        context_features = connectivity.measure_context(as_mapping(blob), context)
        decision = connectivity.classify_by_connectivity(blob.polarity, context_features)
        if decision is not None:
            display_label, confidence = decision
            return DISPLAY_TO_CONTRACT[display_label], confidence, "connectivity"

    display_label, confidence = descriptors.classify_by_descriptors(
        blob.polarity, features
    )
    return DISPLAY_TO_CONTRACT[display_label], confidence, "descriptor"


# ---------------------------------------------------------------------------
# Task 3.3 — physical measurement
# ---------------------------------------------------------------------------
def measure(blob: Blob, mm_per_px: float) -> float:
    """Convert a pixel area to square millimetres. (Task 3.3)

    Area scales with the square of a linear factor, hence ``mm_per_px``
    squared. A common error is to apply the factor once, which under-reports a
    defect on a 48 px/mm board by a factor of 48.
    """
    return float(blob.area_px) * (mm_per_px ** 2)


def measure_dimensions(features: dict[str, Any],
                       mm_per_px: float) -> tuple[float, float]:
    """Physical length and width of a defect, in millimetres. (Task 3.3)

    Taken from the *rotated* minimum-area rectangle rather than the
    axis-aligned bounding box. A diagonal open circuit two millimetres long
    would otherwise be reported as roughly 1.4 mm by 1.4 mm, which is not a
    measurement of anything physical.
    """
    return (
        float(features["width_px"]) * mm_per_px,
        float(features["height_px"]) * mm_per_px,
    )


# ---------------------------------------------------------------------------
# Task 3.4 — board verdict
# ---------------------------------------------------------------------------
def decide_verdict(defects: list[Defect],
                   params: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    """Pass or fail the board, and explain why. (Task 3.4)

    Three independent conditions can fail a board, checked in this order:

    1. Any single defect at or above ``CRITICAL_SEVERITY``. An open circuit or
       a short is a functional failure, so no count tolerance can excuse it.
    2. More defects than ``max_defects``.
    3. A total severity weight above ``max_severity``, which catches a board
       carrying many individually tolerable defects.

    The reason is returned alongside the verdict so the dashboard and the PDF
    report can state *why* a board failed. A bare PASS or FAIL is not an
    inspection result an operator can act on.
    """
    params = params or {}
    max_defects = int(params.get("max_defects", DEFAULT_MAX_DEFECTS))
    max_severity = params.get("max_severity")

    total_severity = sum(SEVERITY.get(defect.defect_class, 1) for defect in defects)
    critical = [
        defect for defect in defects
        if SEVERITY.get(defect.defect_class, 1) >= CRITICAL_SEVERITY
    ]

    detail: dict[str, Any] = {
        "defect_count": len(defects),
        "total_severity": total_severity,
        "critical_count": len(critical),
        "max_defects": max_defects,
        "max_severity": max_severity,
    }

    if critical:
        classes = sorted({defect.defect_class for defect in critical})
        detail["reason"] = (
            f"{len(critical)} critical defect(s) present: {', '.join(classes)}"
        )
        return "FAIL", detail

    if len(defects) > max_defects:
        detail["reason"] = (
            f"{len(defects)} defect(s) exceeds the tolerance of {max_defects}"
        )
        return "FAIL", detail

    if max_severity is not None and total_severity > int(max_severity):
        detail["reason"] = (
            f"total severity {total_severity} exceeds the limit of {max_severity}"
        )
        return "FAIL", detail

    detail["reason"] = (
        "no defects detected" if not defects
        else f"{len(defects)} defect(s) within the tolerance of {max_defects}"
    )
    return "PASS", detail


# ---------------------------------------------------------------------------
# Assembling the report
# ---------------------------------------------------------------------------
def build_report(localisation: LocalisationResult,
                 mm_per_px: float,
                 started_at: float,
                 params: dict[str, Any] | None = None,
                 context: BoardContext | None = None) -> InspectionReport:
    """Turn candidate blobs into the final inspection report.

    Called by the orchestrator in ``src/pipeline.py``. Note that noise removal
    has already happened twice by this point: morphological opening in Module 1
    and the minimum-area threshold in Module 2. No further area filter is
    applied here, because filtering in three places makes the effective
    threshold impossible to reason about when tuning.

    ``context`` is optional so that the function still runs when a caller has
    only a ``LocalisationResult`` — the classifier then falls back to the
    descriptor baseline for every blob, which is exactly the configuration
    Chapter 4 reports as the baseline row.
    """
    params = params or {}
    method = params.get("classifier", DEFAULT_CLASSIFIER)

    defects: list[Defect] = []
    for blob in localisation.blobs:
        features = describe(blob)
        defect_class, confidence, decided_by = classify(blob, context, method)
        width_mm, height_mm = measure_dimensions(features, mm_per_px)

        defects.append(Defect(
            id=blob.id,
            bbox=blob.bbox,
            defect_class=defect_class,
            area_mm2=measure(blob, mm_per_px),
            confidence=confidence,
            polarity=blob.polarity,
            width_mm=width_mm,
            height_mm=height_mm,
            severity=SEVERITY.get(defect_class, 1),
            decided_by=decided_by,
        ))

    verdict, detail = decide_verdict(defects, params)

    return InspectionReport(
        defects=defects,
        verdict=verdict,
        runtime_s=time.perf_counter() - started_at,
        align_residual=localisation.align_residual,
        mm_per_px=mm_per_px,
        verdict_reason=detail["reason"],
        verdict_detail=detail,
        classifier=method,
    )
