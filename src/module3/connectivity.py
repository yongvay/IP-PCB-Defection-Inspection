"""Trace-connectivity context for stage two — task 3.2.

Owner: Ng Zhi Xuan (Module 3).

Why a second reading is needed
------------------------------
The descriptor-only classifier in ``descriptors.py`` asks what the changed
region *looks like*. That question cannot separate the six classes, because
three pairs of them are shape-ambiguous by definition:

* An open circuit and a mouse bite are both a piece missing from a trace. A
  wide bite and a narrow break have the same outline.
* A short and a spur are both copper attached to a trace. A short that bridges
  two closely spaced traces is the same shape as a spur that happens to be long.
* A pin hole and a small spurious island are both compact and roughly round.

What actually distinguishes them is not the shape of the changed pixels but
**how that region sits against the surrounding copper**. The definitions in the
DeepPCB taxonomy are themselves relational:

===================  ========================================================
Class                Relationship to the surrounding copper
===================  ========================================================
Missing hole         Removed copper fully enclosed by remaining copper
Open circuit         Removed copper separating a trace into two ends
Mouse bite           Removed copper open to the substrate on one side
Spurious copper      Added copper touching no existing trace
Short                Added copper joining two otherwise separate traces
Spur                 Added copper protruding from a single trace
===================  ========================================================

Reading that relationship is a classical morphology problem and needs no
machine learning. A ring is dilated around the candidate region, intersected
with the copper mask, and the resulting contact patches are counted. Zero,
one, two-or-more and fully-enclosed are four distinct outcomes, and they map
one-to-one onto the taxonomy above.

Which copper mask is the right one to look at
---------------------------------------------
This is the detail that is easy to get backwards. For **removed** copper the
question is what the trace looks like *after* the defect, so the test board's
copper is the reference. For **added** copper the question is what the stray
metal is touching that was legitimately there, so the *template* copper is the
reference — measuring against the test board would count the added copper
itself as part of the trace and make every spur look isolated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

# Width in pixels of the ring examined around each candidate region. Four
# pixels is wide enough to survive the one-pixel gaps morphological opening
# leaves behind, and narrow enough that a neighbouring but unrelated trace
# 20 pixels away is not counted as contact. Swept in
# experiments/benchmark_module3.py.
RING_WIDTH_PX = 4

# A contact patch smaller than this is a stray pixel from binarisation jitter
# rather than a trace end, and counting it would turn every mouse bite into an
# open circuit.
MIN_CONTACT_PX = 6

# Fraction of the ring that must be copper before the region counts as fully
# enclosed. Not 1.0, because a hole near a trace edge has a few ring pixels
# over substrate while still plainly being a hole.
ENCLOSED_COVERAGE = 0.88

# Below this, the region is touching effectively nothing and is an island.
ISOLATED_COVERAGE = 0.02


@dataclass
class BoardContext:
    """The two copper masks the connectivity rules read.

    Both are binary images in which 255 means copper, produced by
    ``src.module2.difference.copper_mask``. Normalising the polarity there
    rather than here means this file never has to know whether copper rendered
    light or dark in the source imagery.
    """

    template_copper: np.ndarray
    test_copper: np.ndarray

    def reference_for(self, polarity: str) -> np.ndarray:
        """Return the mask that the surrounding copper should be read from."""
        return self.test_copper if polarity == "removed" else self.template_copper


def measure_context(blob: dict[str, Any],
                    context: BoardContext,
                    ring_px: int | None = None) -> dict[str, Any]:
    """Measure how one candidate region sits against the surrounding copper.

    Returns ``contact_count`` (how many separate copper patches the region
    touches), ``coverage`` (what fraction of the surrounding ring is copper)
    and ``ring_px`` (how many pixels the ring itself contains, which is zero
    for a degenerate contour and signals that the reading is unusable).

    The work is done inside a cropped window around the bounding box rather
    than on the full board. A 640 x 640 dilation per blob would dominate the
    runtime budget of three seconds per board once a board carries a dozen
    defects, whereas a 60 x 60 window is negligible.
    """
    # Resolved at call time, not as a default argument. A default argument is
    # bound once when the function is defined, so the sweep in
    # experiments/benchmark_module3.py would have silently measured the same
    # width five times over.
    ring_px = RING_WIDTH_PX if ring_px is None else ring_px

    reference = context.reference_for(blob["polarity"])
    height, width = reference.shape[:2]

    x, y, w, h = blob["bbox"]
    margin = ring_px + 2
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(width, x + w + margin), min(height, y + h + margin)
    if x1 <= x0 or y1 <= y0:
        return {"contact_count": 0, "coverage": 0.0, "ring_px": 0}

    # Redraw the region inside the cropped window. The contour carries
    # full-image coordinates, so it is shifted by the crop origin.
    region = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    contour = blob["contour"]
    if len(contour) >= 3:
        cv2.drawContours(region, [contour], -1, 255, thickness=cv2.FILLED,
                         offset=(-x0, -y0))
    else:
        region[y - y0:y - y0 + h, x - x0:x - x0 + w] = 255

    element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1)
    )
    grown = cv2.dilate(region, element)
    ring = cv2.subtract(grown, region)

    ring_area = int(np.count_nonzero(ring))
    if ring_area == 0:
        return {"contact_count": 0, "coverage": 0.0, "ring_px": 0}

    contact = cv2.bitwise_and(ring, reference[y0:y1, x0:x1])
    coverage = float(np.count_nonzero(contact)) / ring_area

    # Count how many *separate* copper patches the ring touches. Two patches on
    # opposite sides is the signature of a severed trace or a bridge; one patch
    # is an edge defect; none is an island.
    count, _, stats, _ = cv2.connectedComponentsWithStats(contact, connectivity=8)
    contact_count = sum(
        1 for label in range(1, count)
        if stats[label, cv2.CC_STAT_AREA] >= MIN_CONTACT_PX
    )

    return {
        "contact_count": contact_count,
        "coverage": round(coverage, 4),
        "ring_px": ring_area,
    }


def classify_by_connectivity(polarity: str,
                             context_features: dict[str, Any]
                             ) -> tuple[str, float] | None:
    """Assign a class from the copper-adjacency reading.

    Returns the display label and a confidence, or ``None`` when the reading is
    unusable — a degenerate contour with no ring, or a removed region touching
    no copper at all, which cannot physically be a removal defect and indicates
    a registration failure rather than a manufacturing one. Returning ``None``
    rather than guessing lets ``classify.py`` fall back to the descriptor rules
    and, importantly, lets Chapter 4 report how often that happened.
    """
    if context_features["ring_px"] == 0:
        return None

    contacts = context_features["contact_count"]
    coverage = context_features["coverage"]

    if polarity == "removed":
        if coverage >= ENCLOSED_COVERAGE:
            # Surrounded by copper on every side: a hole through a pad or trace.
            return "Missing hole/pin-hole", _confidence(coverage, ENCLOSED_COVERAGE, 0.12)
        if contacts >= 2:
            # Copper on two or more separate sides: the trace has been severed.
            return "Open circuit", _confidence(coverage, 0.30, 0.40)
        if contacts == 1:
            # Copper on one side only: a bite out of a trace edge.
            return "Mouse bite", _confidence(0.60 - abs(coverage - 0.45), 0.30, 0.30)
        return None

    if coverage <= ISOLATED_COVERAGE or contacts == 0:
        # Touching nothing that was on the reference board: a stray island.
        return "Spurious copper", _confidence(ISOLATED_COVERAGE - coverage + 0.5, 0.5, 0.05)
    if contacts >= 2:
        # Joining two separate traces: a bridge.
        return "Short", _confidence(coverage, 0.20, 0.40)
    return "Spur", _confidence(0.60 - abs(coverage - 0.35), 0.25, 0.30)


def _confidence(value: float, threshold: float, scale: float) -> float:
    """Normalise a decision margin into the range 0.5 to 1.0.

    Deliberately identical in behaviour to ``descriptors._margin`` so that a
    confidence produced by either classifier means the same thing and the two
    can be compared in Chapter 4 without a correction.
    """
    if scale <= 0:
        return 0.5
    margin = (value - threshold) / scale
    return float(round(0.5 + 0.5 * min(max(margin, 0.0), 1.0), 3))
