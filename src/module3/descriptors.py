"""Region descriptors and the descriptor-only classifier — tasks 3.1 and 3.2.

Owner: Ng Zhi Xuan (Module 3).

Two responsibilities live here, and they are kept apart on purpose:

* ``extract_descriptors`` measures one candidate blob. It makes no decision.
* ``classify_by_descriptors`` decides a class from those measurements alone.

The second of these is the *baseline* classifier. It reasons only about the
shape of the changed pixels, which is the obvious approach and the one most
student projects stop at. Its weakness is structural rather than a matter of
tuning: an open circuit and a mouse bite can have identical shapes, and what
separates them is not the shape of the missing copper but where that copper sat
relative to the trace. That second reading is supplied by ``connectivity.py``,
and Chapter 4 compares the two directly.

Keeping the weaker classifier rather than deleting it is deliberate. A measured
comparison between a descriptor-only baseline and a context-aware alternative
is worth considerably more in Chapter 4 than a single unexplained accuracy
figure would be.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# The six display labels. Every one maps to a contract label in classify.py;
# nothing else in the system is allowed to hold a second copy of this list.
DISPLAY_LABELS = (
    "Open circuit",
    "Mouse bite",
    "Missing hole/pin-hole",
    "Short",
    "Spur",
    "Spurious copper",
)


# ---------------------------------------------------------------------------
# Task 3.1 — region descriptors
# ---------------------------------------------------------------------------
def extract_descriptors(blob: dict[str, Any]) -> dict[str, Any]:
    """Measure one candidate defect and return its geometric descriptors.

    The blob is supplied as a mapping rather than as a ``Blob`` dataclass so
    that this function can be exercised from the prototyping notebook without
    importing the rest of the system.

    Descriptors returned
    --------------------
    ``area_px``
        Connected-component pixel count, taken from Module 2 rather than
        recomputed from the contour. ``cv2.contourArea`` applies the shoelace
        formula to the traced outline, which disagrees with the pixel count by
        a few per cent on small regions, and the pixel count is what the
        physical measurement in task 3.3 is derived from.
    ``perimeter_px``
        Contour arc length. Combined with the area it gives circularity, which
        is what separates a compact pin hole from a ragged mouse bite.
    ``aspect_ratio``
        Major over minor axis of the *rotated* minimum-area rectangle. The
        rotated box matters: a 45-degree open circuit has an axis-aligned
        bounding box that is very nearly square, so an axis-aligned aspect
        ratio would report it as compact and misclassify it.
    ``extent``
        Pixel area over rotated bounding-box area. Low for a curved or
        L-shaped region, high for a solid rectangle.
    ``solidity``
        Pixel area over convex hull area. A notch bitten out of a trace edge
        leaves a concave region, so solidity falls; a hole punched through
        copper stays convex.
    ``circularity``
        4*pi*area / perimeter^2. Exactly 1.0 for a perfect disc and falling
        towards 0 as the outline becomes elongated or ragged.
    ``eccentricity``
        Derived from the second-order central moments, in the range 0 (a disc)
        to just under 1 (a line). Computed from moments rather than from
        ``cv2.fitEllipse`` because the latter needs at least five contour
        points and raises on the very small regions a pin hole produces.
    ``hu_moments``
        Seven values that are invariant to translation, scale and rotation.
        Carried for the descriptor table in Chapter 4 and for future work; the
        current rules do not read them, and saying so is more honest than
        implying a shape-matching stage that does not exist.
    """
    contour = blob["contour"]
    area_px = float(blob["area_px"])

    # Rotated minimum-area rectangle: ((cx, cy), (w, h), angle).
    if len(contour) >= 3:
        rect = cv2.minAreaRect(contour)
    else:
        rect = ((0.0, 0.0), (1.0, 1.0), 0.0)
    (_, _), (rect_w, rect_h), angle = rect

    major_axis = max(rect_w, rect_h)
    minor_axis = min(rect_w, rect_h)
    minor_axis = minor_axis if minor_axis > 0 else 1.0
    aspect_ratio = float(major_axis) / minor_axis

    rect_area = major_axis * minor_axis
    extent = area_px / rect_area if rect_area > 0 else 0.0

    hull_area = cv2.contourArea(cv2.convexHull(contour)) if len(contour) >= 3 else 0.0
    solidity = area_px / hull_area if hull_area > 0 else 1.0

    perimeter = float(cv2.arcLength(contour, True)) if len(contour) >= 2 else 0.0
    circularity = (4.0 * np.pi * area_px / (perimeter ** 2)) if perimeter > 0 else 0.0

    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()

    return {
        "area_px": area_px,
        "perimeter_px": round(perimeter, 3),
        "rect": rect,
        "angle_deg": float(angle),
        "width_px": float(major_axis),
        "height_px": float(minor_axis),
        "aspect_ratio": round(aspect_ratio, 4),
        "extent": round(float(extent), 4),
        "solidity": round(float(min(solidity, 1.0)), 4),
        "circularity": round(float(min(circularity, 1.0)), 4),
        "eccentricity": round(_eccentricity(moments), 4),
        "hu_moments": hu_moments,
    }


def _eccentricity(moments: dict[str, float]) -> float:
    """Eccentricity of the ellipse with the same second moments as the region.

    The two eigenvalues of the covariance matrix built from the normalised
    central moments are the squared semi-axis lengths. Eccentricity is then
    sqrt(1 - minor/major), which is 0 for a disc and approaches 1 for a line.
    """
    m00 = moments.get("m00", 0.0)
    if m00 <= 0:
        return 0.0

    mu20 = moments["mu20"] / m00
    mu02 = moments["mu02"] / m00
    mu11 = moments["mu11"] / m00

    common = np.sqrt(max(0.0, 4.0 * mu11 ** 2 + (mu20 - mu02) ** 2))
    major = (mu20 + mu02 + common) / 2.0
    minor = (mu20 + mu02 - common) / 2.0

    if major <= 0:
        return 0.0
    return float(np.sqrt(max(0.0, 1.0 - minor / major)))


# ---------------------------------------------------------------------------
# Task 3.2 — stage two, descriptor-only baseline
# ---------------------------------------------------------------------------
# Thresholds are chosen on the DeepPCB trainval split and reported on the
# held-out test split. They are named constants rather than literals inside the
# branches so that the sweep in experiments/benchmark_module3.py can move them
# without editing the rules themselves.
ELONGATED_ASPECT_RATIO = 2.5   # above this, the region reads as a bar not a blob
COMPACT_CIRCULARITY = 0.60     # above this, the region reads as a disc
COMPACT_SOLIDITY = 0.85        # above this, nothing has been bitten out of it


def classify_by_descriptors(polarity: str,
                            features: dict[str, Any]) -> tuple[str, float]:
    """Baseline stage two: assign a class from shape descriptors alone.

    Stage one is the polarity already carried by the blob, which comes from the
    sign of the template-test difference in Module 2 and is measured, not
    guessed. Stage two splits each polarity three ways:

    ==========  ==============  ====================  ==============
    Polarity    Elongated       Compact and convex    Otherwise
    ==========  ==============  ====================  ==============
    removed     Open circuit    Missing hole          Mouse bite
    added       Short           Spurious copper       Spur
    ==========  ==============  ====================  ==============

    Returns the display label and a confidence in the range 0.5 to 1.0. The
    confidence is the margin by which the deciding descriptor cleared its
    threshold, normalised, so a blob that only just qualified as elongated is
    reported as uncertain rather than being presented with the same authority
    as an obvious one. It is not a probability, and Chapter 4 must not describe
    it as one.
    """
    aspect_ratio = features["aspect_ratio"]
    solidity = features["solidity"]
    circularity = features["circularity"]

    elongated = aspect_ratio >= ELONGATED_ASPECT_RATIO
    compact = circularity >= COMPACT_CIRCULARITY and solidity >= COMPACT_SOLIDITY

    if elongated:
        label = "Open circuit" if polarity == "removed" else "Short"
        confidence = _margin(aspect_ratio, ELONGATED_ASPECT_RATIO, scale=2.5)
    elif compact:
        label = "Missing hole/pin-hole" if polarity == "removed" else "Spurious copper"
        confidence = _margin(circularity, COMPACT_CIRCULARITY, scale=0.4)
    else:
        label = "Mouse bite" if polarity == "removed" else "Spur"
        # This class is defined by failing both tests, so its confidence is
        # highest when both failures were decisive rather than marginal.
        confidence = min(
            _margin(ELONGATED_ASPECT_RATIO, aspect_ratio, scale=1.5),
            _margin(COMPACT_SOLIDITY, solidity, scale=0.3),
        )

    return label, confidence


def _margin(value: float, threshold: float, scale: float) -> float:
    """Normalise how decisively a value cleared a threshold into 0.5 to 1.0.

    A blob sitting exactly on the threshold scores 0.5, meaning the rule could
    have gone either way. One that clears it by ``scale`` or more scores 1.0.
    The floor is 0.5 rather than 0 because the class was still assigned:
    reporting near-zero confidence for a decision the system nonetheless acted
    on would misdescribe what happened.
    """
    if scale <= 0:
        return 0.5
    margin = (value - threshold) / scale
    return float(round(0.5 + 0.5 * min(max(margin, 0.0), 1.0), 3))


# ---------------------------------------------------------------------------
# Whole-board similarity, reported on the dashboard and in the PDF
# ---------------------------------------------------------------------------
def calculate_board_ssim(template_img: np.ndarray, test_img: np.ndarray) -> float:
    """Structural similarity between the reference and the inspected board.

    This is a single summary number for the board as a whole, not a detector.
    It is displayed because it gives an operator an immediate sense of how far
    a unit has drifted from the reference before any individual defect is read,
    and because a board with a high defect count but an SSIM close to 1.0
    indicates a registration problem rather than a manufacturing one.
    """
    grey_template = _as_grey(template_img)
    grey_test = _as_grey(test_img)

    if grey_template.shape != grey_test.shape:
        grey_test = cv2.resize(
            grey_test, (grey_template.shape[1], grey_template.shape[0])
        )

    score, _ = ssim(grey_template, grey_test, full=True)
    return float(round(score, 4))


def _as_grey(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
