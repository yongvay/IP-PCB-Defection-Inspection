"""Frozen interface contracts between the three modules.

Every value that crosses a module boundary is defined here and nowhere else.
This is the file that lets all three members build against stubs in parallel:
Module 1 can be rewritten entirely without Module 2 changing, provided the
shape of PreprocessResult is preserved.

Rule agreed at the kick-off meeting: nobody changes a contract unilaterally.
Raise it at the weekly checkpoint; the leader updates this file.

Owner: Ng Yong Vay (Module 2, integration)
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

# A defect either adds copper to the board or removes it. The stage-one
# classifier in Module 3 does nothing more than read this value, which is
# derived from the sign of the template-test difference in Module 2.
Polarity = Literal["added", "removed"]

# The six standard bare-board defect classes. The strings match those produced
# by the ground-truth parser in Module 1, so evaluation can compare labels
# directly without a translation table.
DEFECT_CLASSES = (
    "open_circuit",
    "mouse_bite",
    "pin_hole",
    "short",
    "spur",
    "spurious_copper",
)

Verdict = Literal["PASS", "FAIL"]


# --------------------------------------------------------------------------
# Module 1 -> Module 2
# --------------------------------------------------------------------------
@dataclass
class PreprocessResult:
    """Output of the preprocessing and calibration module.

    Both images are binary and share a shape, which is what allows the
    golden-template differencing in Module 2 to be a plain bitwise operation.
    """

    template_bin: np.ndarray          # uint8, values in {0, 255}
    test_bin: np.ndarray              # uint8, values in {0, 255}, same shape
    mm_per_px: float                  # spatial calibration factor
    params: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Fail loudly at the module boundary rather than deep inside Module 2."""
        if self.template_bin.shape != self.test_bin.shape:
            raise ValueError(
                f"Template shape {self.template_bin.shape} does not match "
                f"test shape {self.test_bin.shape}"
            )
        for name, image in (("template_bin", self.template_bin),
                            ("test_bin", self.test_bin)):
            if image.dtype != np.uint8:
                raise TypeError(f"{name} must be uint8, got {image.dtype}")
            if not np.isin(np.unique(image), (0, 255)).all():
                raise ValueError(f"{name} must be binary with values 0 or 255")
        if self.mm_per_px <= 0:
            raise ValueError(f"mm_per_px must be positive, got {self.mm_per_px}")


# --------------------------------------------------------------------------
# Module 2 -> Module 3
# --------------------------------------------------------------------------
@dataclass
class Blob:
    """One connected region of difference between the template and the test image.

    A blob is a *candidate* defect. It carries no class label: assigning the
    class is Module 3's responsibility.
    """

    id: int
    bbox: tuple[int, int, int, int]   # (x, y, w, h) in pixels, top-left origin
    contour: np.ndarray               # OpenCV contour, shape (N, 1, 2)
    centroid: tuple[float, float]     # (cx, cy) in pixels
    area_px: int
    polarity: Polarity


@dataclass
class LocalisationResult:
    """Output of the registration and localisation module.

    align_residual is the mean reprojection error of the inlier keypoint
    matches, in pixels. A large value means the pair was badly registered and
    every blob downstream is suspect, so it is surfaced rather than hidden.
    """

    blobs: list[Blob]
    align_residual: float
    registration_ok: bool = True
    method: str = ""                  # which registration path actually ran


# --------------------------------------------------------------------------
# Module 3 output
# --------------------------------------------------------------------------
@dataclass
class Defect:
    """A classified, physically measured defect.

    Amendment raised by Ng Zhi Xuan, tasks 3.2 to 3.4. Everything below
    ``confidence`` is new and every new field carries a default, so existing
    callers that construct a Defect with the original five arguments continue
    to work unchanged. Closes open questions 5 and 6 in docs/MEETING_LOG.md.
    """

    id: int
    bbox: tuple[int, int, int, int]
    defect_class: str                 # one of DEFECT_CLASSES
    area_mm2: float
    confidence: float                 # 0.5 to 1.0, a decision margin not a probability

    # Carried rather than re-derived. The dashboard previously recovered
    # polarity from the class lookup table in Module 1, which meant a display
    # concern reached into another member's module for a value Module 2 had
    # already measured.
    polarity: Polarity | str = "removed"

    # Physical dimensions from the rotated minimum-area rectangle, not the
    # axis-aligned bounding box. The dashboard and the PDF report previously
    # multiplied the axis-aligned box by a hard-coded constant, which
    # mismeasured every diagonal defect.
    width_mm: float = 0.0
    height_mm: float = 0.0

    # Ordinal severity weight used by the verdict rule. See SEVERITY in
    # src/module3/classify.py.
    severity: int = 1

    # Which rule set decided this class: "connectivity" or "descriptor". Needed
    # so that Chapter 4 can report the fallback rate rather than presenting one
    # classifier's results as though the other never ran.
    decided_by: str = ""


@dataclass
class InspectionReport:
    """The final answer for one template-test pair."""

    defects: list[Defect]
    verdict: Verdict
    runtime_s: float
    align_residual: float = 0.0

    # Module 1's calibration factor, surfaced so that any consumer can convert
    # a pixel reading without importing Module 1 or assuming a constant.
    mm_per_px: float = 0.0

    # A bare PASS or FAIL is not an inspection result an operator can act on,
    # so the deciding condition travels with it.
    verdict_reason: str = ""
    verdict_detail: dict[str, Any] = field(default_factory=dict)

    # Which classifier produced these labels, recorded on the result itself so
    # that a saved report cannot be misattributed later.
    classifier: str = ""
