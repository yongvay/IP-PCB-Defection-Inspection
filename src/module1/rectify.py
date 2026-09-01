"""Geometric rectification — Module 1, task 1.8.

Owner: Chan Xing Szen.

Golden-template inspection compares a board under test against a reference
board pixel by pixel. That comparison is only meaningful if the two boards
occupy the same position in their images, and on HRIPCB they do not: the
dataset authors rephotographed each board turned through an angle to simulate
one placed carelessly on the workbench. Left uncorrected, the entire board
registers as a difference and every trace becomes a defect.

Correcting orientation before comparison is what the assignment specification
means by rectification, and it is the reason the requirement exists at all.

Why the board's own outline is the right cue
--------------------------------------------
A PCB is a rectangle of substrate against a contrasting background, which is
the easiest possible object to segment: threshold, take the largest connected
region, fit a minimum-area rectangle, and that rectangle's angle is the board's
angle. Nothing about the traces, the components or the defects is involved, so
a heavily defective board rectifies exactly as well as a clean one.

Measured, not asserted
----------------------
HRIPCB records the angle it applied to every rotated image in
``rotation/*_angles.txt``. That makes this the one part of the pipeline with a
directly checkable answer: the estimate can be compared against the true angle
in degrees. experiments/benchmark_rectify.py reports that error, so Chapter 4
carries a measured accuracy rather than a pair of before-and-after screenshots.

Author: Chan Xing Szen (Member A, Module 1)
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def board_mask(image: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
    """Segment the board from the background as a filled binary mask.

    Two strategies, tried in order, because "background" means two different
    things across the HRIPCB set.

    **Flood fill from the corners.** When a board has been rotated inside its
    frame, the exposed corners are filled with one uniform value, and a board
    photographed on a bench sits against one uniform surface. In both cases the
    background is the region reachable from a corner without crossing an edge,
    which is exactly what a flood fill finds. Thresholding cannot substitute
    here: on HRIPCB the padding introduced by rotation is mid-grey, *brighter*
    than the dark green board, so an intensity rule labels the padding as
    foreground and the board as background.

    **Otsu on the whole frame.** The fallback, used when flood filling escapes
    into the board and swallows most of the image. That happens for the upright
    HRIPCB images, where the board fills the frame edge to edge and there is no
    background at all — and returning the whole frame is then the right answer,
    because a board that fills its frame is already square with it.
    """
    params = params or {}
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    flooded = _mask_by_corner_flood(grey, params)
    return flooded if flooded is not None else _mask_by_threshold(grey, params)


def _mask_by_corner_flood(grey: np.ndarray,
                          params: dict[str, Any]) -> np.ndarray | None:
    """Mark everything reachable from a corner as background; return the rest.

    The tolerance is not a fixed constant but a ladder, tried from strict to
    permissive, accepting the first setting whose background covers a plausible
    fraction of the frame. That is necessary because the tolerance has a cliff
    rather than a gentle optimum: measured on a board rotated by 4 degrees,
    whose padding must account for 14.5% of the frame, the flood covers

        tolerance     0      1      2      3      5     12
        background  14.2%  16.4%  16.9%  80.1%  92.4%  99.7%

    Below the cliff the fill traces the board outline; one step above it, the
    fill leaks through the anti-aliased boundary and swallows the board. A
    single hard-coded tolerance would therefore be tuned to one dataset's
    padding and would fail silently on any other, whereas a ladder that checks
    its own result adapts and reports honestly when nothing works.

    Returns None when no tolerance yields a plausible background, which is the
    signal that this image has no uniform background at all and the caller
    should fall back to thresholding.
    """
    height, width = grey.shape[:2]
    low = params.get("flood_min_background", 0.02)
    high = params.get("flood_max_background", 0.60)
    seeds = ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1))

    for tolerance in params.get("flood_tolerances", (2, 6, 12, 20)):
        # floodFill needs a mask two pixels larger than the image in each direction.
        scratch = np.zeros((height + 2, width + 2), np.uint8)
        canvas = grey.copy()
        for seed in seeds:
            cv2.floodFill(canvas, scratch, seed, 255,
                          loDiff=int(tolerance), upDiff=int(tolerance),
                          flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8) | 8)

        background = scratch[1:-1, 1:-1]
        covered = float(np.count_nonzero(background)) / background.size
        if low <= covered <= high:
            board = cv2.bitwise_not(background)
            size = int(params.get("board_close_kernel", 25))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            return cv2.morphologyEx(board, cv2.MORPH_CLOSE, kernel)

    return None


def board_rect(image: np.ndarray, params: dict[str, Any] | None = None):
    """Return the minimum-area rectangle enclosing the board, or None."""
    return board_rect_detailed(image, params)[0]


def board_rect_detailed(image: np.ndarray, params: dict[str, Any] | None = None):
    """As board_rect, but also report which segmentation strategy produced it.

    The caller needs to know this. When flood filling fails and the threshold
    fallback returns the whole frame, the fitted rectangle is the image itself
    and its angle is zero — a confident-looking answer that is really "no board
    was found". Measured on 120 rotated HRIPCB boards, six such boards reported
    exactly 0 degrees against true angles as large as 10, and because a returned
    number is indistinguishable from a successful estimate they were scored as
    detections with large errors rather than as the failures they are.
    """
    params = params or {}
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = _mask_by_corner_flood(grey, params)
    method = "flood"
    if mask is None:
        mask = _mask_by_threshold(grey, params)
        method = "threshold"
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, method

    largest = max(contours, key=cv2.contourArea)
    # A contour that covers almost nothing is a segmentation failure, not a board.
    if cv2.contourArea(largest) < 0.05 * grey.shape[0] * grey.shape[1]:
        return None, method
    return cv2.minAreaRect(largest), method


def _mask_by_threshold(grey: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Otsu fallback: used when no uniform background could be flood filled."""
    smoothed = cv2.GaussianBlur(grey, (5, 5), 0)
    _, mask = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(mask) > mask.size * 0.5:
        mask = cv2.bitwise_not(mask)
    size = int(params.get("board_close_kernel", 25))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def estimate_angle(image: np.ndarray, params: dict[str, Any] | None = None) -> float | None:
    """Estimate the board's rotation in degrees, or None if it cannot be found.

    OpenCV reports a rectangle's angle in a quarter turn, and which side it
    calls the width depends on the corner ordering rather than on anything
    physical. A board turned by 88 degrees and one turned by -2 are the same
    placement, so the estimate is folded into (-45, 45] before being returned.
    Skipping that step produces occasional 90-degree errors that look like a
    broken estimator but are only a convention mismatch.
    """
    return estimate_angle_detailed(image, params)[0]


def estimate_angle_detailed(image: np.ndarray,
                            params: dict[str, Any] | None = None
                            ) -> tuple[float | None, str]:
    """As estimate_angle, but also report the segmentation strategy used."""
    rect, method = board_rect_detailed(image, params)
    if rect is None:
        return None, method
    return normalise_angle(rect[2]), method


def normalise_angle(angle: float) -> float:
    """Fold any angle into (-45, 45], the range of distinct board placements."""
    angle = float(angle) % 90.0
    return angle - 90.0 if angle > 45.0 else angle


def deskew(image: np.ndarray, angle: float,
           border_value: int | tuple = 255) -> np.ndarray:
    """Turn an image about its centre so its measured angle increases by ``angle``.

    Stated in terms of the measurement rather than of OpenCV's rotation sign,
    because the two conventions do not agree and assuming they do is how a
    correction becomes a doubling. Concretely: if ``estimate_angle`` reports a
    on an image, it reports a + b after ``deskew(image, b)``. Levelling an
    image therefore means calling ``deskew(image, -a)``, which is what
    ``rectify`` does and what the round-trip test pins down.

    The frame is deliberately not expanded to fit the rotated content. The
    contract requires the template and the test image to share a shape, and a
    board turned by a few degrees still lies well inside its own frame, so
    growing the canvas would break the contract to solve a problem that does not
    arise. The border fills with the substrate value so that the exposed corners
    do not read as copper and become defects.
    """
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), -angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (width, height),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def rectify(image: np.ndarray,
            params: dict[str, Any] | None = None) -> tuple[np.ndarray, float | None]:
    """Detect the board's orientation and level it. Returns the image and the angle.

    A board already within a fraction of a degree of square is left untouched.
    Every warp resamples, and resampling a binary image adds jitter along every
    trace edge, which is precisely the false-positive source Module 1 exists to
    suppress. Correcting an angle that was never there costs accuracy and buys
    nothing.
    """
    params = params or {}
    angle = estimate_angle(image, params)
    if angle is None:
        return image, None
    if abs(angle) < params.get("min_rectify_angle", 0.25):
        return image, angle
    # Negated: deskew adds to the measured angle, so levelling subtracts it.
    # Passing +angle here turns the board further instead of straightening it,
    # doubling a 7 degree tilt into 14 while still returning a plausible image.
    return deskew(image, -angle, params.get("rectify_border_value", 255)), angle


def four_point_transform(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp a quadrilateral onto a rectangle: perspective rather than rotation.

    Rotation is enough for HRIPCB, whose boards were photographed square-on and
    turned in the plane. This is the general case, for a board imaged from an
    angle so that its far edge is foreshortened, and it is retained because the
    specification asks for rectification rather than deskewing alone.
    """
    ordered = _order_corners(corners)
    (top_left, top_right, bottom_right, bottom_left) = ordered

    width = int(max(np.linalg.norm(bottom_right - bottom_left),
                    np.linalg.norm(top_right - top_left)))
    height = int(max(np.linalg.norm(top_right - bottom_right),
                     np.linalg.norm(top_left - bottom_left)))
    if width < 2 or height < 2:
        raise ValueError("Degenerate quadrilateral: cannot rectify")

    destination = np.array([[0, 0], [width - 1, 0],
                            [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(ordered.astype(np.float32), destination)
    return cv2.warpPerspective(image, matrix, (width, height))


def _order_corners(corners: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left.

    The sum of a point's coordinates is smallest at the top-left corner and
    largest at the bottom-right; their difference separates the other two. This
    holds for any rotation short of a quarter turn, which is why it is preferred
    to sorting by x and y separately.
    """
    points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    total = points.sum(axis=1)
    difference = np.diff(points, axis=1).ravel()
    return np.array([
        points[np.argmin(total)],
        points[np.argmin(difference)],
        points[np.argmax(total)],
        points[np.argmax(difference)],
    ], dtype=np.float32)


def crop_to_board(image: np.ndarray,
                  params: dict[str, Any] | None = None) -> np.ndarray:
    """Trim an image down to the board's own bounding box.

    Necessary after deskewing, because rotating a board inside a frame leaves
    padding on every side, and the padded frame no longer covers the same
    physical extent as the reference board's frame. Comparing the two directly
    would mean comparing a board against a scaled copy of itself, which registers
    as a difference everywhere.
    """
    rect, method = board_rect_detailed(image, params)
    if rect is None or method != "flood":
        return image

    box = cv2.boxPoints(rect)
    x, y, width, height = cv2.boundingRect(box.astype(np.int32))
    x, y = max(0, x), max(0, y)
    width = min(width, image.shape[1] - x)
    height = min(height, image.shape[0] - y)
    if width < 2 or height < 2:
        return image
    return image[y:y + height, x:x + width]


def rectify_to_template(image: np.ndarray,
                        template_shape: tuple[int, ...],
                        params: dict[str, Any] | None = None
                        ) -> tuple[np.ndarray, float | None]:
    """Level a board, trim it to its own extent, and scale it onto the template.

    The order is the whole point, and getting it wrong makes rectification
    actively harmful. Measured on ten rotated HRIPCB boards, resizing to the
    template *before* deskewing scores F1 0.286 against 0.387 for not
    rectifying at all: the resize squeezes a 1793 x 3137 rotated canvas into the
    template's 1586 x 3034, changing the aspect ratio, and the subsequent
    rotation then aligns a board that is no longer the right shape.

    Levelling first, trimming the padding the rotation introduced, and only then
    scaling what remains onto the template means each step operates on a board
    that still matches the reference in orientation and extent.
    """
    levelled, angle = rectify(image, params)
    cropped = crop_to_board(levelled, params)
    resized = cv2.resize(cropped, (template_shape[1], template_shape[0]),
                         interpolation=cv2.INTER_AREA)
    return resized, angle
