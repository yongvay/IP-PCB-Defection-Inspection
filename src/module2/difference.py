"""Golden-template differencing — Task 2.3.

Owner: Ng Yong Vay.

The insight the whole classifier rests on: a defect is not merely a
difference, it is a *signed* difference. Copper present in the template but
absent from the test image means copper was removed, which can only be an
open circuit, a mouse bite or a pin hole. Copper present in the test image but
absent from the template means copper was added, which can only be a short, a
spur or spurious copper. Splitting the difference by sign therefore solves
half the classification problem in two bitwise operations, before any shape
descriptor is computed.

Polarity convention — verified empirically, do not assume it
-----------------------------------------------------------
In the DeepPCB binarised images copper is **dark** and the substrate is
bright: roughly 86% of every image is white, and an open circuit (copper
removed) makes its region brighter, not darker. Assuming the intuitive
"copper is white" convention silently inverts every polarity label, which
inverts stage one of the Module 3 classifier and makes every class prediction
wrong while the bounding boxes still look correct. That failure was observed
during integration testing on group00041, so the convention is now an explicit
parameter with a verified default rather than an unwritten assumption.
"""

import cv2
import numpy as np

# True for DeepPCB: copper renders dark against a bright substrate.
# Confirm before switching datasets — HRIPCB is colour and may differ.
COPPER_IS_DARK = True


def copper_mask(binary: np.ndarray, copper_is_dark: bool = COPPER_IS_DARK) -> np.ndarray:
    """Return a mask in which 255 always means copper, whatever the input polarity.

    Normalising here means the rest of the system can reason in terms of
    copper rather than in terms of pixel brightness.
    """
    return cv2.bitwise_not(binary) if copper_is_dark else binary


def signed_difference(template_bin: np.ndarray,
                      test_bin: np.ndarray,
                      copper_is_dark: bool = COPPER_IS_DARK
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Split the template-test difference into copper-removed and copper-added.

    Returns (removed, added), both binary images the same shape as the inputs,
    where 255 marks a differing pixel.
    """
    if template_bin.shape != test_bin.shape:
        raise ValueError(
            f"Shape mismatch: template {template_bin.shape} vs test {test_bin.shape}"
        )

    template_copper = copper_mask(template_bin, copper_is_dark)
    test_copper = copper_mask(test_bin, copper_is_dark)

    # Copper in the template that is missing from the test image. = Copper Removed
    removed = cv2.bitwise_and(template_copper, cv2.bitwise_not(test_copper))
    # Copper in the test image that is not in the template. = Copper Added
    added = cv2.bitwise_and(test_copper, cv2.bitwise_not(template_copper))

    return removed.astype(np.uint8), added.astype(np.uint8)
