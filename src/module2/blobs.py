"""Blob extraction — Task 2.4.

Owner: Ng Yong Vay. Scheduled for Week 4; signature fixed now so Module 3 can
be written against real Blob objects rather than invented ones.

Connected-component analysis groups touching foreground pixels into regions.
Each surviving region becomes one candidate defect, carrying the polarity of
the difference image it came from. No class is assigned here: that is Module
3's job, and keeping the boundary strict is what allows the classifier to be
rewritten without touching this file.
"""

import cv2
import numpy as np

from src.contracts import Blob, Polarity

# Regions smaller than this are treated as residual noise rather than defects.
# Measured on a 30-pair DeepPCB sample: 40 px gives F1 0.82, whereas raising it
# to 100 px collapses recall to 0.36 because genuine pin holes are only 70 to
# 90 pixels in area. Noise removal is therefore done by morphological opening,
# not by an area threshold, and this value only catches the residue.
MIN_BLOB_AREA_PX = 40

# DeepPCB annotates each defect with a box drawn around the defect region with
# a consistent margin, not a tight box around the changed pixels. Measured on
# group00041, every ground-truth corner sits almost exactly 10 pixels outside
# the corresponding difference blob.
#
# This is not a detail. Comparing tight predicted boxes against padded
# ground-truth boxes gives recall 0.01 at IoU 0.5; applying the same 10-pixel
# convention to the predictions gives recall 0.71 from identical detections.
# The pipeline is not better, the comparison is finally like for like. The
# report must state this explicitly in Chapter 3, because a marker who sees
# only the corrected figure has no way to tell the difference between an
# annotation convention and a thumb on the scale.
GT_BOX_PADDING_PX = 10


def extract_blobs(removed: np.ndarray,
                  added: np.ndarray,
                  min_area: int = MIN_BLOB_AREA_PX) -> list[Blob]:
    """Convert the two signed difference images into a list of candidate defects."""
    blobs: list[Blob] = []
    next_id = 0

    for image, polarity in ((removed, "removed"), (added, "added")):
        blobs_found, next_id = _extract_one_polarity(image, polarity, min_area, next_id)
        blobs.extend(blobs_found)

    return blobs


def to_evaluation_bbox(bbox: tuple[int, int, int, int],
                       image_shape: tuple[int, int],
                       padding: int = GT_BOX_PADDING_PX) -> tuple[int, int, int, int]:
    """Expand a tight detection box to the DeepPCB annotation convention.

    Used only when scoring against ground truth, never when measuring a defect
    physically: the padding is an artefact of how the dataset was labelled, so
    including it in an area measurement would inflate every reading.
    """
    x, y, w, h = bbox
    height, width = image_shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)
    return (x1, y1, x2 - x1, y2 - y1)


def _extract_one_polarity(image: np.ndarray,
                          polarity: Polarity,
                          min_area: int,
                          next_id: int) -> tuple[list[Blob], int]:
    """Run connected-component analysis on a single polarity image."""
    # Connectivity 8 counts diagonal neighbours as connected, which keeps a
    # thin diagonal spur as one region instead of a broken chain of specks.
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        image, connectivity=8
    )

    blobs: list[Blob] = []
    # Label 0 is the background, so enumeration starts at 1.
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        # Module 3 needs the outline itself for its shape descriptors, not just
        # the bounding box, so the contour is traced from an isolated mask of
        # this one component.
        mask = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0] if contours else np.empty((0, 1, 2), dtype=np.int32)

        blobs.append(Blob(
            id=next_id,
            bbox=(x, y, w, h),
            contour=contour,
            centroid=(float(centroids[label][0]), float(centroids[label][1])),
            area_px=area,
            polarity=polarity,
        ))
        next_id += 1

    return blobs, next_id
