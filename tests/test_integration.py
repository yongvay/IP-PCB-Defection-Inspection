"""Integration regression tests.

These lock in the four findings from the Week 2 integration run. Each one cost
real debugging time, and each would fail silently if a later change reverted
it, so each gets a test rather than a comment.

Run with:  python -m pytest tests -v
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.contracts import PreprocessResult
from src.module2 import registration
from src.module1.ground_truth import parse_annotation_file, parse_all
from src.module1.preprocess import preprocess_pair
from src.module2 import blobs as blob_extraction
from src.module2 import difference
from src.module3.evaluate import iou, score_pair
from src.pipeline import inspect_pair

REPO_ROOT = Path(__file__).resolve().parents[1]
PCB_DATA = REPO_ROOT / "data" / "DeepPCB" / "PCBData"
SAMPLE_GROUP = PCB_DATA / "group00041"
SAMPLE_TEMPLATE = SAMPLE_GROUP / "00041" / "00041000_temp.jpg"
SAMPLE_TEST = SAMPLE_GROUP / "00041" / "00041000_test.jpg"
SAMPLE_ANNOTATION = SAMPLE_GROUP / "00041_not" / "00041000.txt"

needs_dataset = pytest.mark.skipif(
    not SAMPLE_TEMPLATE.exists(),
    reason="DeepPCB is not committed to the repository; download it into data/",
)


# --- Ground truth (Module 1, Xing Szen) ---------------------------------
@needs_dataset
def test_ground_truth_parses_whole_dataset():
    """The parser must read all 1,500 annotated pairs, not merely not crash."""
    files, boxes = parse_all(PCB_DATA)
    assert len(files) == 1500
    assert len(boxes) == 10013


@needs_dataset
def test_ground_truth_boxes_are_width_height_not_corners():
    """Corner coordinates must be converted, or every downstream box is wrong."""
    boxes = parse_annotation_file(SAMPLE_ANNOTATION)
    for box in boxes:
        _, _, w, h = box["bbox"]
        assert w > 0 and h > 0


# --- Polarity convention (Module 2, Yong Vay) ---------------------------
def test_copper_is_dark_convention_is_applied():
    """Copper is dark in DeepPCB. Getting this backwards inverts every class.

    A synthetic pair: the template has a dark square (copper), the test image
    does not. That is copper removed, and it must be reported as removed.
    """
    template = np.full((64, 64), 255, dtype=np.uint8)
    template[20:40, 20:40] = 0          # a patch of copper
    test = np.full((64, 64), 255, dtype=np.uint8)   # copper missing

    removed, added = difference.signed_difference(template, test)

    assert removed.sum() > 0, "copper present in template but not test is 'removed'"
    assert added.sum() == 0


def test_polarity_reverses_when_convention_flipped():
    """Guards the copper_is_dark switch itself against being ignored."""
    template = np.full((64, 64), 255, dtype=np.uint8)
    template[20:40, 20:40] = 0
    test = np.full((64, 64), 255, dtype=np.uint8)

    removed, added = difference.signed_difference(template, test, copper_is_dark=False)
    assert added.sum() > 0 and removed.sum() == 0


@needs_dataset
def test_polarity_matches_ground_truth_on_real_board():
    """Every blob landing inside a ground-truth box must carry its polarity."""
    prepared = preprocess_pair(SAMPLE_TEMPLATE, SAMPLE_TEST)
    removed, added = difference.signed_difference(
        prepared.template_bin, prepared.test_bin
    )
    found = blob_extraction.extract_blobs(removed, added, min_area=20)
    truth = parse_annotation_file(SAMPLE_ANNOTATION)

    checked = 0
    for defect in truth:
        x, y, w, h = defect["bbox"]
        inside = [b for b in found if x <= b.centroid[0] <= x + w
                  and y <= b.centroid[1] <= y + h]
        if not inside:
            continue
        largest = max(inside, key=lambda b: b.area_px)
        assert largest.polarity == defect["polarity"], (
            f"{defect['label']} should be {defect['polarity']}, "
            f"got {largest.polarity}"
        )
        checked += 1

    assert checked >= 8, f"expected to check at least 8 defects, checked {checked}"


# --- Annotation padding convention --------------------------------------
@needs_dataset
def test_padding_is_required_to_score_against_ground_truth():
    """Tight boxes score near zero; the same detections padded score well.

    This test exists to make the padding impossible to remove by accident. If
    it ever fails, the reported F1 in Chapter 4 is meaningless.
    """
    report = inspect_pair(str(SAMPLE_TEMPLATE), str(SAMPLE_TEST))
    truth = [d["bbox"] for d in parse_annotation_file(SAMPLE_ANNOTATION)]

    tight = [d.bbox for d in report.defects]
    padded = [blob_extraction.to_evaluation_bbox(b, (640, 640)) for b in tight]

    tight_score = score_pair(tight, truth, iou_threshold=0.5)
    padded_score = score_pair(padded, truth, iou_threshold=0.5)

    assert padded_score.recall > tight_score.recall


# --- Registration (Module 2, Yong Vay) ----------------------------------
@needs_dataset
def test_registration_leaves_aligned_pairs_alone():
    """DeepPCB pairs are pre-aligned, so warping them can only do harm."""
    prepared = preprocess_pair(SAMPLE_TEMPLATE, SAMPLE_TEST)
    result = registration.register(prepared.test_bin, prepared.template_bin)
    assert result.method == "identity"


@needs_dataset
def test_registration_corrects_a_shifted_board():
    """The feature path must still fire when there is real misalignment."""
    prepared = preprocess_pair(SAMPLE_TEMPLATE, SAMPLE_TEST)
    shift = np.float32([[1, 0, 15], [0, 1, 15]])
    shifted = cv2.warpAffine(
        prepared.test_bin, shift, (640, 640),
        flags=cv2.INTER_NEAREST, borderValue=255,
    )

    before = registration.disagreement(shifted, prepared.template_bin)
    result = registration.register(shifted, prepared.template_bin)
    after = registration.disagreement(result.aligned, prepared.template_bin)

    assert result.method != "identity", "a 15 px shift must be corrected"
    assert after < before


@needs_dataset
def test_registration_corrects_a_rotated_board():
    """Rotation is why the feature path exists rather than phase correlation."""
    prepared = preprocess_pair(SAMPLE_TEMPLATE, SAMPLE_TEST)
    rotation = cv2.getRotationMatrix2D((320, 320), 3.0, 1.0)
    rotated = cv2.warpAffine(
        prepared.test_bin, rotation, (640, 640),
        flags=cv2.INTER_NEAREST, borderValue=255,
    )

    before = registration.disagreement(rotated, prepared.template_bin)
    after = registration.disagreement(
        registration.register(rotated, prepared.template_bin).aligned,
        prepared.template_bin,
    )
    assert after < before


# --- Contract enforcement ------------------------------------------------
def test_contract_rejects_mismatched_shapes():
    """The boundary check must fire before Module 2 sees bad input."""
    result = PreprocessResult(
        template_bin=np.zeros((64, 64), dtype=np.uint8),
        test_bin=np.zeros((32, 32), dtype=np.uint8),
        mm_per_px=0.02,
    )
    with pytest.raises(ValueError, match="does not match"):
        result.validate()


def test_contract_rejects_non_binary_images():
    grey = np.full((64, 64), 128, dtype=np.uint8)
    result = PreprocessResult(template_bin=grey, test_bin=grey, mm_per_px=0.02)
    with pytest.raises(ValueError, match="binary"):
        result.validate()


# --- Evaluation ----------------------------------------------------------
def test_iou_of_identical_boxes_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (50, 50, 10, 10)) == 0.0


def test_each_truth_box_is_claimed_at_most_once():
    """Two overlapping predictions on one defect is one hit and one false alarm."""
    truth = [(0, 0, 10, 10)]
    predictions = [(0, 0, 10, 10), (1, 1, 10, 10)]
    scores = score_pair(predictions, truth, iou_threshold=0.5)
    assert scores.true_positives == 1
    assert scores.false_positives == 1


# --- End to end ----------------------------------------------------------
@needs_dataset
def test_pipeline_runs_end_to_end_within_objective_3_budget():
    """Objective 3 requires under 3 seconds per board."""
    report = inspect_pair(str(SAMPLE_TEMPLATE), str(SAMPLE_TEST))
    assert report.verdict in ("PASS", "FAIL")
    assert report.runtime_s < 3.0
    assert all(d.area_mm2 > 0 for d in report.defects)
