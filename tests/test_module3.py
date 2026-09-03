"""Module 3 regression tests — task 3.2 and task 3.7.

Owner: Ng Zhi Xuan.

The DeepPCB dataset is not committed, so these tests build a synthetic board in
memory: a bright substrate carrying dark copper traces and pads, into which one
defect of a known class is cut or added. That is enough to prove the rules do
what they claim, and unlike a dataset test it runs anywhere and cannot fail
because a folder is missing.

The synthetic board is a test fixture, not evidence. Chapter 4 quotes the
DeepPCB figures produced by experiments/benchmark_module3.py; these tests only
guarantee the rules have not been broken by an edit.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.contracts import Blob
from src.module2 import blobs as blob_extraction
from src.module2 import difference
from src.module3 import classify, connectivity, descriptors
from src.module3.evaluate import BACKGROUND, Evaluation, iou, match_boxes

BOARD_SIZE = 200
COPPER = 0        # copper renders dark, as it does in DeepPCB
SUBSTRATE = 255


def blank_board() -> np.ndarray:
    return np.full((BOARD_SIZE, BOARD_SIZE), SUBSTRATE, dtype=np.uint8)


def reference_board() -> np.ndarray:
    """A defect-free board: two horizontal traces and one pad."""
    board = blank_board()
    cv2.rectangle(board, (20, 40), (180, 52), COPPER, thickness=cv2.FILLED)
    cv2.rectangle(board, (20, 90), (180, 102), COPPER, thickness=cv2.FILLED)
    cv2.circle(board, (100, 150), 22, COPPER, thickness=cv2.FILLED)
    return board


def board_with(defect: str) -> np.ndarray:
    """The reference board with exactly one defect of the named class."""
    board = reference_board()

    if defect == "open_circuit":
        # A clean break straight through the upper trace.
        cv2.rectangle(board, (95, 38), (107, 54), SUBSTRATE, thickness=cv2.FILLED)
    elif defect == "mouse_bite":
        # A notch out of the lower edge of the upper trace, open to substrate.
        cv2.circle(board, (70, 52), 7, SUBSTRATE, thickness=cv2.FILLED)
    elif defect == "pin_hole":
        # A hole punched through the middle of the pad, copper all around it.
        cv2.circle(board, (100, 150), 8, SUBSTRATE, thickness=cv2.FILLED)
    elif defect == "short":
        # A bridge joining the two traces.
        cv2.rectangle(board, (140, 46), (152, 96), COPPER, thickness=cv2.FILLED)
    elif defect == "spur":
        # A stub hanging off the upper trace, attached at one end only.
        cv2.rectangle(board, (50, 52), (60, 74), COPPER, thickness=cv2.FILLED)
    elif defect == "spurious_copper":
        # An island touching nothing.
        cv2.circle(board, (40, 170), 9, COPPER, thickness=cv2.FILLED)
    else:
        raise ValueError(f"unknown defect class {defect}")

    return board


def localise(template: np.ndarray, test: np.ndarray) -> list[Blob]:
    """Run the real Module 2 path over a synthetic pair."""
    removed, added = difference.signed_difference(template, test)
    return blob_extraction.extract_blobs(removed, added, min_area=20)


def context_for(template: np.ndarray, test: np.ndarray) -> connectivity.BoardContext:
    return connectivity.BoardContext(
        template_copper=difference.copper_mask(template),
        test_copper=difference.copper_mask(test),
    )


# ---------------------------------------------------------------------------
# Task 3.1 — descriptors
# ---------------------------------------------------------------------------
def test_descriptors_report_a_disc_as_circular_and_a_bar_as_elongated():
    disc = blank_board()
    cv2.circle(disc, (100, 100), 20, COPPER, thickness=cv2.FILLED)
    bar = blank_board()
    cv2.rectangle(bar, (40, 96), (160, 104), COPPER, thickness=cv2.FILLED)

    measurements = {}
    for name, image in (("disc", disc), ("bar", bar)):
        mask = cv2.bitwise_not(image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        measurements[name] = descriptors.extract_descriptors({
            "contour": contours[0],
            "area_px": int(np.count_nonzero(mask)),
        })

    assert measurements["disc"]["circularity"] > 0.8
    assert measurements["disc"]["aspect_ratio"] < 1.2
    assert measurements["disc"]["eccentricity"] < 0.3

    assert measurements["bar"]["circularity"] < 0.4
    assert measurements["bar"]["aspect_ratio"] > 10
    assert measurements["bar"]["eccentricity"] > 0.9


def test_every_descriptor_the_report_promises_is_present():
    """The plan lists seven descriptors by name; none may quietly go missing."""
    board = blank_board()
    cv2.circle(board, (100, 100), 15, COPPER, thickness=cv2.FILLED)
    mask = cv2.bitwise_not(board)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    measured = descriptors.extract_descriptors({
        "contour": contours[0], "area_px": int(np.count_nonzero(mask)),
    })
    for name in ("area_px", "perimeter_px", "aspect_ratio", "solidity",
                 "extent", "eccentricity", "circularity", "hu_moments"):
        assert name in measured, f"descriptor {name} is missing"
    assert len(measured["hu_moments"]) == 7


# ---------------------------------------------------------------------------
# Task 3.2 — stage one, polarity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("defect_class,expected_polarity", [
    ("open_circuit", "removed"),
    ("mouse_bite", "removed"),
    ("pin_hole", "removed"),
    ("short", "added"),
    ("spur", "added"),
    ("spurious_copper", "added"),
])
def test_stage_one_polarity_matches_the_taxonomy(defect_class, expected_polarity):
    template = reference_board()
    test = board_with(defect_class)
    found = localise(template, test)

    assert found, f"no blob extracted for {defect_class}"
    assert all(blob.polarity == expected_polarity for blob in found)


# ---------------------------------------------------------------------------
# Task 3.2 — stage two, connectivity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("defect_class", [
    "open_circuit", "mouse_bite", "pin_hole", "short", "spur", "spurious_copper",
])
def test_connectivity_classifier_recovers_every_class(defect_class):
    template = reference_board()
    test = board_with(defect_class)
    found = localise(template, test)
    context = context_for(template, test)

    assert len(found) == 1, f"{defect_class} produced {len(found)} blobs, expected 1"
    predicted, confidence, decided_by = classify.classify(
        found[0], context, method="connectivity"
    )

    assert predicted == defect_class
    assert decided_by == "connectivity"
    assert 0.5 <= confidence <= 1.0


def test_connectivity_separates_the_pairs_the_descriptors_cannot():
    """The claim Chapter 4 rests on: shape alone confuses these, context does not."""
    ambiguous = ("open_circuit", "mouse_bite")
    template = reference_board()

    descriptor_labels, connectivity_labels = [], []
    for defect_class in ambiguous:
        test = board_with(defect_class)
        blob = localise(template, test)[0]
        context = context_for(template, test)
        descriptor_labels.append(classify.classify(blob, None, "descriptor")[0])
        connectivity_labels.append(classify.classify(blob, context, "connectivity")[0])

    assert list(ambiguous) == connectivity_labels
    assert descriptor_labels != list(ambiguous), (
        "the baseline is expected to confuse this pair; if it no longer does, "
        "the Chapter 4 comparison needs rewording"
    )


def test_classifier_falls_back_when_the_context_is_unreadable():
    """A degenerate contour must not crash the run, and must be recorded."""
    blob = Blob(
        id=0, bbox=(10, 10, 0, 0), contour=np.empty((0, 1, 2), dtype=np.int32),
        centroid=(10.0, 10.0), area_px=0, polarity="removed",
    )
    context = context_for(reference_board(), reference_board())
    _, _, decided_by = classify.classify(blob, context, method="connectivity")
    assert decided_by == "descriptor"


# ---------------------------------------------------------------------------
# Task 3.3 — physical measurement
# ---------------------------------------------------------------------------
def test_area_scales_with_the_square_of_the_calibration_factor():
    blob = Blob(id=0, bbox=(0, 0, 10, 10), contour=np.empty((0, 1, 2), dtype=np.int32),
                centroid=(5.0, 5.0), area_px=100, polarity="removed")
    # 48 px/mm, so 100 px^2 is 100 / 48^2 mm^2. Applying the factor once
    # instead of twice would over-report by a factor of 48.
    assert classify.measure(blob, 1 / 48) == pytest.approx(100 / (48 ** 2))


def test_dimensions_come_from_the_rotated_box_not_the_axis_aligned_one():
    """A diagonal bar is as long as it is, not as wide as its bounding box."""
    board = blank_board()
    cv2.line(board, (60, 60), (140, 140), COPPER, thickness=6)
    mask = cv2.bitwise_not(board)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = descriptors.extract_descriptors({
        "contour": contours[0], "area_px": int(np.count_nonzero(mask)),
    })

    length_mm, width_mm = classify.measure_dimensions(features, 1.0)
    axis_aligned_width = float(cv2.boundingRect(contours[0])[2])

    assert length_mm > axis_aligned_width, "rotated box should be longer than the AABB side"
    assert width_mm < 12


# ---------------------------------------------------------------------------
# Task 3.4 — verdict
# ---------------------------------------------------------------------------
def make_defect(defect_class: str, defect_id: int = 0):
    from src.contracts import Defect
    return Defect(id=defect_id, bbox=(0, 0, 5, 5), defect_class=defect_class,
                  area_mm2=0.01, confidence=0.9,
                  severity=classify.SEVERITY[defect_class])


def test_clean_board_passes():
    verdict, detail = classify.decide_verdict([])
    assert verdict == "PASS"
    assert detail["total_severity"] == 0


def test_a_single_short_fails_however_generous_the_tolerance():
    verdict, detail = classify.decide_verdict(
        [make_defect("short")], {"max_defects": 99}
    )
    assert verdict == "FAIL"
    assert "critical" in detail["reason"]


def test_a_cosmetic_defect_can_be_tolerated():
    verdict, _ = classify.decide_verdict(
        [make_defect("spur")], {"max_defects": 2}
    )
    assert verdict == "PASS"


def test_many_minor_defects_fail_on_accumulated_severity():
    defects = [make_defect("mouse_bite", index) for index in range(5)]
    verdict, detail = classify.decide_verdict(
        defects, {"max_defects": 10, "max_severity": 3}
    )
    assert verdict == "FAIL"
    assert detail["total_severity"] == 5


# ---------------------------------------------------------------------------
# Task 3.7 — evaluation harness
# ---------------------------------------------------------------------------
def test_iou_of_identical_boxes_is_one_and_disjoint_boxes_zero():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (50, 50, 10, 10)) == 0.0


def test_matching_is_greedy_and_claims_each_truth_once():
    outcome = match_boxes(
        [(0, 0, 10, 10), (1, 1, 10, 10)], [(0, 0, 10, 10)], iou_threshold=0.5
    )
    assert len(outcome.matches) == 1
    assert len(outcome.unmatched_predictions) == 1
    assert outcome.unmatched_truths == []


def test_localisation_and_classification_are_scored_separately():
    """A found-but-misnamed defect is a localisation hit and a classification miss."""
    evaluation = Evaluation(iou_threshold=0.5)
    evaluation.add_board(
        "board_a",
        predictions=[((0, 0, 10, 10), "short")],
        truths=[((0, 0, 10, 10), "spur")],
        runtime_s=0.2,
    )

    assert evaluation.localisation.f1 == pytest.approx(1.0)
    assert evaluation.classification.f1 == 0.0
    assert evaluation.conditional_class_accuracy() == 0.0
    assert evaluation.confusion["spur"]["short"] == 1


def test_false_positives_and_misses_land_in_the_background_row_and_column():
    evaluation = Evaluation(iou_threshold=0.5)
    evaluation.add_board(
        "board_b",
        predictions=[((100, 100, 10, 10), "spur")],
        truths=[((0, 0, 10, 10), "pin_hole")],
    )
    assert evaluation.confusion[BACKGROUND]["spur"] == 1
    assert evaluation.confusion["pin_hole"][BACKGROUND] == 1


def test_per_class_table_covers_all_six_classes_and_agrees_with_the_matrix():
    evaluation = Evaluation(iou_threshold=0.5)
    evaluation.add_board(
        "board_c",
        predictions=[((0, 0, 10, 10), "short"), ((40, 40, 10, 10), "spur")],
        truths=[((0, 0, 10, 10), "short"), ((40, 40, 10, 10), "spur")],
        runtime_s=0.1,
    )

    rows = {row["defect_class"]: row for row in evaluation.per_class()}
    assert len(rows) == 6
    assert rows["short"]["f1"] == 1.0
    assert rows["spur"]["support"] == 1
    assert rows["pin_hole"]["support"] == 0
    assert evaluation.summary()["class_accuracy"] == 1.0


def test_csv_export_writes_all_four_tables(tmp_path):
    evaluation = Evaluation()
    evaluation.add_board("board_d", [((0, 0, 10, 10), "spur")],
                         [((0, 0, 10, 10), "spur")], runtime_s=0.3)
    written = evaluation.write_csv(tmp_path, prefix="test")

    assert len(written) == 4
    for path in written:
        assert path.exists() and path.stat().st_size > 0
