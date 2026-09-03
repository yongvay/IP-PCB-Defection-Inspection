"""Module 1 unit tests — task 1.9.

Owner: Chan Xing Szen.

These tests are not here to raise a coverage number. Each one locks in a
decision that was made from a measurement, so that a later change which quietly
reverses it fails loudly instead of silently degrading Chapter 4's results.

Run:  python -m pytest tests -v
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.contracts import PreprocessResult
from src.module1 import calibration, ingest, morphology, preprocess, rectify

REPO_ROOT = Path(__file__).resolve().parents[1]
DEEPPCB = REPO_ROOT / "data" / "DeepPCB" / "PCBData"
HRIPCB = REPO_ROOT / "data" / "HRIPCB" / "PCB_DATASET"
SAMPLE_TEMPLATE = DEEPPCB / "group00041" / "00041" / "00041000_temp.jpg"
SAMPLE_TEST = DEEPPCB / "group00041" / "00041" / "00041000_test.jpg"

needs_deeppcb = pytest.mark.skipif(
    not SAMPLE_TEMPLATE.exists(),
    reason="DeepPCB is not committed; download it into data/ (see README)")
needs_hripcb = pytest.mark.skipif(
    not (HRIPCB / "PCB_USED").exists(),
    reason="HRIPCB is not committed; download it into data/ (see README)")


# --- Ingestion (task 1.3) ------------------------------------------------
def test_loader_raises_rather_than_returning_none():
    """cv2.imread returns None on a bad path; a None must not reach a filter."""
    with pytest.raises(FileNotFoundError):
        ingest.load_grey("no/such/board.jpg")


@needs_deeppcb
def test_deeppcb_splits_are_the_published_sizes():
    """1,000 trainval and 500 test. A silent change here invalidates every figure."""
    assert len(ingest.index_deeppcb(split="trainval")) == 1000
    assert len(ingest.index_deeppcb(split="test")) == 500


@needs_deeppcb
def test_trainval_and_test_do_not_overlap():
    """Tuning and reporting on the same boards would overstate every result."""
    trainval = {pair.name for pair in ingest.index_deeppcb(split="trainval")}
    test = {pair.name for pair in ingest.index_deeppcb(split="test")}
    assert not (trainval & test)


@needs_hripcb
def test_hripcb_pairs_every_image_with_its_own_reference_board():
    """The prefix before the first underscore names the board. It is the whole rule."""
    pairs = ingest.index_hripcb()
    assert len(pairs) == 693
    for pair in pairs[:50]:
        assert pair.template_path.stem.lstrip("0") == pair.name.split("_")[0].lstrip("0")


@needs_hripcb
def test_hripcb_rotation_set_carries_a_known_angle():
    """Without the recorded angle the rectification result is unmeasurable."""
    pairs = ingest.index_hripcb(rotated=True)
    assert all(pair.angle_deg is not None for pair in pairs)


@needs_hripcb
def test_voc_parser_matches_the_deeppcb_parser_shape():
    """Both datasets must produce identical dictionaries or every consumer forks."""
    pair = next(p for p in ingest.index_hripcb() if p.annotation_path)
    boxes = ingest.parse_voc_annotation(pair.annotation_path)
    assert boxes
    for box in boxes:
        assert set(box) == {"file", "bbox", "label", "polarity"}
        _, _, width, height = box["bbox"]
        assert width > 0 and height > 0
        assert box["polarity"] in ("added", "removed")


# --- Profile detection ---------------------------------------------------
def test_a_binary_image_is_detected_as_already_binarised():
    image = np.where(np.random.default_rng(0).random((64, 64)) > 0.5, 255, 0).astype(np.uint8)
    assert preprocess.detect_profile(image) == "prebinarised"


def test_a_photographic_image_is_detected_as_photographic():
    image = np.random.default_rng(0).integers(60, 190, (64, 64), dtype=np.uint8)
    assert preprocess.detect_profile(image) == "photographic"


@needs_deeppcb
def test_deeppcb_has_effectively_no_midtones():
    """The measurement that moved the enhancement study to HRIPCB.

    If this ever fails, DeepPCB has been replaced by an unbinarised version and
    the scoping argument in Chapter 4 no longer holds.
    """
    image = ingest.load_grey(SAMPLE_TEMPLATE)
    assert preprocess.midtone_fraction(image) < 0.001


@needs_hripcb
def test_hripcb_is_genuinely_photographic():
    pair = ingest.index_hripcb()[0]
    assert preprocess.midtone_fraction(ingest.load_grey(pair.test_path)) > 0.10


# --- The three algorithm banks (tasks 1.4 to 1.6) ------------------------
@pytest.mark.parametrize("bank,stage", [(preprocess.DENOISERS, "denoise"),
                                        (preprocess.ENHANCERS, "enhance"),
                                        (preprocess.BINARISERS, "binarise")])
def test_every_named_method_runs_and_preserves_shape_and_type(bank, stage):
    image = np.random.default_rng(1).integers(0, 255, (128, 128), dtype=np.uint8)
    for name, method in bank.items():
        result = method(image, {})
        assert result.shape == image.shape, f"{stage}/{name} changed the shape"
        assert result.dtype == np.uint8, f"{stage}/{name} changed the dtype"


def test_an_unknown_method_names_the_alternatives():
    """A typo in a sweep must fail immediately, not fall back to a default."""
    with pytest.raises(KeyError, match="Available"):
        preprocess.binarise(np.zeros((8, 8), np.uint8), {"binarise": "otsu2"})


def test_every_binariser_returns_only_two_values():
    """The contract requires 0 or 255; anything else fails at the boundary."""
    image = np.random.default_rng(2).integers(0, 255, (128, 128), dtype=np.uint8)
    for name, method in preprocess.BINARISERS.items():
        assert set(np.unique(method(image, {}))) <= {0, 255}, name


def test_median_beats_gaussian_on_salt_and_pepper():
    """The headline finding of task 1.4, reduced to its smallest reproducible case.

    A median is an order statistic, so a minority of extreme values cannot move
    it; an average is moved by every one of them.
    """
    clean = np.full((128, 128), 120, dtype=np.uint8)
    clean[40:90, 40:90] = 30
    generator = np.random.default_rng(3)
    noisy = clean.copy()
    draw = generator.random(clean.shape)
    noisy[draw < 0.02] = 0
    noisy[draw > 0.98] = 255

    error = lambda image: float(np.mean(np.abs(image.astype(int) - clean.astype(int))))
    assert error(preprocess.DENOISERS["median"](noisy, {})) < \
           error(preprocess.DENOISERS["gaussian"](noisy, {}))


def test_kernel_sizes_are_forced_odd():
    """A sweep that steps by one must not crash on OpenCV's odd-size requirement."""
    image = np.random.default_rng(4).integers(0, 255, (64, 64), dtype=np.uint8)
    assert preprocess.DENOISERS["median"](image, {"median_kernel": 4}).shape == image.shape


# --- Calibration (task 1.7) ----------------------------------------------
def test_supplied_calibration_wins_and_records_its_provenance():
    params = {"mm_per_px": 0.01}
    assert calibration.mm_per_px(np.zeros((8, 8), np.uint8), params) == 0.01
    assert params["calibration_source"] == "supplied"


def test_the_published_figure_applies_only_to_the_dataset_that_published_it():
    """DeepPCB's 48 px/mm is used for pre-binarised input, its own signature."""
    params = {"profile": "prebinarised"}
    value = calibration.mm_per_px(np.zeros((8, 8), np.uint8), params)
    assert value == pytest.approx(1.0 / calibration.DEEPPCB_PIXELS_PER_MM)
    assert params["calibration_source"] == "documented"
    assert params["area_unit"] == "mm2"


def test_an_unknown_scale_is_admitted_rather_than_borrowed():
    """The bug this prevents: HRIPCB silently inheriting DeepPCB's 48 px/mm.

    The two datasets were photographed at different resolutions, so borrowing
    the figure made every HRIPCB area wrong by roughly five times while still
    printing a plausible-looking number of square millimetres.
    """
    params = {"profile": "photographic"}
    value = calibration.mm_per_px(np.zeros((8, 8), np.uint8), params)
    assert value == calibration.UNCALIBRATED
    assert params["calibration_source"] == "uncalibrated"
    assert params["area_unit"] == "px2"


def test_a_supplied_scale_calibrates_a_photographic_board():
    """The escape hatch: once someone measures HRIPCB, one parameter fixes it."""
    params = {"profile": "photographic", "pixels_per_mm": 20.0}
    assert calibration.mm_per_px(np.zeros((8, 8), np.uint8), params) == pytest.approx(0.05)
    assert params["area_unit"] == "mm2"


def test_area_conversion_squares_the_factor():
    """Applying the factor once is wrong by ~50x on DeepPCB and still looks plausible."""
    assert calibration.area_mm2(100, 0.02) == pytest.approx(100 * 0.0004)


def test_hole_measurement_recovers_a_known_scale():
    """Draw circles of a known radius; the estimator must recover the scale.

    A synthetic board is used rather than a real one because only here is the
    true answer known exactly, which is the whole point of a unit test.
    """
    board = np.full((400, 400), 240, dtype=np.uint8)
    for x in range(60, 400, 90):
        for y in range(60, 400, 90):
            cv2.circle(board, (x, y), 12, 20, -1)

    estimate = calibration.estimate_pixels_per_mm(board, {"nominal_hole_diameter_mm": 1.0})
    assert estimate is not None
    assert estimate == pytest.approx(24.0, abs=4.0)


# --- Rectification (task 1.8) --------------------------------------------
def test_angles_fold_into_a_quarter_turn():
    """A board at 88 degrees and one at -2 are the same placement."""
    assert rectify.normalise_angle(88.0) == pytest.approx(-2.0)
    assert rectify.normalise_angle(-88.0) == pytest.approx(2.0)
    assert rectify.normalise_angle(0.0) == pytest.approx(0.0)


def test_rotating_a_board_and_measuring_it_is_a_round_trip():
    """deskew and estimate_angle must agree on which way is positive.

    The property that matters is the round trip, not the sign in isolation:
    a board turned by a known angle must be measured at that same angle, so
    that feeding the measurement straight back into deskew undoes the turn.
    A convention mismatch here would double the rotation instead of removing
    it, while still producing a confident-looking number.
    """
    board = np.full((600, 600), 200, dtype=np.uint8)
    board[150:450, 100:500] = 60

    for applied in (-7.0, 4.0):
        rotated = rectify.deskew(board, applied, border_value=200)
        estimate = rectify.estimate_angle(rotated)
        assert estimate is not None
        assert estimate == pytest.approx(applied, abs=1.0)

        # Levelling must return the board to square, not turn it further.
        restored, _ = rectify.rectify(rotated, {"rectify_border_value": 200})
        assert rectify.estimate_angle(restored) == pytest.approx(0.0, abs=1.0)


def test_a_square_board_is_left_untouched():
    """Every warp resamples, and resampling adds the jitter Module 1 exists to remove."""
    board = np.full((400, 400), 200, dtype=np.uint8)
    board[100:300, 80:320] = 60
    rectified, angle = rectify.rectify(board)
    assert np.array_equal(rectified, board)


def test_the_threshold_fallback_is_reported_rather_than_disguised():
    """A frame with no background yields an angle of zero that means 'not found'.

    Reported as a plain number it is indistinguishable from a correct estimate,
    which is how six rotated boards were scored as accurate detections.
    """
    uniform = np.random.default_rng(5).integers(0, 255, (200, 200), dtype=np.uint8)
    _, method = rectify.estimate_angle_detailed(uniform)
    assert method == "threshold"


# --- Morphology (task 1.10) ----------------------------------------------
def test_opening_deletes_a_thin_ribbon_and_keeps_a_compact_blob():
    """The geometric reason opening is the highest-leverage parameter."""
    image = np.zeros((200, 200), dtype=np.uint8)
    image[50, 20:180] = 255                 # one-pixel jitter along a trace edge
    cv2.circle(image, (120, 140), 9, 255, -1)   # a genuine compact defect

    cleaned = morphology.clean_difference(image, {"morph_open_kernel": 5,
                                                  "morph_close_kernel": 5})
    assert cleaned[45:56, 20:180].sum() == 0, "the ribbon should be removed"
    assert cleaned[120:160, 100:140].sum() > 0, "the compact blob should survive"


def test_cleanup_preserves_shape_and_does_not_mutate_its_input():
    """The contract with the orchestrator: same shape out, no side effects."""
    image = np.zeros((64, 64), dtype=np.uint8)
    image[20:40, 20:40] = 255
    before = image.copy()
    result = morphology.clean_difference(image, {})
    assert result.shape == image.shape and result.dtype == np.uint8
    assert np.array_equal(image, before)


# --- The contract --------------------------------------------------------
@needs_deeppcb
def test_preprocess_pair_satisfies_the_contract():
    result = preprocess.preprocess_pair(SAMPLE_TEMPLATE, SAMPLE_TEST)
    result.validate()
    assert isinstance(result, PreprocessResult)
    assert result.params["profile"] == "prebinarised"
    assert result.params["calibration_source"] in ("documented", "measured_holes", "supplied")
    assert result.params["area_unit"] == "mm2"


@needs_hripcb
def test_hripcb_areas_are_reported_in_pixels_until_someone_measures_the_scale():
    pair = ingest.index_hripcb()[0]
    result = preprocess.preprocess_pair(pair.template_path, pair.test_path)
    assert result.params["area_unit"] == "px2"
    assert "uncalibrated" in calibration.describe_scale(result.params)


@needs_deeppcb
def test_the_pair_is_processed_with_identical_settings():
    """If each image chose its own threshold, an exposure difference would read
    as a defect and the entire golden-template method would be unsound."""
    result = preprocess.preprocess_pair(SAMPLE_TEMPLATE, SAMPLE_TEST,
                                        {"binarise": "adaptive_mean"})
    assert result.params["binarise"] == "adaptive_mean"


@needs_deeppcb
def test_preprocessing_is_deterministic():
    """Two runs must agree exactly, or no reported figure can be reproduced."""
    first = preprocess.preprocess_pair(SAMPLE_TEMPLATE, SAMPLE_TEST)
    second = preprocess.preprocess_pair(SAMPLE_TEMPLATE, SAMPLE_TEST)
    assert np.array_equal(first.template_bin, second.template_bin)
    assert np.array_equal(first.test_bin, second.test_bin)
    assert first.mm_per_px == second.mm_per_px


@needs_hripcb
def test_a_mismatched_pair_is_resized_to_one_shape():
    """HRIPCB boards are photographed individually; the contract requires one shape."""
    pair = ingest.index_hripcb(rotated=True)[0]
    result = preprocess.preprocess_pair(pair.template_path, pair.test_path)
    result.validate()
    assert result.template_bin.shape == result.test_bin.shape


@needs_hripcb
def test_rectification_levels_a_rotated_board_end_to_end():
    """The order of operations, pinned. Resizing before deskewing distorts the
    board's aspect ratio and makes rectification worse than doing nothing."""
    pair = ingest.index_hripcb(rotated=True)[0]
    test = ingest.load_grey(pair.test_path)
    template = ingest.load_grey(pair.template_path)

    levelled, angle = rectify.rectify_to_template(test, template.shape)
    assert levelled.shape == template.shape
    assert angle == pytest.approx(pair.angle_deg, abs=1.5)


@needs_hripcb
def test_rectification_levels_a_rotated_board_end_to_end():
    """The order of operations, pinned.

    Resizing to the template before deskewing distorts the board's aspect ratio,
    and the rotation then aligns a board that is no longer the right shape. Doing
    it in that order measured worse than not rectifying at all, so the order is
    a decision worth a test rather than a comment.
    """
    pair = ingest.index_hripcb(rotated=True)[0]
    test = ingest.load_grey(pair.test_path)
    template = ingest.load_grey(pair.template_path)

    levelled, angle = rectify.rectify_to_template(test, template.shape)
    assert levelled.shape == template.shape
    assert angle == pytest.approx(pair.angle_deg, abs=1.5)
