"""Preprocessing and calibration — Module 1, tasks 1.3 to 1.7.

Owner: Chan Xing Szen.

This file is the Module 1 side of the contract in src/contracts.py: given two
image paths it returns a PreprocessResult whose two images are binary, the same
shape, and safe for Module 2 to difference with a bitwise operation.

Three algorithm banks, one job each
-----------------------------------
Denoising, contrast enhancement and binarisation are each exposed as a
dictionary of named implementations rather than as a hard-coded call. Every
method the assignment requires is present and selectable by name, which is what
allows experiments/benchmark_preprocess.py to sweep all of them without a
single branch, and what keeps tuned constants out of the function bodies.

The measurement that shapes this file
-------------------------------------
The two datasets need genuinely different treatment, and the difference is not
a matter of taste. Measured over the pixel histograms:

    dataset    pixels at the extremes    pixels in the midtones (50-200)
    DeepPCB          98.4 - 99.96%                    0.0000%
    HRIPCB                    0.05%                      68.0%

DeepPCB was binarised by its authors before publication, so it has no midtones
at all. Histogram equalisation, CLAHE and contrast stretching have nothing to
operate on there, and Otsu, adaptive mean and adaptive Gaussian thresholding
all return the same image because any threshold between the two populations
cuts in the same place. Running the enhancement study on DeepPCB would produce
a table of identical numbers and an empty discussion.

HRIPCB is an ordinary photographic greyscale distribution, so it is where the
preprocessing comparison for SMART Objective 2 is actually carried out. That is
a finding, not an inconvenience, and Chapter 4 should report it as one.

Because the right treatment differs, the profile is detected from the image
rather than passed in by the caller, and recorded in ``params`` so that every
result carries the evidence of how it was produced.

Author: Chan Xing Szen (Member A, Module 1)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from src.contracts import PreprocessResult
from src.module1 import calibration, rectify as rectification
from src.module1.ingest import load_colour, load_grey, load_pair  # noqa: F401

# A pixel is a "midtone" if it is neither nearly black nor nearly white. An
# image with almost none has already been thresholded by somebody else.
MIDTONE_LOW, MIDTONE_HIGH = 50, 200
PREBINARISED_MIDTONE_FRACTION = 0.01


# ---------------------------------------------------------------------------
# Task 1.4 — noise removal
# ---------------------------------------------------------------------------
def _denoise_none(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return image


def _denoise_gaussian(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Weighted average over a neighbourhood: optimal against Gaussian sensor noise.

    Linear and separable, so it is the cheapest of the three, but it treats an
    edge as just another intensity change and blurs trace boundaries along with
    the noise.
    """
    k = _odd(params.get("gaussian_kernel", 5))
    return cv2.GaussianBlur(image, (k, k), params.get("gaussian_sigma", 0))


def _denoise_median(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Replace each pixel with the neighbourhood median.

    A median is unmoved by a minority of extreme values, which is exactly what
    salt-and-pepper speckle is, and being an order statistic it can only return
    a value that genuinely occurred nearby — so it does not invent the soft
    ramps that blur a trace edge.
    """
    return cv2.medianBlur(image, _odd(params.get("median_kernel", 3)))


def _denoise_bilateral(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Gaussian in space multiplied by a Gaussian in intensity.

    A neighbour only contributes if it is both near and similar, so smoothing
    stops at an edge instead of crossing it. Edge preservation is the reason to
    care on a PCB, where every defect is defined by a trace boundary; the cost
    is that it is by far the slowest of the three.
    """
    return cv2.bilateralFilter(
        image,
        params.get("bilateral_diameter", 9),
        params.get("bilateral_sigma_colour", 75),
        params.get("bilateral_sigma_space", 75),
    )


DENOISERS: dict[str, Callable[[np.ndarray, dict], np.ndarray]] = {
    "none": _denoise_none,
    "gaussian": _denoise_gaussian,
    "median": _denoise_median,
    "bilateral": _denoise_bilateral,
}


# ---------------------------------------------------------------------------
# Task 1.5 — contrast enhancement
# ---------------------------------------------------------------------------
def _enhance_none(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return image


def _enhance_hist_eq(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Global histogram equalisation: remap intensities by the cumulative histogram.

    Spreads the occupied range across the full scale and is parameter-free, but
    it is global — one transform for the whole image — so a board lit unevenly
    ends up over-stretched where it was already bright.
    """
    return cv2.equalizeHist(image)


def _enhance_clahe(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation.

    Equalises within small tiles so uneven illumination across a board is
    handled locally, and clips each tile's histogram before equalising so that a
    nearly uniform tile of bare substrate does not have its noise amplified into
    visible texture. The clip limit is what separates this from plain adaptive
    equalisation, and it is the parameter worth sweeping.
    """
    clahe = cv2.createCLAHE(
        clipLimit=params.get("clahe_clip", 2.0),
        tileGridSize=(params.get("clahe_tiles", 8), params.get("clahe_tiles", 8)),
    )
    return clahe.apply(image)


def _enhance_linear_stretch(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Linear contrast stretching between two intensity percentiles.

    Rescales the band between the low and high percentiles onto 0-255. Cutting
    at percentiles rather than at the true minimum and maximum means a handful
    of hot or dead pixels cannot decide the mapping for the whole image, which
    is the standard failure of naive min-max stretching.
    """
    low = np.percentile(image, params.get("stretch_low_pct", 1.0))
    high = np.percentile(image, params.get("stretch_high_pct", 99.0))
    if high <= low:
        return image
    stretched = (image.astype(np.float32) - low) * (255.0 / (high - low))
    return np.clip(stretched, 0, 255).astype(np.uint8)


ENHANCERS: dict[str, Callable[[np.ndarray, dict], np.ndarray]] = {
    "none": _enhance_none,
    "hist_eq": _enhance_hist_eq,
    "clahe": _enhance_clahe,
    "linear_stretch": _enhance_linear_stretch,
}


# ---------------------------------------------------------------------------
# Task 1.6 — binarisation
# ---------------------------------------------------------------------------
def _binarise_fixed(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Threshold at a constant. Only defensible on an already-binarised image."""
    _, binary = cv2.threshold(image, params.get("fixed_threshold", 127), 255,
                              cv2.THRESH_BINARY)
    return binary


def _binarise_otsu(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Otsu's method: the threshold minimising within-class intensity variance.

    Derived from the histogram alone with no parameter to choose, and optimal
    when the histogram is bimodal — which a bare board is, being copper against
    substrate. Its weakness is that it is global, so a shadow across one corner
    shifts the single threshold for the entire image.
    """
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _binarise_adaptive_mean(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Threshold each pixel against the mean of its own neighbourhood, minus C.

    Local by construction, so uneven illumination stops mattering. The block
    size sets the scale of feature it can resolve: too small and the interior of
    a wide trace is compared against itself and dissolves into texture.
    """
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        _odd(params.get("adaptive_block", 35)), params.get("adaptive_c", 5),
    )


def _binarise_adaptive_gaussian(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """As adaptive mean, but the neighbourhood is Gaussian-weighted.

    Weighting by distance makes the local threshold vary smoothly instead of
    jumping as the window slides across an edge, which shows up as cleaner trace
    boundaries and fewer one-pixel artefacts along them.
    """
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        _odd(params.get("adaptive_block", 35)), params.get("adaptive_c", 5),
    )


BINARISERS: dict[str, Callable[[np.ndarray, dict], np.ndarray]] = {
    "fixed": _binarise_fixed,
    "otsu": _binarise_otsu,
    "adaptive_mean": _binarise_adaptive_mean,
    "adaptive_gaussian": _binarise_adaptive_gaussian,
}


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------
# Defaults per profile. These are the only place a tuned constant is allowed to
# live; every caller overrides by passing params, and nothing downstream
# hard-codes a kernel size.
PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    # Already thresholded upstream: denoise lightly, do not touch contrast,
    # and cut at the obvious place.
    "prebinarised": {"denoise": "median", "median_kernel": 3,
                     "enhance": "none", "binarise": "otsu"},
    # A real photograph: all three stages do measurable work. Every value here
    # is the winner of a sweep scored in detection F1 on HRIPCB, not a default
    # copied from a tutorial. See experiments/benchmark_pipeline.py, and note
    # in particular that adaptive mean thresholding scores 0.62 here against
    # Otsu's 0.26 — the reverse of the DeepPCB result, because a photograph has
    # the uneven illumination that a single global threshold cannot follow.
    "photographic": {"denoise": "bilateral", "bilateral_diameter": 9,
                     "bilateral_sigma_colour": 75, "bilateral_sigma_space": 75,
                     "enhance": "clahe", "clahe_clip": 2.0, "clahe_tiles": 8,
                     "binarise": "adaptive_mean",
                     "adaptive_block": 35, "adaptive_c": 5},
}


def midtone_fraction(image: np.ndarray) -> float:
    """Fraction of pixels that are neither nearly black nor nearly white."""
    counts = np.bincount(image.ravel(), minlength=256)
    return float(counts[MIDTONE_LOW:MIDTONE_HIGH + 1].sum()) / image.size


def detect_profile(image: np.ndarray) -> str:
    """Decide whether an image arrived already binarised.

    Detecting this rather than requiring the caller to declare it means the same
    pipeline runs unmodified on both datasets, and means a future dataset is
    handled correctly without an added branch.
    """
    return ("prebinarised" if midtone_fraction(image) < PREBINARISED_MIDTONE_FRACTION
            else "photographic")


def resolve_params(image: np.ndarray, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge caller overrides onto the defaults for the detected profile."""
    params = params or {}
    profile = params.get("profile") or detect_profile(image)
    resolved = dict(PROFILE_DEFAULTS[profile])
    resolved.update(params)
    resolved["profile"] = profile
    return resolved


# ---------------------------------------------------------------------------
# The three stages, applied
# ---------------------------------------------------------------------------
def denoise(image: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
    """Apply the selected noise-removal filter (task 1.4)."""
    params = resolve_params(image, params)
    return _dispatch(DENOISERS, params.get("denoise", "median"), "denoise")(image, params)


def enhance(image: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
    """Apply the selected contrast enhancement (task 1.5)."""
    params = resolve_params(image, params)
    return _dispatch(ENHANCERS, params.get("enhance", "none"), "enhance")(image, params)


def binarise(image: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
    """Apply the selected thresholding method (task 1.6)."""
    params = resolve_params(image, params)
    method = _dispatch(BINARISERS, params.get("binarise", "otsu"), "binarise")
    return method(image, params).astype(np.uint8)


def calibrate(image: np.ndarray, params: dict[str, Any] | None = None) -> float:
    """Millimetres per pixel for this board (task 1.7). See calibration.py."""
    return calibration.mm_per_px(image, params)


def preprocess_image(image: np.ndarray,
                     params: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
    """Run denoise then enhance then binarise, returning the image and the settings used.

    The order is not arbitrary. Denoising first stops the enhancement stage from
    amplifying noise into structure; enhancing before thresholding gives the
    threshold a wider separation to find; thresholding last is what the contract
    requires. Reversing any pair measurably worsens the result.
    """
    resolved = resolve_params(image, params)
    stage = _dispatch(DENOISERS, resolved["denoise"], "denoise")(image, resolved)
    stage = _dispatch(ENHANCERS, resolved["enhance"], "enhance")(stage, resolved)
    stage = _dispatch(BINARISERS, resolved["binarise"], "binarise")(stage, resolved)
    return stage.astype(np.uint8), resolved


def preprocess_pair(template_path: str | Path,
                    test_path: str | Path,
                    params: dict[str, Any] | None = None) -> PreprocessResult:
    """Module 1's entry point, called by the orchestrator. Signature fixed by contract.

    The template decides the profile and the settings, and the test image is put
    through exactly the same ones. Letting each image choose its own threshold
    would mean any difference between them could be a difference in exposure
    rather than a defect, and the whole method rests on that not being possible.

    Rectification (task 1.8) runs before thresholding when ``params['rectify']``
    is set, and is off by default. Two reasons for the default. Every warp
    resamples, and resampling adds jitter along exactly the trace edges this
    module exists to keep clean, so correcting an angle that was never there
    costs accuracy for nothing — which is the case for all 1,500 DeepPCB pairs,
    already aligned by their authors. And rectifying in greyscale before
    thresholding, rather than afterwards, means the interpolation happens where
    intermediate values are meaningful instead of smearing a binary edge.
    """
    template, test = load_pair(template_path, test_path)
    resolved = resolve_params(template, params)

    if resolved.get("rectify", False):
        # Level, trim, then scale — in that order. See rectify_to_template.
        template = rectification.crop_to_board(
            rectification.rectify(template, resolved)[0], resolved)
        test, angle = rectification.rectify_to_template(test, template.shape, resolved)
        resolved["test_angle_deg"] = angle
    else:
        test = _match_shape(test, template.shape)

    template_bin, resolved = preprocess_image(template, resolved)
    test_bin, _ = preprocess_image(test, resolved)

    scale = calibrate(template, resolved)
    # Kept so that describe_scale can report the factor alongside its source.
    resolved["mm_per_px_used"] = scale

    result = PreprocessResult(
        template_bin=template_bin,
        test_bin=test_bin,
        mm_per_px=scale,
        params=resolved,
    )
    result.validate()
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _dispatch(bank: dict[str, Callable], name: str, stage: str) -> Callable:
    """Look up a named algorithm, naming the alternatives when it is not found."""
    try:
        return bank[name]
    except KeyError:
        raise KeyError(
            f"Unknown {stage} method '{name}'. Available: {', '.join(sorted(bank))}"
        ) from None


def _odd(value: int) -> int:
    """OpenCV kernel sizes must be odd; a sweep that steps by one should not crash."""
    value = int(value)
    return value if value % 2 == 1 else value + 1


def _match_shape(image: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Resize to the template's shape if the pair disagrees.

    DeepPCB pairs always agree. HRIPCB boards are photographed individually, so
    a test image can differ from its reference by a few pixels; the contract
    requires one shape, and INTER_AREA is used because it averages rather than
    samples, which is the correct choice when downscaling detailed line work.
    """
    if image.shape[:2] == tuple(shape[:2]):
        return image
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
