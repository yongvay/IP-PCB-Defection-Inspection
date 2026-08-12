"""Image registration and alignment quality measurement.

Tasks 2.1 and 2.2 of Module 2.

The golden-template method compares a test board against a defect-free
reference. That comparison is only meaningful if the two images occupy the
same coordinate frame, so every pixel of copper in the template lands on the
corresponding pixel of copper in the test image. Misalignment of even two or
three pixels produces a bright outline around every trace in the difference
image, which the blob extractor then reports as hundreds of false defects.
Registration is therefore the algorithm the whole system rests on.

Two strategies are provided:

  Feature-based (primary)
      Detect keypoints with ORB or SIFT, match descriptors, then fit a
      homography with RANSAC so that mismatched pairs are rejected as
      outliers. This handles rotation, translation, scale and mild
      perspective, which is what the misoriented HRIPCB boards require.

  Phase correlation (fallback)
      Bare PCB images are highly repetitive, so descriptor matching can fail
      when too few distinctive keypoints survive. Phase correlation works in
      the frequency domain on the whole image at once and recovers pure
      translation robustly. It cannot recover rotation, which is exactly why
      it is the fallback rather than the primary method.

Author: Ng Yong Vay (Member B in the plan, Module 2)
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Above this mean reprojection error the pair is treated as badly registered.
# Chosen because DeepPCB is ~48 pixels per millimetre, so 3 px is roughly
# 0.06 mm: tighter than the smallest defect the system is expected to find,
# and loose enough not to reject sound alignments over sub-pixel noise.
DEFAULT_RESIDUAL_THRESHOLD_PX = 3.0

# RANSAC needs at least four point correspondences to solve for a homography.
MIN_MATCHES_FOR_HOMOGRAPHY = 4

# Lowe's ratio test threshold. A match is kept only if the best descriptor
# distance is clearly better than the second best, which removes the ambiguous
# matches that repetitive copper patterns generate in large numbers.
LOWE_RATIO = 0.75


@dataclass
class RegistrationResult:
    """Everything the caller needs to know about one alignment attempt."""

    aligned: np.ndarray               # test image warped into template frame
    homography: np.ndarray            # 3x3 transform applied
    residual: float                   # mean reprojection error, pixels
    inliers: int                      # correspondences RANSAC kept
    total_matches: int                # correspondences offered to RANSAC
    method: str                       # "orb", "sift" or "phase_correlation"
    ok: bool                          # residual within threshold

    @property
    def inlier_ratio(self) -> float:
        """Proportion of matches RANSAC accepted. Low values mean a weak fit."""
        if self.total_matches == 0:
            return 0.0
        return self.inliers / self.total_matches


def _to_greyscale(image: np.ndarray) -> np.ndarray:
    """Reduce to a single channel, because all detectors below expect one."""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _create_detector(method: str):
    """Return the requested keypoint detector.

    ORB is the default: it is free of patent restrictions, considerably faster
    than SIFT, and binary descriptors suit the high-contrast binarised copper
    patterns of a bare board. SIFT is offered because it tolerates scale
    change better, which matters for the HRIPCB images captured at varying
    working distances.
    """
    if method == "orb":
        # A high feature budget is deliberate. Bare boards are visually
        # repetitive, so a large fraction of keypoints will be discarded by
        # the ratio test and RANSAC; starting with more survives that loss.
        return cv2.ORB_create(nfeatures=5000)
    if method == "sift":
        return cv2.SIFT_create(nfeatures=5000)
    raise ValueError(f"Unknown detector '{method}'. Use 'orb' or 'sift'.")


def _match_descriptors(desc_test, desc_template, method: str) -> list:
    """Match descriptors and filter them with Lowe's ratio test.

    ORB produces binary descriptors compared with the Hamming distance;
    SIFT produces float descriptors compared with the L2 norm. Using the wrong
    norm silently produces meaningless matches, so the norm is selected from
    the detector rather than hard-coded.
    """
    norm = cv2.NORM_HAMMING if method == "orb" else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm, crossCheck=False)

    # k=2 so the ratio test has a second-best distance to compare against.
    raw_matches = matcher.knnMatch(desc_test, desc_template, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        best, second_best = pair
        if best.distance < LOWE_RATIO * second_best.distance:
            good.append(best)
    return good


def _reprojection_residual(src_points: np.ndarray,
                           dst_points: np.ndarray,
                           homography: np.ndarray,
                           mask: np.ndarray) -> tuple[float, int]:
    """Measure how well the fitted homography actually maps the inliers.

    This is Task 2.2. The homography is applied to the source keypoints and
    the Euclidean distance to where they should have landed is averaged over
    the RANSAC inliers only. Outliers are excluded deliberately: they were
    already judged to be wrong correspondences, so including them would
    measure the quality of the matcher rather than the quality of the fit.
    """
    inlier_mask = mask.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())
    if inlier_count == 0:
        return float("inf"), 0

    src_inliers = src_points[inlier_mask]
    dst_inliers = dst_points[inlier_mask]

    projected = cv2.perspectiveTransform(src_inliers.reshape(-1, 1, 2), homography)
    errors = np.linalg.norm(projected.reshape(-1, 2) - dst_inliers, axis=1)
    return float(errors.mean()), inlier_count


def register_features(test: np.ndarray,
                      template: np.ndarray,
                      method: str = "orb",
                      residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD_PX
                      ) -> RegistrationResult | None:
    """Align the test image to the template using keypoints and RANSAC.

    Returns None when the image pair does not yield enough reliable
    correspondences to fit a homography, which is the caller's signal to fall
    back to phase correlation. Returning None rather than raising keeps the
    fallback path ordinary control flow instead of exception handling.
    """
    test_grey = _to_greyscale(test)
    template_grey = _to_greyscale(template)

    detector = _create_detector(method)
    keypoints_test, desc_test = detector.detectAndCompute(test_grey, None)
    keypoints_template, desc_template = detector.detectAndCompute(template_grey, None)

    if desc_test is None or desc_template is None:
        return None
    if len(keypoints_test) < MIN_MATCHES_FOR_HOMOGRAPHY:
        return None
    if len(keypoints_template) < MIN_MATCHES_FOR_HOMOGRAPHY:
        return None

    matches = _match_descriptors(desc_test, desc_template, method)
    if len(matches) < MIN_MATCHES_FOR_HOMOGRAPHY:
        return None

    src_points = np.float32([keypoints_test[m.queryIdx].pt for m in matches])
    dst_points = np.float32([keypoints_template[m.trainIdx].pt for m in matches])

    # RANSAC repeatedly fits a homography to a random minimal subset and keeps
    # the model that the most correspondences agree with. This is what makes
    # the method survive the large number of wrong matches that repetitive
    # copper traces inevitably produce.
    homography, mask = cv2.findHomography(
        src_points.reshape(-1, 1, 2),
        dst_points.reshape(-1, 1, 2),
        cv2.RANSAC,
        ransacReprojThreshold=5.0,
    )
    if homography is None:
        return None

    residual, inliers = _reprojection_residual(src_points, dst_points, homography, mask)

    height, width = template_grey.shape[:2]
    # INTER_NEAREST preserves the strict two-value nature of a binarised image.
    # Interpolating would create intermediate grey values along every trace
    # edge, which the differencing stage would then have to re-threshold.
    aligned = cv2.warpPerspective(
        test, homography, (width, height),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    return RegistrationResult(
        aligned=aligned,
        homography=homography,
        residual=residual,
        inliers=inliers,
        total_matches=len(matches),
        method=method,
        ok=residual <= residual_threshold,
    )


def register_phase_correlation(test: np.ndarray,
                               template: np.ndarray,
                               residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD_PX
                               ) -> RegistrationResult:
    """Recover pure translation in the frequency domain.

    Two images that differ only by a shift have Fourier transforms that differ
    only by a phase ramp, so the inverse transform of the normalised
    cross-power spectrum is a sharp peak whose position is the shift. It uses
    every pixel rather than a sparse set of keypoints, which is why it stays
    reliable on repetitive boards where descriptor matching collapses.

    The peak response is used as a confidence value. Because there is no
    keypoint set to reproject, the residual is derived from the sub-pixel
    fractional part of the recovered shift: a clean translation lands close to
    a whole pixel, whereas a poor match produces an arbitrary fraction.
    """
    test_grey = _to_greyscale(test).astype(np.float32)
    template_grey = _to_greyscale(template).astype(np.float32)

    # A Hann window suppresses the spurious edge response caused by the FFT
    # treating the image as if it tiled the plane periodically.
    window = cv2.createHanningWindow(test_grey.shape[::-1], cv2.CV_32F)
    (shift_x, shift_y), response = cv2.phaseCorrelate(
        test_grey * window, template_grey * window
    )

    translation = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    height, width = template_grey.shape[:2]
    aligned = cv2.warpAffine(
        test, translation, (width, height),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    homography = np.float32([[1, 0, shift_x], [0, 1, shift_y], [0, 0, 1]])
    fractional = np.hypot(shift_x - round(shift_x), shift_y - round(shift_y))

    return RegistrationResult(
        aligned=aligned,
        homography=homography,
        residual=float(fractional),
        inliers=0,
        total_matches=0,
        method="phase_correlation",
        # Phase correlation is only trusted when the peak is well defined.
        # A low response means the transform found no dominant shift at all.
        ok=response > 0.05 and fractional <= residual_threshold,
    )


def _identity_result(test: np.ndarray,
                     template: np.ndarray) -> RegistrationResult:
    """The 'do nothing' alignment, used when the pair is already registered."""
    return RegistrationResult(
        aligned=test,
        homography=np.eye(3, dtype=np.float32),
        residual=0.0,
        inliers=0,
        total_matches=0,
        method="identity",
        ok=True,
    )


def disagreement(test: np.ndarray, template: np.ndarray) -> float:
    """Fraction of pixels where the two binary images differ.

    This is the objective function registration exists to minimise. Keypoint
    reprojection residual measures how well the homography fits the keypoints
    it was fitted to, which is not the same thing and can look excellent while
    the images themselves have been pulled apart.
    """
    test_grey = _to_greyscale(test)
    template_grey = _to_greyscale(template)
    if test_grey.shape != template_grey.shape:
        return 1.0
    return float(np.count_nonzero(cv2.bitwise_xor(test_grey, template_grey))
                 / test_grey.size)


def register(test: np.ndarray,
             template: np.ndarray,
             method: str = "orb",
             residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD_PX
             ) -> RegistrationResult:
    """Align the test image to the template, but only if that helps.

    This is the only function the orchestrator calls.

    Registration has to earn its place. DeepPCB pairs are already aligned by
    the dataset authors, and on such a pair a feature-based fit still returns
    a homography roughly 1.2 pixels away from the identity, because keypoint
    positions on a repetitive binarised board are noisy. Applying that warp
    displaces every trace by a pixel or so, and the difference image gains a
    thin outline around every conductor. Measured over 88 pairs, blindly
    trusting the fitted homography cost 0.12 of F1:

        applied unconditionally    precision 0.64   recall 0.70   F1 0.67
        applied only when it helps precision 0.87   recall 0.73   F1 0.79

    So each candidate transform is scored by the quantity that actually
    matters, the fraction of pixels that still disagree after warping, and the
    identity transform competes on the same terms. A small margin is required
    before a warp is accepted, so that a transform which is merely equivalent
    to doing nothing loses to doing nothing.

    The result is a system that leaves aligned pairs alone and still corrects
    the misoriented HRIPCB boards, without the caller needing to know which
    kind of pair it holds.
    """
    candidates: list[RegistrationResult] = [_identity_result(test, template)]

    feature_result = register_features(test, template, method, residual_threshold)
    if feature_result is not None:
        candidates.append(feature_result)

    # Phase correlation is only worth attempting when the feature path failed
    # outright or produced an alignment that does not meet the residual check.
    if feature_result is None or not feature_result.ok:
        candidates.append(
            register_phase_correlation(test, template, residual_threshold)
        )

    baseline = disagreement(test, template)
    best, best_score = candidates[0], baseline

    for candidate in candidates[1:]:
        score = disagreement(candidate.aligned, template)
        # Require a clear improvement, not a marginal one. Ten per cent of the
        # baseline disagreement is comfortably outside the noise of a
        # near-identity warp and comfortably inside the gain from correcting a
        # genuinely misoriented board.
        if score < best_score * 0.9:
            best, best_score = candidate, score

    # Report honestly on the pair even when the identity transform won: a pair
    # that still disagrees over a large fraction of its pixels after the best
    # available alignment is one whose defects should not be trusted.
    if best.method == "identity":
        best.residual = baseline
        best.ok = baseline < 0.05

    return best
