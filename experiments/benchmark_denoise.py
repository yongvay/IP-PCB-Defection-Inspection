"""Noise-removal benchmark — Module 1, task 1.4. Evidence for SMART Objective 2.

Owner: Chan Xing Szen.

The experimental design, and why it is not simply "filter the dataset"
----------------------------------------------------------------------
Restoration quality cannot be measured without knowing the clean answer, and
neither dataset ships a noise-free counterpart to its images. Filtering the
images as they are and comparing the results to each other measures nothing:
whichever filter blurs most produces the smoothest picture, and smoothest is
not the same as most correct.

The standard resolution is to supply the missing ground truth. A clean image is
taken as the reference, a known quantity of noise of a known type is added to
it, each filter is asked to recover the original, and the recovery is scored
against the reference the noise was added to. The clean image is then, by
construction, the correct answer.

Two noise models are used because they break filters in different ways.
Gaussian noise perturbs every pixel slightly and models sensor and thermal
noise. Salt-and-pepper noise replaces a few pixels entirely with black or white
and models transmission dropouts and dead sensor elements. A filter that leads
on one can lose badly on the other, and reporting only one hides that.

Two metrics, because they disagree in a useful way. PSNR is a function of mean
squared error and rewards a filter for being close on average, which favours
blurring. SSIM compares local structure — luminance, contrast and correlation
in a sliding window — and penalises a filter for destroying an edge even when
its average error is small. On an image whose meaning lives entirely in its
edges, the two ranking differently is itself a finding.

Run:  python -m experiments.benchmark_denoise
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from experiments.common import centre_crop, rng, save_table
from src.module1 import ingest, preprocess

CROP = 768          # a square from the middle of each board, at native resolution
SAMPLE_SIZE = 12    # boards; every filter sees the identical set

GAUSSIAN_SIGMAS = (5, 10, 20)
SALT_PEPPER_RATES = (0.01, 0.03, 0.05)
FILTERS = ("none", "gaussian", "median", "bilateral")


def add_gaussian_noise(image: np.ndarray, sigma: float,
                       generator: np.random.Generator) -> np.ndarray:
    """Additive zero-mean Gaussian noise of a given standard deviation."""
    noisy = image.astype(np.float32) + generator.normal(0.0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image: np.ndarray, rate: float,
                          generator: np.random.Generator) -> np.ndarray:
    """Replace a fraction of pixels with black or white, half each."""
    noisy = image.copy()
    draw = generator.random(image.shape)
    noisy[draw < rate / 2.0] = 0
    noisy[draw > 1.0 - rate / 2.0] = 255
    return noisy


def score(reference: np.ndarray, restored: np.ndarray) -> tuple[float, float]:
    return (float(peak_signal_noise_ratio(reference, restored, data_range=255)),
            float(structural_similarity(reference, restored, data_range=255)))


def run() -> pd.DataFrame:
    generator = rng()
    boards = [centre_crop(ingest.load_grey(pair.test_path), CROP)
              for pair in ingest.index_hripcb()[:SAMPLE_SIZE]]

    conditions = ([("gaussian", sigma, add_gaussian_noise) for sigma in GAUSSIAN_SIGMAS]
                  + [("salt_pepper", rate, add_salt_pepper_noise) for rate in SALT_PEPPER_RATES])

    rows = []
    for noise_name, level, corrupt in conditions:
        for filter_name in FILTERS:
            psnrs, ssims, elapsed = [], [], 0.0
            for clean in boards:
                noisy = corrupt(clean, level, generator)
                started = time.perf_counter()
                restored = preprocess.DENOISERS[filter_name](noisy, {})
                elapsed += time.perf_counter() - started
                psnr_value, ssim_value = score(clean, restored)
                psnrs.append(psnr_value)
                ssims.append(ssim_value)

            rows.append({
                "noise": noise_name,
                "level": level,
                "filter": filter_name,
                "psnr_db": round(float(np.mean(psnrs)), 3),
                "ssim": round(float(np.mean(ssims)), 4),
                "ms_per_image": round(1000.0 * elapsed / len(boards), 2),
            })

    return pd.DataFrame(rows)


def run_deeppcb_control() -> pd.DataFrame:
    """Control: repeat the study on DeepPCB, where the images are already binarised.

    This is the experiment that justifies moving the preprocessing study to
    HRIPCB. A binarised image has no midtones, so contrast enhancement has
    nothing to act on and every thresholding method cuts in the same place. The
    control is included rather than merely asserted, because a reader is
    entitled to the evidence for a scoping decision.
    """
    generator = rng()
    boards = [centre_crop(ingest.load_grey(pair.template_path), CROP)
              for pair in ingest.index_deeppcb(split="test", limit=SAMPLE_SIZE)]

    rows = []
    for filter_name in FILTERS:
        psnrs, ssims = [], []
        for clean in boards:
            noisy = add_salt_pepper_noise(clean, 0.03, generator)
            restored = preprocess.DENOISERS[filter_name](noisy, {})
            psnr_value, ssim_value = score(clean, restored)
            psnrs.append(psnr_value)
            ssims.append(ssim_value)
        rows.append({
            "dataset": "deeppcb",
            "noise": "salt_pepper",
            "level": 0.03,
            "filter": filter_name,
            "psnr_db": round(float(np.mean(psnrs)), 3),
            "ssim": round(float(np.mean(ssims)), 4),
        })

    midtones = [preprocess.midtone_fraction(board) for board in boards]
    print(f"\nDeepPCB midtone fraction: mean {np.mean(midtones):.6f} "
          f"(threshold for 'already binarised' is {preprocess.PREBINARISED_MIDTONE_FRACTION})")
    return pd.DataFrame(rows)


def main() -> None:
    print(f"Denoising benchmark: {SAMPLE_SIZE} HRIPCB boards, {CROP}x{CROP} crops, "
          f"{len(FILTERS)} filters, 6 noise conditions.")
    save_table(run(), "denoise_benchmark.csv")
    save_table(run_deeppcb_control(), "denoise_control_deeppcb.csv")


if __name__ == "__main__":
    main()
