"""Visual inspection of the Module 1 pipeline — companion to the benchmarks.

Owner: Chan Xing Szen.

The benchmarks answer "which setting scores best". They cannot answer "what is
this filter actually doing to my board", and that second question is the one
worth being able to answer out loud: preprocessing is judged individually and
live, and a table of F1 scores is a poor thing to point at when asked why
adaptive thresholding beats Otsu on a photograph.

Every panel here is produced by the same functions the pipeline calls, not by a
separate drawing path, so what is on screen is what the system does.

Four figures:

  1. stages      the board through denoise -> enhance -> binarise, in order
  2. gallery     every method in each bank, side by side on the same crop
  3. detection   template, test, raw difference, cleaned difference, and the
                 detections drawn against the ground truth
  4. rectify     a rotated board, the segmentation that finds it, and the result

Run:  python -m experiments.visualise_stages [deeppcb|hripcb]
"""

from __future__ import annotations

import sys

import cv2
import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from experiments.common import FIGURE_DIR
from src.module1 import ingest, morphology, preprocess, rectify
from src.module1.ground_truth import parse_annotation_file
from src.module2 import blobs as blob_extraction
from src.module2 import difference, registration
from src.pipeline import inspect_pair

INK, INK_SOFT, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
TRUTH_COLOUR = "#1baf7a"        # ground truth
FOUND_COLOUR = "#eb6834"        # what the pipeline detected

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.size": 8.5, "text.color": INK, "figure.dpi": 190,
    "savefig.bbox": "tight",
})

# A crop keeps fine detail legible. A whole 3034-pixel HRIPCB board printed
# across three inches of a report shows nothing at all.
CROP = {"deeppcb": 320, "hripcb": 700}


def show(axis, image: np.ndarray, title: str, subtitle: str = "") -> None:
    axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axis.set_title(title, fontsize=9, fontweight="bold", color=INK, pad=4)
    if subtitle:
        axis.set_xlabel(subtitle, fontsize=7.5, color=INK_SOFT, labelpad=3)
    axis.set_xticks([]); axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_edgecolor("#dcdbd6")


def save(figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / name)
    plt.close(figure)
    print(f"  outputs/figures/{name}")


def crop(image: np.ndarray, size: int, centre=None) -> np.ndarray:
    """A square window, centred on a point of interest when one is given."""
    height, width = image.shape[:2]
    if centre is None:
        centre = (width // 2, height // 2)
    left = int(np.clip(centre[0] - size // 2, 0, max(0, width - size)))
    top = int(np.clip(centre[1] - size // 2, 0, max(0, height - size)))
    return image[top:top + size, left:left + size]


def _first_defect_centre(pair: ingest.Pair):
    """Centre the crops on a real defect rather than on the middle of the board."""
    if pair.annotation_path is None:
        return None
    boxes = (parse_annotation_file(pair.annotation_path) if pair.dataset == "deeppcb"
             else ingest.parse_voc_annotation(pair.annotation_path))
    if not boxes:
        return None
    x, y, w, h = boxes[0]["bbox"]
    return (x + w // 2, y + h // 2)


# ---------------------------------------------------------------------------
def figure_stages(pair: ingest.Pair, dataset: str) -> None:
    """The board through each stage in the order the pipeline applies them."""
    grey = ingest.load_grey(pair.test_path)
    params = preprocess.resolve_params(grey)
    centre = _first_defect_centre(pair)
    size = CROP[dataset]

    denoised = preprocess.DENOISERS[params["denoise"]](grey, params)
    enhanced = preprocess.ENHANCERS[params["enhance"]](denoised, params)
    binary = preprocess.BINARISERS[params["binarise"]](enhanced, params)

    panels = [
        (grey, "1. greyscale", f"as loaded · {grey.shape[1]}x{grey.shape[0]}"),
        (denoised, "2. denoised", f"{params['denoise']}"),
        (enhanced, "3. enhanced", f"{params['enhance']}"),
        (binary, "4. binarised", f"{params['binarise']}"),
    ]

    figure, axes = plt.subplots(1, 4, figsize=(12.5, 3.6))
    for axis, (image, title, subtitle) in zip(axes, panels):
        show(axis, crop(image, size, centre), title, subtitle)

    figure.suptitle(
        f"Module 1 preprocessing stages — {dataset.upper()} "
        f"({params['profile']} profile, {size}x{size} crop on a defect)",
        fontsize=10.5, fontweight="bold", x=0.005, ha="left", y=1.02, color=INK)
    save(figure, f"fig_stages_{dataset}.png")


def figure_gallery(pair: ingest.Pair, dataset: str) -> None:
    """Every method in every bank, on one crop. The visual form of the benchmark."""
    grey = ingest.load_grey(pair.test_path)
    params = preprocess.resolve_params(grey)
    centre = _first_defect_centre(pair)
    window = crop(grey, CROP[dataset], centre)

    # (row label, the params key that selects this bank, the bank, its input)
    banks = [("Noise removal\n(task 1.4)", "denoise", preprocess.DENOISERS, window),
             ("Contrast enhancement\n(task 1.5)", "enhance", preprocess.ENHANCERS, window),
             ("Binarisation\n(task 1.6)", "binarise", preprocess.BINARISERS,
              preprocess.ENHANCERS[params["enhance"]](window, params))]

    columns = max(len(bank) for _, _, bank, _ in banks)
    figure, axes = plt.subplots(len(banks), columns, figsize=(3.0 * columns, 3.4 * len(banks)))

    for row, (label, key, bank, source) in enumerate(banks):
        for column in range(columns):
            axis = axes[row][column]
            if column >= len(bank):
                axis.axis("off")
                continue
            name = sorted(bank)[column]
            # Naming the method the pipeline actually selected turns the gallery
            # from a catalogue into an explanation of the current configuration.
            in_use = params.get(key) == name
            show(axis, bank[name](source, params),
                 f"{name}{'  ← in use' if in_use else ''}")
            if in_use:
                for spine in axis.spines.values():
                    spine.set_edgecolor(FOUND_COLOUR)
                    spine.set_linewidth(2.2)
            if column == 0:
                # A row label belongs beside its row, not underneath the first
                # panel where it reads as a caption for that one image.
                axis.set_ylabel(label, fontsize=9, fontweight="bold",
                                color=INK, labelpad=8)

    figure.suptitle(
        f"Every method in each bank — {dataset.upper()} "
        f"({params['profile']} profile). Row 3 is thresholded after the enhancement in use.",
        fontsize=10.5, fontweight="bold", x=0.005, ha="left", y=1.005, color=INK)
    figure.tight_layout()
    save(figure, f"fig_gallery_{dataset}.png")


def figure_detection(pair: ingest.Pair, dataset: str) -> None:
    """Template, test, the difference, the cleanup, and the detections scored."""
    from experiments.benchmark_pipeline import BASELINES, PREDICTION_PADDING_PX
    params = dict(BASELINES[dataset])

    prepared = preprocess.preprocess_pair(pair.template_path, pair.test_path, params)
    alignment = registration.register(prepared.test_bin, prepared.template_bin)
    removed, added = difference.signed_difference(prepared.template_bin, alignment.aligned)
    raw = cv2.bitwise_or(removed, added)
    cleaned = cv2.bitwise_or(morphology.clean_difference(removed, params),
                             morphology.clean_difference(added, params))

    report = inspect_pair(str(pair.template_path), str(pair.test_path), params)
    truth = (parse_annotation_file(pair.annotation_path) if pair.dataset == "deeppcb"
             else ingest.parse_voc_annotation(pair.annotation_path))

    scale = prepared.template_bin.shape[1] / ingest.load_grey(pair.test_path).shape[1]
    figure, axes = plt.subplots(1, 5, figsize=(15.5, 3.7))
    show(axes[0], prepared.template_bin, "template", "the reference board")
    show(axes[1], prepared.test_bin, "test", "the board under inspection")
    show(axes[2], raw, "raw difference",
         f"{int(raw.sum() // 255):,} differing pixels")
    show(axes[3], cleaned, "after morphology",
         f"opening {params['morph_open_kernel']} -> "
         f"{int(cleaned.sum() // 255):,} pixels")
    show(axes[4], prepared.test_bin, "result",
         f"{len(truth)} in truth, {len(report.defects)} found")

    for box in truth:
        x, y, w, h = [v * scale for v in box["bbox"]]
        axes[4].add_patch(patches.Rectangle((x, y), w, h, fill=False,
                                            edgecolor=TRUTH_COLOUR, linewidth=1.6))
    for defect in report.defects:
        x, y, w, h = blob_extraction.to_evaluation_bbox(
            defect.bbox, prepared.test_bin.shape,
            padding=PREDICTION_PADDING_PX[dataset])
        axes[4].add_patch(patches.Rectangle((x, y), w, h, fill=False,
                                            edgecolor=FOUND_COLOUR, linewidth=1.2,
                                            linestyle="--"))

    # Below the panel, not on it: a legend box over the board hides the very
    # detections it is labelling.
    axes[4].legend(handles=[
        patches.Patch(edgecolor=TRUTH_COLOUR, facecolor="none", label="ground truth"),
        patches.Patch(edgecolor=FOUND_COLOUR, facecolor="none", label="detected"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2,
        fontsize=7, frameon=False)

    figure.suptitle(
        f"From two images to a verdict — {dataset.upper()}, board {pair.name} "
        f"(registration: {alignment.method})",
        fontsize=10.5, fontweight="bold", x=0.005, ha="left", y=1.02, color=INK)
    save(figure, f"fig_detection_{dataset}.png")


def figure_rectify() -> None:
    """A rotated board, how it is found, and the result of levelling it."""
    pair = next(p for p in ingest.index_hripcb(rotated=True)
                if p.angle_deg and abs(p.angle_deg) >= 8)
    rotated = ingest.load_grey(pair.test_path)
    small = cv2.resize(rotated, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    mask = rectify.board_mask(small)
    estimate, method = rectify.estimate_angle_detailed(small)
    levelled, _ = rectify.rectify(small)

    outline = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    rect, _ = rectify.board_rect_detailed(small)
    if rect is not None:
        # OpenCV takes BGR; this is FOUND_COLOUR so the outline matches the
        # colour used for detections in the other figures.
        cv2.drawContours(outline, [cv2.boxPoints(rect).astype(np.int32)], -1,
                         (52, 104, 235), 6)

    figure, axes = plt.subplots(1, 4, figsize=(13.0, 3.6))
    show(axes[0], small, "1. rotated board", f"dataset says {pair.angle_deg:+.0f}°")
    show(axes[1], mask, "2. board segmented", f"corner flood fill ({method})")
    axes[2].imshow(cv2.cvtColor(outline, cv2.COLOR_BGR2RGB), interpolation="nearest")
    axes[2].set_title("3. minimum-area rectangle", fontsize=9, fontweight="bold", color=INK, pad=4)
    axes[2].set_xlabel(f"measured {estimate:+.2f}°", fontsize=7.5, color=INK_SOFT, labelpad=3)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    show(axes[3], levelled, "4. levelled",
         f"residual {rectify.estimate_angle(levelled):+.2f}°")

    figure.suptitle(
        f"Geometric rectification — HRIPCB board {pair.name}",
        fontsize=10.5, fontweight="bold", x=0.005, ha="left", y=1.02, color=INK)
    save(figure, "fig_rectify_demo.png")


def main() -> None:
    datasets = sys.argv[1:] or ["deeppcb", "hripcb"]
    print("Writing figures to outputs/figures/")
    for dataset in datasets:
        pairs = (ingest.index_deeppcb(split="test", limit=1) if dataset == "deeppcb"
                 else ingest.index_hripcb()[:1])
        pair = pairs[0]
        figure_stages(pair, dataset)
        figure_gallery(pair, dataset)
        figure_detection(pair, dataset)
    figure_rectify()


if __name__ == "__main__":
    main()
