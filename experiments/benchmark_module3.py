"""Classification benchmark — Module 3, tasks 3.2 and 3.7.

Owner: Ng Zhi Xuan. Primary evidence for the Chapter 4 discussion.

What this measures, and why it is not the same as the Module 1 benchmark
-----------------------------------------------------------------------
``experiments/benchmark_pipeline.py`` sweeps preprocessing choices and scores
them on localisation: did the pipeline find the defect. That question is
answered by Modules 1 and 2. This script holds the preprocessing fixed and
varies the *classifier*, scoring three things separately:

* localisation F1, which should barely move across classifier settings and is
  reported precisely so that it can be checked — if it shifts, something other
  than the classifier changed and the comparison is invalid;
* classification F1, which requires the label to match as well;
* conditional class accuracy, the fraction of correctly localised defects that
  were also correctly named.

The third figure is the one the Chapter 4 argument rests on, because it is the
only one that does not move when Module 1 improves.

Datasets
--------
``deeppcb``
    The primary set. Pre-aligned template and test pairs, six classes, an
    answer key for every board. Thresholds are tuned here.
``hripcb``
    The secondary set: colour photographs at roughly five times the linear
    resolution, referenced against the ten boards in ``PCB_USED``. Running the
    classifier here tests whether the connectivity rules describe a real
    structural property or merely a property of one imaging setup.
``hripcb-rotated``
    The same boards photographed at an angle, rectified by task 1.8 before
    inspection. This is the robustness figure, and it is expected to be the
    weakest of the three: rectification leaves residual misalignment, which
    fragments the contact patches the connectivity rules count.

The preprocessing baseline and the annotation padding for each dataset are
imported from ``benchmark_pipeline`` rather than copied. Those constants were
measured by Chan Xing Szen, and a second copy that quietly drifts out of step
would make the two chapters contradict each other.

Splits
------
Thresholds are chosen on ``trainval`` and reported on ``test``. Tuning and
reporting on the same images overstates performance and is the easiest
methodological criticism to make of a project like this. Set
``DEEPPCB_SPLIT=trainval`` in the environment when tuning.

Run::

    python -m experiments.benchmark_module3                        # DeepPCB
    python -m experiments.benchmark_module3 compare hripcb         # HRIPCB upright
    python -m experiments.benchmark_module3 compare hripcb-rotated # robustness
    python -m experiments.benchmark_module3 ring                   # ring sweep
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

from experiments.benchmark_pipeline import (
    BASELINES,
    PREDICTION_PADDING_PX,
    SAMPLE,
)
from experiments.common import OUTPUT_DIR, save_table
from src.module1 import ingest
from src.module1.ground_truth import parse_annotation_file
from src.module2.blobs import to_evaluation_bbox
from src.module3 import connectivity
from src.module3.evaluate import Evaluation
from src.pipeline import inspect_pair

CLASSIFIERS = ("descriptor", "connectivity")
RING_WIDTHS = (2, 3, 4, 6, 8)

# MIN_CONTACT_PX is a pixel *area*, so the value that works at DeepPCB's
# ~48 px/mm is roughly 25 times too small at HRIPCB's ~5x linear resolution.
# The range spans both regimes so that one sweep covers either dataset.
CONTACT_AREAS = (6, 25, 50, 100, 150, 250, 400)
IOU_THRESHOLDS = (0.33, 0.50)

DATASETS = ("deeppcb", "hripcb", "hripcb-rotated")

# DeepPCB images are a known 640 x 640, so a predicted box can be clipped to
# the frame. HRIPCB images vary in size, and reading every one to learn its
# dimensions would cost more than the clipping is worth, so an effectively
# unbounded sentinel is used instead. Mirrors benchmark_pipeline.
IMAGE_SHAPE = {"deeppcb": (640, 640), "hripcb": (10 ** 6, 10 ** 6)}

# Which pixel value means copper, per dataset. DeepPCB is binarised so that
# copper falls dark; HRIPCB is colour photography thresholded with adaptive
# mean, where it falls light. Measured with tools/diagnose_polarity.py, which
# reported 100% polarity agreement on DeepPCB and 5% on HRIPCB before this was
# separated out. Leaving it hard-coded cost 92 points of class accuracy on
# HRIPCB while leaving localisation untouched.
COPPER_IS_DARK = {"deeppcb": True, "hripcb": False}


def base_name(dataset: str) -> str:
    """Strip the rotation qualifier to get the underlying dataset key."""
    return "hripcb" if dataset.startswith("hripcb") else "deeppcb"


def load_pairs(dataset: str) -> list[ingest.Pair]:
    """Index one dataset, sampled to a size that finishes in a coffee break."""
    key = base_name(dataset)

    if key == "deeppcb":
        split = os.environ.get("DEEPPCB_SPLIT", "test")
        return ingest.index_deeppcb(split=split, limit=SAMPLE[key])

    everything = ingest.index_hripcb(rotated=dataset.endswith("rotated"))
    # Sampled with a stride rather than by taking the first n, because HRIPCB
    # is ordered by defect class. Taking the first 30 would return 30 boards
    # carrying missing holes and nothing else, and every per-class row but one
    # would read zero.
    step = max(1, len(everything) // SAMPLE[key])
    return everything[::step][:SAMPLE[key]]


def truth_for(pair: ingest.Pair) -> list[tuple[tuple[int, int, int, int], str]]:
    """The answer key for one board, as (bbox, label) pairs.

    Both parsers return the same dictionary shape, so the only branch needed is
    which one to call.
    """
    if pair.annotation_path is None or not Path(pair.annotation_path).exists():
        return []

    if pair.dataset == "deeppcb":
        boxes = parse_annotation_file(Path(pair.annotation_path))
    else:
        boxes = ingest.parse_voc_annotation(pair.annotation_path)

    return [(box["bbox"], box["label"]) for box in boxes]


def run_configuration(pairs: list[ingest.Pair],
                      params: dict,
                      iou_threshold: float,
                      dataset: str) -> tuple[Evaluation, int]:
    """Inspect every board under one configuration and accumulate the scores."""
    key = base_name(dataset)
    evaluation = Evaluation(iou_threshold=iou_threshold)
    failures = 0

    for pair in pairs:
        truths = truth_for(pair)
        if not truths:
            continue
        try:
            report = inspect_pair(str(pair.template_path), str(pair.test_path),
                                  dict(params))
        except Exception:
            # A configuration that cannot process a board is a result about
            # that configuration, so it is counted rather than allowed to stop
            # the run.
            failures += 1
            continue

        predictions = [
            (to_evaluation_bbox(defect.bbox, IMAGE_SHAPE[key],
                                padding=PREDICTION_PADDING_PX[key]),
             defect.defect_class)
            for defect in report.defects
        ]
        evaluation.add_board(pair.name, predictions, truths, report.runtime_s)

    return evaluation, failures


def compare_classifiers(pairs: list[ingest.Pair], dataset: str) -> pd.DataFrame:
    """The headline table: the descriptor baseline against connectivity."""
    key = base_name(dataset)
    baseline = {**BASELINES[key], "copper_is_dark": COPPER_IS_DARK[key]}
    rows = []

    for classifier in CLASSIFIERS:
        params = {**baseline, "classifier": classifier}
        for threshold in IOU_THRESHOLDS:
            started = time.perf_counter()
            evaluation, failures = run_configuration(
                pairs, params, threshold, dataset
            )
            row = {"dataset": dataset, "classifier": classifier,
                   **evaluation.summary(), "failures": failures,
                   "wall_clock_s": round(time.perf_counter() - started, 1)}
            rows.append(row)
            print(f"  {classifier:<14} IoU {threshold:.2f}  "
                  f"loc F1 {row['loc_f1']:.4f}  cls F1 {row['cls_f1']:.4f}  "
                  f"class acc {row['class_accuracy']:.4f}")

            # The per-class table, confusion matrix and per-board log are only
            # written for the reported threshold, because four near-identical
            # sets of files would obscure which one Chapter 4 quotes.
            if threshold == 0.50:
                evaluation.write_csv(
                    OUTPUT_DIR, prefix=f"module3_{dataset}_{classifier}"
                )

    return pd.DataFrame(rows)


def sweep_parameter(pairs: list[ingest.Pair],
                    dataset: str,
                    attribute: str,
                    values: tuple[int, ...],
                    column: str) -> pd.DataFrame:
    """Sweep one connectivity threshold, restoring it afterwards.

    Both free parameters live as module attributes read at call time, so the
    sweep sets the attribute rather than threading an argument through the
    pipeline. The finally block matters: leaving a swept value in place would
    silently contaminate every later run in the same session.
    """
    key = base_name(dataset)
    baseline = {**BASELINES[key], "copper_is_dark": COPPER_IS_DARK[key]}
    rows = []
    original = getattr(connectivity, attribute)

    try:
        for value in values:
            setattr(connectivity, attribute, value)
            evaluation, failures = run_configuration(
                pairs, {**baseline, "classifier": "connectivity"}, 0.50, dataset
            )
            rows.append({"dataset": dataset, column: value,
                         **evaluation.summary(), "failures": failures})
            print(f"  {column} {value:>4}   class acc "
                  f"{rows[-1]['class_accuracy']:.4f}   "
                  f"cls F1 {rows[-1]['cls_f1']:.4f}")
    finally:
        setattr(connectivity, attribute, original)

    return pd.DataFrame(rows)


def sweep_ring_width(pairs: list[ingest.Pair], dataset: str) -> pd.DataFrame:
    """How sensitive the connectivity classifier is to the ring width.

    A rule set with a parameter that has never been swept is a rule set whose
    result cannot be trusted, so this is reported even though the default is
    unlikely to move. A flat curve is itself the finding: it means the
    classifier is reading a real structural property rather than an artefact of
    one kernel size.

    The optimum is expected to differ between the two datasets. HRIPCB images
    are about five times DeepPCB's linear resolution, and a ring measured in
    pixels does not transfer across that gap any more than a structuring
    element does.
    """
    return sweep_parameter(pairs, dataset, "RING_WIDTH_PX",
                           RING_WIDTHS, "ring_width_px")


def main() -> None:
    arguments = sys.argv[1:]
    mode = arguments[0] if arguments else "compare"
    dataset = arguments[1] if len(arguments) > 1 else "deeppcb"

    if dataset not in DATASETS:
        raise SystemExit(f"Unknown dataset '{dataset}'. Choose from: "
                         f"{', '.join(DATASETS)}")

    pairs = load_pairs(dataset)
    label = dataset
    if base_name(dataset) == "deeppcb":
        label += f" ({os.environ.get('DEEPPCB_SPLIT', 'test')} split)"
    print(f"{label}: {len(pairs)} pairs, mode {mode}")

    if mode == "ring":
        save_table(sweep_ring_width(pairs, dataset),
                   f"module3_{dataset}_ring_sweep.csv")
        return

    if mode == "contact":
        save_table(
            sweep_parameter(pairs, dataset, "MIN_CONTACT_PX",
                            CONTACT_AREAS, "min_contact_px"),
            f"module3_{dataset}_contact_sweep.csv")
        return

    save_table(compare_classifiers(pairs, dataset),
               f"module3_{dataset}_classifier_comparison.csv")

    print(
        f"\nFiles written to outputs/:\n"
        f"  module3_{dataset}_classifier_comparison.csv   headline table\n"
        f"  module3_{dataset}_<classifier>_per_class.csv  per-class P/R/F1\n"
        f"  module3_{dataset}_<classifier>_confusion.csv  confusion matrix\n"
        f"  module3_{dataset}_<classifier>_per_board.csv  one row per board\n"
        f"  module3_{dataset}_<classifier>_summary.csv    one row per run\n"
    )


if __name__ == "__main__":
    main()
