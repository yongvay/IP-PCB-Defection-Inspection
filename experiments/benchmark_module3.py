"""Classification benchmark — Module 3, tasks 3.2 and 3.7.

Owner: Ng Zhi Xuan. Primary evidence for the Chapter 4 discussion.

What this measures, and why it is not the same as the Module 1 benchmark
-----------------------------------------------------------------------
``experiments/benchmark_pipeline.py`` sweeps preprocessing choices and scores
them on localisation: did the pipeline find the defect. That question is
answered by Modules 1 and 2. This script holds the preprocessing fixed and
varies the *classifier*, scoring three things separately:

* localisation F1, which should be identical across classifier settings and is
  reported precisely so that it can be checked — if it moves, something other
  than the classifier changed and the comparison is invalid;
* classification F1, which requires the label to match as well;
* conditional class accuracy, the fraction of correctly localised defects that
  were also correctly named, which is the classifier's own score isolated from
  detection performance.

The third figure is the one the Chapter 4 argument rests on, because it is the
only one that does not move when Module 1 improves.

Splits
------
Thresholds are chosen on ``trainval`` and reported on ``test``. Tuning and
reporting on the same images overstates performance and is the easiest
methodological criticism to make of a project like this. Set
``DEEPPCB_SPLIT=trainval`` in the environment when tuning.

Run::

    python -m experiments.benchmark_module3                # both classifiers
    python -m experiments.benchmark_module3 ring           # ring-width sweep
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

from experiments.common import OUTPUT_DIR, save_table
from src.module1 import ingest
from src.module1.ground_truth import parse_annotation_file
from src.module2.blobs import to_evaluation_bbox
from src.module3 import connectivity
from src.module3.evaluate import Evaluation
from src.pipeline import inspect_pair

# The preprocessing baseline is held fixed at Module 1's verified DeepPCB
# setting, so that any change in the numbers is attributable to the classifier
# and to nothing else.
FIXED_PIPELINE = {
    "denoise": "median", "median_kernel": 3, "enhance": "none",
    "binarise": "otsu", "morph_open_kernel": 5, "morph_close_kernel": 5,
    "min_blob_area": 40,
}

CLASSIFIERS = ("descriptor", "connectivity")
RING_WIDTHS = (2, 3, 4, 6, 8)
IOU_THRESHOLDS = (0.33, 0.50)
PREDICTION_PADDING_PX = 10
SAMPLE_SIZE = 80


def load_pairs(limit: int = SAMPLE_SIZE) -> list[ingest.Pair]:
    split = os.environ.get("DEEPPCB_SPLIT", "test")
    return ingest.index_deeppcb(split=split, limit=limit)


def truth_for(pair: ingest.Pair) -> list[tuple[tuple[int, int, int, int], str]]:
    """The answer key for one board, as (bbox, label) pairs."""
    if pair.annotation_path is None or not Path(pair.annotation_path).exists():
        return []
    return [(box["bbox"], box["label"])
            for box in parse_annotation_file(Path(pair.annotation_path))]


def run_configuration(pairs: list[ingest.Pair],
                      params: dict,
                      iou_threshold: float) -> tuple[Evaluation, int]:
    """Inspect every board under one configuration and accumulate the scores."""
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
            (to_evaluation_bbox(defect.bbox, (640, 640),
                                padding=PREDICTION_PADDING_PX),
             defect.defect_class)
            for defect in report.defects
        ]
        evaluation.add_board(pair.name, predictions, truths, report.runtime_s)

    return evaluation, failures


def compare_classifiers(pairs: list[ingest.Pair]) -> pd.DataFrame:
    """The headline table: the descriptor baseline against connectivity."""
    rows = []
    for classifier in CLASSIFIERS:
        params = {**FIXED_PIPELINE, "classifier": classifier}
        for threshold in IOU_THRESHOLDS:
            started = time.perf_counter()
            evaluation, failures = run_configuration(pairs, params, threshold)
            row = {"classifier": classifier, **evaluation.summary(),
                   "failures": failures,
                   "wall_clock_s": round(time.perf_counter() - started, 1)}
            rows.append(row)
            print(f"  {classifier:<14} IoU {threshold:.2f}  "
                  f"loc F1 {row['loc_f1']:.4f}  cls F1 {row['cls_f1']:.4f}  "
                  f"class acc {row['class_accuracy']:.4f}")

            # The per-class table, confusion matrix and per-board log are only
            # written for the reported threshold, because four near-identical
            # sets of files would obscure which one Chapter 4 quotes.
            if threshold == 0.50:
                evaluation.write_csv(OUTPUT_DIR, prefix=f"module3_{classifier}")

    return pd.DataFrame(rows)


def sweep_ring_width(pairs: list[ingest.Pair]) -> pd.DataFrame:
    """How sensitive the connectivity classifier is to its one free parameter.

    A rule set with a parameter that has never been swept is a rule set whose
    result cannot be trusted, so this is reported even though the default is
    unlikely to move. A flat curve is itself the finding: it means the
    classifier is reading a real structural property rather than an artefact of
    one kernel size.
    """
    rows = []
    original = connectivity.RING_WIDTH_PX
    try:
        for width in RING_WIDTHS:
            # The rule set reads its default at call time, so the module
            # attribute is the sweep knob. Restored in the finally block.
            connectivity.RING_WIDTH_PX = width
            evaluation, failures = run_configuration(
                pairs, {**FIXED_PIPELINE, "classifier": "connectivity"}, 0.50
            )
            rows.append({"ring_width_px": width, **evaluation.summary(),
                         "failures": failures})
            print(f"  ring {width:>2} px   class acc "
                  f"{rows[-1]['class_accuracy']:.4f}")
    finally:
        connectivity.RING_WIDTH_PX = original

    return pd.DataFrame(rows)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    split = os.environ.get("DEEPPCB_SPLIT", "test")
    pairs = load_pairs()
    print(f"DeepPCB {split} split: {len(pairs)} pairs, mode {mode}")

    if mode == "ring":
        frame = sweep_ring_width(pairs)
        frame.insert(0, "split", split)
        save_table(frame, "module3_ring_sweep.csv")
        return

    frame = compare_classifiers(pairs)
    frame.insert(0, "split", split)
    save_table(frame, "module3_classifier_comparison.csv")

    print(
        "\nFiles written to outputs/:\n"
        "  module3_classifier_comparison.csv   headline table, Chapter 4\n"
        "  module3_<classifier>_per_class.csv  per-class precision/recall/F1\n"
        "  module3_<classifier>_confusion.csv  confusion matrix\n"
        "  module3_<classifier>_per_board.csv  one row per board\n"
        "  module3_<classifier>_summary.csv    one row per run\n"
    )


if __name__ == "__main__":
    main()
