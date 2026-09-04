"""Polarity diagnostic — is stage one reading copper the right way round?

Owner: Ng Zhi Xuan (Module 3).

Why this exists
---------------
On the HRIPCB benchmark the system localises defects acceptably (F1 0.685 at
IoU 0.50) but names almost none of them correctly (class accuracy 0.023 for the
connectivity classifier, 0.000 for the descriptor baseline). Those two figures
cannot both be explained by a weak classifier.

Stage one splits the six classes into two disjoint families from the sign of
the template-test difference: copper-removed gives open circuit, mouse bite and
pin hole; copper-added gives short, spur and spurious copper. A classifier
guessing at random *within the correct family* would still score around 0.33.
Scoring near zero means predictions are landing in the wrong family every time,
which is what a globally inverted polarity looks like.

``src/module2/difference.py`` sets ``COPPER_IS_DARK = True`` as a module
constant. That is correct for DeepPCB, whose images are binarised so that
copper falls dark. HRIPCB is colour photography of green boards thresholded
with adaptive mean, and there is no reason the same convention should hold.

What this measures
------------------
For each matched detection, the polarity the pipeline assigned is compared with
the polarity the ground-truth label implies. Agreement near 100% means stage
one is correct; agreement near 0% means it is inverted, and the fix is one
parameter rather than anything in Module 3.

DeepPCB is run first as a control. If the control does not come back near 100%,
the diagnostic itself is wrong and its HRIPCB reading should not be trusted.

Run::

    python -m tools.diagnose_polarity
    python -m tools.diagnose_polarity hripcb
"""

from __future__ import annotations

import sys
from pathlib import Path

from experiments.benchmark_module3 import COPPER_IS_DARK
from experiments.benchmark_pipeline import BASELINES, PREDICTION_PADDING_PX
from src.module1 import ingest
from src.module1.ground_truth import POLARITY, parse_annotation_file
from src.module2.blobs import to_evaluation_bbox
from src.module3.evaluate import match_boxes

BOARDS = 15
IMAGE_SHAPE = {"deeppcb": (640, 640), "hripcb": (10 ** 6, 10 ** 6)}


def truth_for(pair: ingest.Pair) -> list[tuple[tuple[int, int, int, int], str]]:
    if pair.annotation_path is None or not Path(pair.annotation_path).exists():
        return []
    if pair.dataset == "deeppcb":
        boxes = parse_annotation_file(Path(pair.annotation_path))
    else:
        boxes = ingest.parse_voc_annotation(pair.annotation_path)
    return [(box["bbox"], box["label"]) for box in boxes]


def agreement(predicted_polarities: list[str],
              truth_polarities: list[str]) -> tuple[int, int]:
    """Count how many matched detections agree on polarity.

    Kept separate from the pipeline call so that it can be tested without a
    dataset, which is the only part of this script that has any logic in it.
    """
    assert len(predicted_polarities) == len(truth_polarities)
    matches = sum(1 for predicted, truth
                  in zip(predicted_polarities, truth_polarities)
                  if predicted == truth)
    return matches, len(truth_polarities)


def diagnose(dataset: str) -> None:
    from src.pipeline import inspect_pair

    key = "hripcb" if dataset.startswith("hripcb") else "deeppcb"
    if key == "deeppcb":
        pairs = ingest.index_deeppcb(split="test", limit=BOARDS)
    else:
        everything = ingest.index_hripcb(rotated=dataset.endswith("rotated"))
        step = max(1, len(everything) // BOARDS)
        pairs = everything[::step][:BOARDS]

    # Honours whatever polarity the benchmark is configured to use, so that
    # re-running this after a fix confirms the fix rather than re-reporting
    # the original fault.
    params = {**BASELINES[key], "classifier": "connectivity",
              "copper_is_dark": COPPER_IS_DARK[key]}
    predicted_polarities: list[str] = []
    truth_polarities: list[str] = []

    for pair in pairs:
        truths = truth_for(pair)
        if not truths:
            continue
        try:
            report = inspect_pair(str(pair.template_path), str(pair.test_path),
                                  dict(params))
        except Exception as error:
            print(f"  {pair.name}: skipped ({type(error).__name__})")
            continue

        predictions = [
            (to_evaluation_bbox(defect.bbox, IMAGE_SHAPE[key],
                                padding=PREDICTION_PADDING_PX[key]),
             str(defect.polarity))
            for defect in report.defects
        ]
        outcome = match_boxes([box for box, _ in predictions],
                              [box for box, _ in truths],
                              iou_threshold=0.50)

        for prediction_index, truth_index, _ in outcome.matches:
            predicted_polarities.append(predictions[prediction_index][1])
            truth_polarities.append(POLARITY[truths[truth_index][1]])

    correct, total = agreement(predicted_polarities, truth_polarities)
    if total == 0:
        print(f"{dataset}: no matched detections — nothing to diagnose")
        return

    percentage = 100.0 * correct / total
    print(f"\n{dataset}: {correct} of {total} matched detections agree on "
          f"polarity ({percentage:.1f}%)")

    # Show the direction of any disagreement, because "inverted" and "noisy"
    # produce very different percentages and call for different fixes.
    swapped = sum(1 for predicted, truth
                  in zip(predicted_polarities, truth_polarities)
                  if predicted != truth)
    print(f"  removed predicted: "
          f"{predicted_polarities.count('removed')}, "
          f"added predicted: {predicted_polarities.count('added')}")
    print(f"  removed in truth:  {truth_polarities.count('removed')}, "
          f"added in truth:  {truth_polarities.count('added')}")

    if percentage >= 90:
        print("  VERDICT: stage one is reading copper correctly.")
    elif percentage <= 15:
        print("  VERDICT: polarity is INVERTED on this dataset. Copper is "
              "light here, not dark, so COPPER_IS_DARK in "
              "src/module2/difference.py is wrong for it. This is a Module 2 "
              "configuration issue, not a classifier issue.")
    else:
        print(f"  VERDICT: polarity is unreliable rather than inverted "
              f"({swapped} disagreements). Binarisation is probably producing "
              "inconsistent copper polarity across boards.")


def main() -> None:
    datasets = sys.argv[1:] or ["deeppcb", "hripcb"]
    for dataset in datasets:
        if dataset == "deeppcb":
            print("Control — DeepPCB should come back near 100%.")
        diagnose(dataset)


if __name__ == "__main__":
    main()
