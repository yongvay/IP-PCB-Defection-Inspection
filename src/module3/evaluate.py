"""Evaluation harness — task 3.7.

Owner: Ng Zhi Xuan (Module 3).

Separating the two questions being asked
----------------------------------------
"Did the system find the defect?" and "did it name the defect correctly?" are
different questions with different owners, and reporting one number for both
hides which module is responsible for a shortfall. This harness answers them
separately:

* **Localisation** scores a detection as correct if its box overlaps a
  ground-truth box by at least the IoU threshold, whatever label it carries.
  This measures Modules 1 and 2 — preprocessing, registration, differencing
  and blob extraction.
* **Classification** additionally requires the label to match. The gap between
  the two figures is Module 3's error and nobody else's.
* **Conditional class accuracy** is the fraction of correctly localised
  detections that were also correctly named. This is the cleanest measure of
  the classifier alone, because it is not depressed by defects the pipeline
  never found in the first place.

Chapter 4 should report all three. A single combined F1 lets a weak classifier
hide behind strong detection, and lets a weak detector hide behind a classifier
that is right about the few defects it is given.

Matching strategy
-----------------
Greedy IoU matching is used rather than optimal (Hungarian) assignment:
detections are matched to the highest-IoU unclaimed ground-truth box. On boards
carrying 3 to 12 well-separated defects the two agree, and greedy matching is
what the DeepPCB benchmark script itself does, so the comparison against
published results stays like for like.

Matching is deliberately performed on geometry alone, before labels are
considered. Matching on box *and* label together would let a misclassified
detection match a different, more distant defect of the coincidentally correct
class, which inflates both figures.

Only the standard library is imported here. The CSV writing uses ``csv``
rather than pandas so that ``src/`` stays free of analysis dependencies —
importing pandas to inspect a single board is the kind of coupling that makes
it unclear which code the marker is being asked to assess. The experiment
scripts under ``experiments/`` are where pandas belongs.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.contracts import DEFECT_CLASSES

# Stands in for "nothing was here" on both axes of the confusion matrix: a
# false positive is background predicted as a class, and a missed defect is a
# class predicted as background. Naming it explicitly keeps every detection
# accounted for, so the matrix rows and columns always sum to the true totals.
BACKGROUND = "background"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Intersection over union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


@dataclass
class Scores:
    """Counts of the three matching outcomes, with the derived rates."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def support(self) -> int:
        """How many ground-truth instances this row was computed from.

        Reported alongside every per-class figure because a recall of 1.00 over
        four instances and one over four hundred are not the same result, and a
        table without support invites the reader to treat them as though they
        were.
        """
        return self.true_positives + self.false_negatives

    def add(self, other: "Scores") -> None:
        self.true_positives += other.true_positives
        self.false_positives += other.false_positives
        self.false_negatives += other.false_negatives


@dataclass
class PairOutcome:
    """The result of matching one board's detections against its answer key."""

    matches: list[tuple[int, int, float]] = field(default_factory=list)
    unmatched_predictions: list[int] = field(default_factory=list)
    unmatched_truths: list[int] = field(default_factory=list)


def match_boxes(predicted_boxes: list[tuple[int, int, int, int]],
                truth_boxes: list[tuple[int, int, int, int]],
                iou_threshold: float = 0.5) -> PairOutcome:
    """Greedily pair detections with ground-truth boxes on overlap alone.

    Returns the matched index pairs with their IoU, plus the indices that found
    no partner on either side. Labels are not consulted; that is the caller's
    job, and keeping the two steps apart is what allows the same matching to
    serve both the localisation and the classification figures.
    """
    claimed: set[int] = set()
    outcome = PairOutcome()

    for prediction_index, prediction in enumerate(predicted_boxes):
        best_score, best_index = 0.0, None
        for truth_index, truth in enumerate(truth_boxes):
            if truth_index in claimed:
                continue
            score = iou(prediction, truth)
            if score > best_score:
                best_score, best_index = score, truth_index

        if best_index is not None and best_score >= iou_threshold:
            claimed.add(best_index)
            outcome.matches.append((prediction_index, best_index, best_score))
        else:
            outcome.unmatched_predictions.append(prediction_index)

    outcome.unmatched_truths = [
        index for index in range(len(truth_boxes)) if index not in claimed
    ]
    return outcome


def score_pair(predicted_boxes: list[tuple[int, int, int, int]],
               truth_boxes: list[tuple[int, int, int, int]],
               iou_threshold: float = 0.5) -> Scores:
    """Localisation-only scoring for one board.

    Retained with its original signature because ``experiments/
    benchmark_pipeline.py`` and the integration tests both call it. It is now a
    thin wrapper over ``match_boxes``, so the two code paths can no longer
    disagree about what counts as a match.

    Note on the threshold. SMART Objective 1 sets IoU 0.5, while the DeepPCB
    authors benchmark their own detector at IoU 0.33. The stricter figure is
    kept because the system meets it, but Chapter 4 reports both so the
    comparison against published results is honest.
    """
    outcome = match_boxes(predicted_boxes, truth_boxes, iou_threshold)
    return Scores(
        true_positives=len(outcome.matches),
        false_positives=len(outcome.unmatched_predictions),
        false_negatives=len(outcome.unmatched_truths),
    )


# ---------------------------------------------------------------------------
# Accumulating results across many boards
# ---------------------------------------------------------------------------
class Evaluation:
    """Accumulates detections across a run and reports the metrics.

    Usage::

        evaluation = Evaluation(iou_threshold=0.5)
        for board in boards:
            evaluation.add_board(
                board.name,
                predictions=[(defect.bbox, defect.defect_class) for defect in report.defects],
                truths=[(box["bbox"], box["label"]) for box in answer_key],
                runtime_s=report.runtime_s,
            )
        print(evaluation.summary())
        evaluation.write_csv(Path("outputs"), prefix="module3")
    """

    def __init__(self,
                 iou_threshold: float = 0.5,
                 classes: tuple[str, ...] = DEFECT_CLASSES) -> None:
        self.iou_threshold = iou_threshold
        self.classes = tuple(classes)

        self.localisation = Scores()
        self.classification = Scores()

        # confusion[truth_label][predicted_label]. Both axes carry BACKGROUND,
        # so a false positive is confusion[BACKGROUND][predicted] and a missed
        # defect is confusion[truth][BACKGROUND].
        self.confusion: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        self.runtimes: list[float] = []
        self.boards: list[dict[str, object]] = []
        self.matched_ious: list[float] = []

    # -- recording ---------------------------------------------------------
    def add_board(self,
                  name: str,
                  predictions: list[tuple[tuple[int, int, int, int], str]],
                  truths: list[tuple[tuple[int, int, int, int], str]],
                  runtime_s: float | None = None) -> dict[str, object]:
        """Score one board and fold its counts into the running totals.

        ``predictions`` and ``truths`` are lists of (bbox, label). The caller is
        responsible for having applied the annotation-padding convention to the
        predicted boxes before they arrive here — that correction belongs to the
        dataset, not to the scoring, and burying it in this method would hide a
        methodological decision inside a metric.
        """
        outcome = match_boxes(
            [box for box, _ in predictions],
            [box for box, _ in truths],
            self.iou_threshold,
        )

        correctly_named = 0
        for prediction_index, truth_index, overlap in outcome.matches:
            predicted_label = predictions[prediction_index][1]
            truth_label = truths[truth_index][1]
            self.confusion[truth_label][predicted_label] += 1
            self.matched_ious.append(overlap)
            if predicted_label == truth_label:
                correctly_named += 1

        for prediction_index in outcome.unmatched_predictions:
            self.confusion[BACKGROUND][predictions[prediction_index][1]] += 1
        for truth_index in outcome.unmatched_truths:
            self.confusion[truths[truth_index][1]][BACKGROUND] += 1

        localised = len(outcome.matches)
        self.localisation.add(Scores(
            true_positives=localised,
            false_positives=len(outcome.unmatched_predictions),
            false_negatives=len(outcome.unmatched_truths),
        ))
        # A detection that is localised but misnamed is counted twice over:
        # once as a false positive of the class it claimed, once as a false
        # negative of the class it should have been. That is the standard
        # detection convention and it is what makes the per-class rows of the
        # confusion matrix add up.
        self.classification.add(Scores(
            true_positives=correctly_named,
            false_positives=len(predictions) - correctly_named,
            false_negatives=len(truths) - correctly_named,
        ))

        if runtime_s is not None:
            self.runtimes.append(runtime_s)

        record = {
            "board": name,
            "predicted": len(predictions),
            "truth": len(truths),
            "localised": localised,
            "correctly_named": correctly_named,
            "runtime_s": round(runtime_s, 4) if runtime_s is not None else "",
        }
        self.boards.append(record)
        return record

    # -- reporting ---------------------------------------------------------
    def per_class(self) -> list[dict[str, object]]:
        """Precision, recall, F1 and support for each of the six classes.

        Derived from the confusion matrix rather than counted separately, so
        the table and the matrix cannot disagree.
        """
        rows = []
        for defect_class in self.classes:
            true_positives = self.confusion[defect_class][defect_class]
            false_negatives = sum(
                count for label, count in self.confusion[defect_class].items()
                if label != defect_class
            )
            false_positives = sum(
                self.confusion[truth_label][defect_class]
                for truth_label in list(self.confusion)
                if truth_label != defect_class
            )
            scores = Scores(true_positives, false_positives, false_negatives)
            rows.append({
                "defect_class": defect_class,
                "support": scores.support,
                "true_positives": scores.true_positives,
                "false_positives": scores.false_positives,
                "false_negatives": scores.false_negatives,
                "precision": round(scores.precision, 4),
                "recall": round(scores.recall, 4),
                "f1": round(scores.f1, 4),
            })
        return rows

    def macro_f1(self) -> float:
        """Unweighted mean of the six per-class F1 scores.

        Reported alongside the micro-averaged figure because the DeepPCB class
        distribution is uneven. A micro average is dominated by the common
        classes and can look healthy while a rare class is never detected at
        all; the macro average makes that failure visible.
        """
        rows = [row for row in self.per_class() if row["support"] > 0]
        if not rows:
            return 0.0
        return round(sum(float(row["f1"]) for row in rows) / len(rows), 4)

    def conditional_class_accuracy(self) -> float:
        """Of the defects that were found, what fraction were named correctly.

        This is the classifier's own score, isolated from detection
        performance, and it is the number Chapter 4 should quote when comparing
        the connectivity classifier against the descriptor baseline.
        """
        localised = self.localisation.true_positives
        return round(
            self.classification.true_positives / localised, 4
        ) if localised else 0.0

    def summary(self) -> dict[str, object]:
        """Every headline figure for one run, in one flat record."""
        runtimes = self.runtimes
        return {
            "iou_threshold": self.iou_threshold,
            "boards": len(self.boards),
            "loc_precision": round(self.localisation.precision, 4),
            "loc_recall": round(self.localisation.recall, 4),
            "loc_f1": round(self.localisation.f1, 4),
            "cls_precision": round(self.classification.precision, 4),
            "cls_recall": round(self.classification.recall, 4),
            "cls_f1": round(self.classification.f1, 4),
            "macro_f1": self.macro_f1(),
            "class_accuracy": self.conditional_class_accuracy(),
            "mean_iou_matched": round(
                sum(self.matched_ious) / len(self.matched_ious), 4
            ) if self.matched_ious else 0.0,
            "mean_runtime_s": round(sum(runtimes) / len(runtimes), 4) if runtimes else 0.0,
            "max_runtime_s": round(max(runtimes), 4) if runtimes else 0.0,
            "within_3s_pct": round(
                100.0 * sum(1 for value in runtimes if value <= 3.0) / len(runtimes), 2
            ) if runtimes else 0.0,
        }

    def confusion_rows(self) -> list[dict[str, object]]:
        """The confusion matrix flattened into one row per truth label."""
        labels = list(self.classes) + [BACKGROUND]
        rows = []
        for truth_label in labels:
            row: dict[str, object] = {"truth": truth_label}
            for predicted_label in labels:
                row[predicted_label] = self.confusion[truth_label][predicted_label]
            rows.append(row)
        return rows

    # -- export ------------------------------------------------------------
    def write_csv(self, directory: Path, prefix: str = "module3") -> list[Path]:
        """Write four tables for the Chapter 4 figures and return their paths.

        Four files rather than one wide table, because they have genuinely
        different shapes: one row per class, one row per truth label, one row
        per board, one row for the run. Forcing them into a single CSV would
        mean padding three of them with empty columns.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written = []

        written.append(_write_rows(
            directory / f"{prefix}_per_class.csv", self.per_class()))
        written.append(_write_rows(
            directory / f"{prefix}_confusion.csv", self.confusion_rows()))
        written.append(_write_rows(
            directory / f"{prefix}_per_board.csv", self.boards))
        written.append(_write_rows(
            directory / f"{prefix}_summary.csv", [self.summary()]))
        return written


def _write_rows(path: Path, rows: list[dict]) -> Path:
    """Write a list of uniform dictionaries as a CSV with a header."""
    if not rows:
        path.write_text("")
        return path
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
