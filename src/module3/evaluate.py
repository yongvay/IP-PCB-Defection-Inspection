"""Evaluation harness — Task 3.7.

Owner: Ng Zhi Xuan. Seeded by Yong Vay during integration so that Objective 1
can be measured from Week 2 rather than Week 6, because an objective that is
only measured at the end cannot be steered towards.

Greedy IoU matching is used rather than optimal assignment: detections are
matched to the highest-IoU unclaimed ground-truth box. On boards with 3 to 12
well-separated defects the two agree, and greedy matching is what the DeepPCB
benchmark script itself does.

TODO (task 3.7): per-class precision/recall/F1, confusion matrix, runtime
benchmarking, and CSV export for Xing Szen's Chapter 4 figures.
"""

from dataclasses import dataclass


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
    true_positives: int
    false_positives: int
    false_negatives: int

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


def score_pair(predicted_boxes: list[tuple[int, int, int, int]],
               truth_boxes: list[tuple[int, int, int, int]],
               iou_threshold: float = 0.5) -> Scores:
    """Match predictions to ground truth for one image and count the outcomes.

    Note on the threshold. SMART Objective 1 sets IoU 0.5, while the DeepPCB
    authors benchmark their own detector at IoU 0.33. The stricter figure is
    kept because the system meets it, but Chapter 4 should report both so the
    comparison against published results is honest.
    """
    claimed: set[int] = set()
    true_positives = 0

    for prediction in predicted_boxes:
        best_score, best_index = 0.0, None
        for index, truth in enumerate(truth_boxes):
            if index in claimed:
                continue
            score = iou(prediction, truth)
            if score > best_score:
                best_score, best_index = score, index

        if best_index is not None and best_score >= iou_threshold:
            claimed.add(best_index)
            true_positives += 1

    return Scores(
        true_positives=true_positives,
        false_positives=len(predicted_boxes) - true_positives,
        false_negatives=len(truth_boxes) - true_positives,
    )
