"""Preprocessing benchmark in detection units — Module 1, tasks 1.4 to 1.6, 1.10.

Owner: Chan Xing Szen. Primary evidence for SMART Objective 2.

Objective 2 is worded in detection F1, not in image-quality scores, and the
distinction is deliberate. A filter that produces a better-looking image has not
yet been shown to produce a better inspection, and only one of those is what the
system is for. This experiment therefore runs the complete pipeline end to end
and scores its output against the ground truth, changing one preprocessing
choice at a time.

One factor at a time, not a full grid
-------------------------------------
Four denoisers, four enhancers, three binarisers and four morphological settings
is a grid of 192 configurations, and at roughly a second per board over a
meaningful sample that is a run measured in hours for a result nobody can read.
Sweeping one factor at a time from a fixed baseline costs 15 runs instead, and
answers the question actually being asked — which choice at each stage is best,
holding the others at a sensible setting. Its limitation is that it cannot see
an interaction between two stages, and Chapter 4 should say so rather than
imply the search was exhaustive.

Scoring convention
------------------
Both datasets annotate a margin around each defect rather than the changed
pixels themselves, so predictions have the same margin applied before scoring.
This is a correction for an annotation convention, not a thumb on the scale, and
Chapter 3 must state it: without it, identical detections score recall 0.01
instead of 0.71 on DeepPCB.

The margin was measured rather than assumed. For each detection whose centroid
falls inside a ground-truth box, the gap between the two boxes' edges was
recorded: DeepPCB sits at 10 px, HRIPCB at a median of 19 px (quartiles 15 and
22) across 20 matched defects. The two differ because the datasets differ by
roughly a factor of five in linear resolution, which is exactly what a margin
expressed in pixels should do.

Run:  python -m experiments.benchmark_pipeline <dataset> [stage ...]

Stages are run separately and appended, because the sandbox this was developed
in caps a single command at three minutes. Passing no stage runs them all.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import OUTPUT_DIR, save_table
from src.module1 import ingest
from src.module1.ground_truth import parse_annotation_file
from src.module2 import blobs as blob_extraction
from src.module3.evaluate import score_pair
from src.pipeline import inspect_pair

# Baselines differ because the datasets differ by a factor of about five in
# linear resolution, so a structuring element measured in pixels does not
# transfer between them.
BASELINES = {
    "deeppcb": {"denoise": "median", "median_kernel": 3, "enhance": "none",
                "binarise": "otsu", "morph_open_kernel": 5,
                "morph_close_kernel": 5, "min_blob_area": 40},
    # Round two. The first pass ran from median + CLAHE + Otsu and scored
    # F1 0.265; sweeping one stage at a time from there showed bilateral
    # (0.455) and adaptive mean (0.621) each beating it on their own, so the
    # baseline was moved to both and the sweep repeated. That is coordinate
    # descent, and Chapter 4 should describe it as such rather than implying a
    # single exhaustive pass.
    "hripcb": {"denoise": "bilateral", "enhance": "clahe",
               "binarise": "adaptive_mean", "adaptive_block": 35, "adaptive_c": 5,
               "morph_open_kernel": 5, "morph_close_kernel": 5,
               "min_blob_area": 300},
}

SWEEPS = {
    "denoise": ["none", "gaussian", "median", "bilateral"],
    "enhance": ["none", "hist_eq", "clahe", "linear_stretch"],
    "binarise": ["otsu", "adaptive_mean", "adaptive_gaussian"],
    "morph_open_kernel": [0, 3, 5, 7, 9, 11],
}

SAMPLE = {"deeppcb": 80, "hripcb": 30}

# Measured, not assumed — see the module docstring.
PREDICTION_PADDING_PX = {"deeppcb": 10, "hripcb": 19}
IOU_THRESHOLDS = (0.33, 0.50)


def load_truth(pair: ingest.Pair) -> list[tuple[int, int, int, int]]:
    """Ground-truth boxes for either dataset, in one shape."""
    if pair.annotation_path is None or not Path(pair.annotation_path).exists():
        return []
    if pair.dataset == "deeppcb":
        boxes = parse_annotation_file(Path(pair.annotation_path))
    else:
        boxes = ingest.parse_voc_annotation(pair.annotation_path)
    return [box["bbox"] for box in boxes]


def evaluate_config(pairs: list[ingest.Pair], params: dict, dataset: str) -> dict:
    """Run one configuration over the sample and return its aggregate scores."""
    totals = {threshold: [0, 0, 0] for threshold in IOU_THRESHOLDS}
    runtimes, failures = [], 0

    for pair in pairs:
        truth = load_truth(pair)
        if not truth:
            continue
        try:
            report = inspect_pair(str(pair.template_path), str(pair.test_path), dict(params))
        except Exception:
            # A configuration that cannot process a board is a result about that
            # configuration, so it is counted rather than allowed to stop the run.
            failures += 1
            continue

        runtimes.append(report.runtime_s)
        shape = (640, 640) if dataset == "deeppcb" else (10 ** 6, 10 ** 6)
        predicted = [blob_extraction.to_evaluation_bbox(
            defect.bbox, shape, padding=PREDICTION_PADDING_PX[dataset])
            for defect in report.defects]

        for threshold in IOU_THRESHOLDS:
            scores = score_pair(predicted, truth, iou_threshold=threshold)
            totals[threshold][0] += scores.true_positives
            totals[threshold][1] += scores.false_positives
            totals[threshold][2] += scores.false_negatives

    row = {"failures": failures,
           "s_per_board": round(float(np.mean(runtimes)), 3) if runtimes else np.nan}
    for threshold, (tp, fp, fn) in totals.items():
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        suffix = f"{int(threshold * 100):02d}"
        row[f"precision_{suffix}"] = round(precision, 4)
        row[f"recall_{suffix}"] = round(recall, 4)
        row[f"f1_{suffix}"] = round(f1, 4)
    return row


def run(dataset: str, stages: list[str] | None = None) -> pd.DataFrame:
    if dataset == "deeppcb":
        # Parameters are chosen on trainval and reported on test. Tuning and
        # reporting on the same images overstates performance, and it is the
        # easiest methodological criticism to make of a project like this.
        split = os.environ.get("DEEPPCB_SPLIT", "test")
        pairs = ingest.index_deeppcb(split=split, limit=SAMPLE[dataset])
    else:
        everything = ingest.index_hripcb()
        step = max(1, len(everything) // SAMPLE[dataset])
        pairs = everything[::step][:SAMPLE[dataset]]

    baseline = BASELINES[dataset]
    stages = stages or list(SWEEPS)
    print(f"{dataset}: {len(pairs)} pairs, stages {stages}")

    started_baseline = time.perf_counter()
    baseline_row = {"stage": "baseline", "setting": "baseline",
                    **evaluate_config(pairs, baseline, dataset)}
    rows = [baseline_row]
    print(f"  baseline  F1@0.5 = {baseline_row['f1_50']:.4f} "
          f"({time.perf_counter() - started_baseline:.0f}s)")

    for stage in stages:
        for option in SWEEPS[stage]:
            if baseline.get(stage) == option:
                row = dict(baseline_row); row.update({"stage": stage, "setting": option})
                rows.append(row)
                continue
            params = dict(baseline)
            params[stage] = option
            started = time.perf_counter()
            row = {"stage": stage, "setting": option,
                   **evaluate_config(pairs, params, dataset)}
            rows.append(row)
            print(f"  {stage:<18} {str(option):<16} F1@0.5 = {row['f1_50']:.4f} "
                  f"({time.perf_counter() - started:.0f}s)")

    frame = pd.DataFrame(rows)
    frame.insert(0, "dataset", dataset)
    frame.insert(1, "split", os.environ.get("DEEPPCB_SPLIT", "test")
                 if dataset == "deeppcb" else "all")
    return frame


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "deeppcb"
    stages = sys.argv[2:] or None
    frame = run(dataset, stages)

    # Appending rather than overwriting lets the sweep be run a stage at a time
    # and still produce one table. Duplicate rows from a rerun are dropped,
    # keeping the most recent measurement of each configuration.
    path = OUTPUT_DIR / f"pipeline_benchmark_{dataset}.csv"
    if path.exists():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
        frame = frame.drop_duplicates(subset=["dataset", "split", "stage", "setting"], keep="last")
    save_table(frame, path.name)


if __name__ == "__main__":
    main()
