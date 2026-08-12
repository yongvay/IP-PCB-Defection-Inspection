# Automated PCB Defect Inspection System

BMDS2133 Image Processing — Mode B, Innovative Solution Development.
Tunku Abdul Rahman University of Management and Technology.

A golden-template inspection system that localises and classifies the six
standard bare-board PCB defect classes by comparing a board under test against
a defect-free reference, using classical image processing only. No machine
learning is used anywhere in the pipeline.

| Member | Module | Cover-page label |
| --- | --- | --- |
| Ng Yong Vay (leader) | Module 2 — Registration, defect localisation, integration | Image Registration & Defect Localisation Module |
| Chan Xing Szen | Module 1 — Acquisition, preprocessing, calibration | Preprocessing & Calibration Module |
| Ng Zhi Xuan | Module 3 — Classification, measurement, dashboard | Defect Classification & Analysis Dashboard Module |

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

The dataset is not committed. Download DeepPCB from
<https://github.com/tangsanli5201/DeepPCB> and place it so that the layout is:

```
data/DeepPCB/PCBData/group00041/00041/00041000_temp.jpg
data/DeepPCB/PCBData/group00041/00041_not/00041000.txt
```

Inspect a single pair:

```bash
python -m src.pipeline data/DeepPCB/PCBData/group00041/00041/00041000_temp.jpg \
                       data/DeepPCB/PCBData/group00041/00041/00041000_test.jpg
```

Run the tests:

```bash
python -m pytest tests -v
```

## Repository layout

```
dashboard.py              Streamlit entry point: streamlit run dashboard.py
src/
  contracts.py            frozen interfaces between the three modules
  pipeline.py             the orchestrator: Module 1 -> Module 2 -> Module 3
  module1/                Chan Xing Szen
    ground_truth.py         DeepPCB annotation parser          (done)
    preprocess.py           ingestion, denoise, enhance, binarise, calibrate
    morphology.py           shared cleanup utility called by the orchestrator
  module2/                Ng Yong Vay
    registration.py         ORB/SIFT + RANSAC, phase-correlation fallback (done)
    difference.py           signed golden-template differencing (done)
    blobs.py                connected-component blob extraction (done)
  module3/                Ng Zhi Xuan
    descriptors.py          region descriptors and the two-stage rules (done)
    classify.py             adapter: Blob -> descriptors.py -> Defect (done)
    pdf_report.py           PDF inspection report export       (done)
    evaluate.py             IoU matching, precision/recall/F1
tests/                    integration and regression tests
notebooks/                exploratory prototypes, not part of the pipeline
docs/                     meeting log, working notes
data/                     datasets (git-ignored, never committed)
```

Files marked *(done)* are implemented. The rest are working stubs with fixed
signatures, so the pipeline runs end to end today and each owner can replace
their stub without coordinating with anyone else.

Two conventions worth stating, because they are easy to break by accident:

- **Everything the dashboard displays comes from `inspect_pair`.** The
  dashboard computes no image processing of its own, so what appears on screen
  is the same result the evaluation harness scores.
- **`descriptors.py` returns display labels; `classify.py` returns contract
  labels.** The mapping between the two lives in `DISPLAY_TO_CONTRACT` and
  nowhere else. Evaluation compares contract labels only.

## Running the dashboard

```bash
streamlit run dashboard.py
```

Run it from the repository root so that `src` is importable. If the DeepPCB
folder is present, the sidebar offers sample pairs directly; otherwise upload a
template and a test image.

## Current baseline

Measured over 88 DeepPCB pairs with **stub preprocessing** and
**localisation-only** scoring (bounding boxes, class labels not yet assessed):

| Metric | IoU 0.33 | IoU 0.50 |
| --- | --- | --- |
| Precision | 0.839 | 0.823 |
| Recall | 0.755 | 0.741 |
| **F1** | **0.795** | **0.780** |

Mean runtime 0.153 s per board against Objective 3's 3 s budget.

SMART Objective 1 targets F1 ≥ 0.80 at IoU 0.5. The classifier is still a
placeholder, so this figure will fall once class labels are scored, and rise
again as Module 1's preprocessing replaces the stub.

## Four findings that the pipeline depends on

These were established by measurement during integration and are each locked
in by a test. Reverting any of them silently breaks the results.

1. **Copper is dark in DeepPCB, not white.** Roughly 86% of every image is
   white substrate. Assuming copper is white inverts every polarity label,
   which inverts stage one of the classifier while leaving the bounding boxes
   looking perfectly correct. See `src/module2/difference.py`.

2. **DeepPCB bounding boxes are padded by about 10 pixels.** The annotations
   mark a region around each defect, not the changed pixels. Scoring tight
   predictions against padded truth gives recall 0.01 at IoU 0.5; applying the
   same convention to the predictions gives 0.71 from identical detections.
   This must be stated explicitly in the report. See `src/module2/blobs.py`.

3. **Morphological opening is the highest-leverage parameter.** A 5 x 5
   elliptical element raises precision from 0.10 to 0.93, because binarisation
   jitter produces differences that are long and one pixel wide, whereas real
   defects are compact. See `src/module1/morphology.py`.

4. **Registration must be conditional.** DeepPCB pairs are already aligned, and
   fitting a homography anyway costs 0.11 of F1. Each candidate transform,
   including the identity, is scored by the fraction of pixels that still
   disagree after warping, and the best one wins. See
   `src/module2/registration.py`.

## Working agreement

- Nobody changes an interface in `src/contracts.py` unilaterally. Raise it at
  the weekly checkpoint; the leader updates the file.
- Definition of done: merged to `main`, runs from the orchestrator with no
  manual steps, has a test or a documented sample input and output, and the
  corresponding report paragraph is drafted.
- Datasets are never committed. They are cited in Section 3.2 of the report.
- Each member writes their own module. Understanding of code is assessed
  individually and live.

## Datasets

Tang, S., He, F., Huang, X., & Yang, J. (2019). *Online PCB defect detector on
a new PCB defect dataset*. <https://github.com/tangsanli5201/DeepPCB>

Huang, W., & Wei, P. (2020). HRIPCB: A challenging dataset for PCB defects
detection and classification. *The Journal of Engineering, 2020*(13), 303–309.
<https://doi.org/10.1049/joe.2019.1183>
