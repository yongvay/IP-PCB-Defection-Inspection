"""Streamlit inspection dashboard — tasks 3.5 and 3.6.

Owner: Ng Zhi Xuan (Module 3).

Run from the repository root so that the ``src`` package is importable:

    streamlit run dashboard.py

This is the application entry point only. Every algorithm it displays lives in
``src/``; nothing is computed here that the pipeline does not already compute,
so what the marker sees on screen is the same result the evaluation harness
scores. The only exception is the annotation drawing, which is presentation
rather than analysis.

Three things the dashboard is expected to make visible, and does
---------------------------------------------------------------
1. **Why a board failed**, not merely that it did. The verdict carries the
   condition that produced it.
2. **Which classifier decided**, and what the alternative would have said. The
   descriptor baseline and the connectivity classifier can be switched live,
   which is the comparison Chapter 4 rests on.
3. **How the result scores**, when an answer key exists. Selecting a DeepPCB
   sample pair loads its annotations and scores the run in place, so the
   accuracy claim is demonstrable rather than merely asserted.
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.module1.ground_truth import parse_annotation_file
from src.module3.classify import CONTRACT_TO_DISPLAY
from src.module3.descriptors import calculate_board_ssim
from src.module3.evaluate import Evaluation
from src.module3.pdf_report import SEVERITY_WORD, generate_pdf_report
from src.module2.blobs import to_evaluation_bbox
from src.pipeline import inspect_pair

REPO_ROOT = Path(__file__).resolve().parent
PCB_DATA = REPO_ROOT / "data" / "DeepPCB" / "PCBData"

# Copper removed is drawn red, copper added blue. Polarity travels on the
# Defect contract, so it is read from the result rather than looked up in
# another member's module.
POLARITY_COLOUR = {"removed": (0, 0, 255), "added": (255, 0, 0)}
GROUND_TRUTH_COLOUR = (0, 200, 0)

# Measured margin between a DeepPCB annotation box and the changed pixels it
# surrounds. Applied to predictions only when scoring, never when measuring a
# defect physically. See src/module2/blobs.py and Chapter 3.
DEEPPCB_PADDING_PX = 10

# Streamlit renamed the image-width argument between the pinned version and
# current releases. Resolving it once here means the dashboard runs on either
# without a deprecation warning or a crash.
_WIDTH_KWARG = (
    "use_container_width"
    if "use_container_width" in inspect.signature(st.image).parameters
    else "use_column_width"
)

st.set_page_config(
    page_title="PCB Defect Inspection Dashboard",
    page_icon="\U0001F50D",
    layout="wide",
)


def show_image(image: np.ndarray, channels: str = "BGR") -> None:
    st.image(image, channels=channels, **{_WIDTH_KWARG: True})


st.title("Automated PCB Defect Inspection Dashboard")
st.caption(
    "BMDS2133 Image Processing · Mode B · "
    "Module 3: Defect Classification, Measurement & Analysis"
)


# ---------------------------------------------------------------------------
# Input selection
# ---------------------------------------------------------------------------
def list_sample_pairs(limit: int = 60) -> list[tuple[str, Path, Path]]:
    """Find template-test pairs already present in the DeepPCB folder."""
    if not PCB_DATA.exists():
        return []
    pairs = []
    for template in sorted(PCB_DATA.glob("*/*/*_temp.jpg"))[:limit]:
        test = template.with_name(template.name.replace("_temp.jpg", "_test.jpg"))
        if test.exists():
            pairs.append((template.stem.replace("_temp", ""), template, test))
    return pairs


def annotation_for(template_path: Path) -> Path | None:
    """Locate the DeepPCB answer key that belongs to a template image.

    The dataset stores annotations in a sibling folder suffixed ``_not``, so
    ``group00041/00041/00041000_temp.jpg`` is answered by
    ``group00041/00041_not/00041000.txt``.
    """
    candidate = (template_path.parent.parent
                 / f"{template_path.parent.name}_not"
                 / f"{template_path.stem.replace('_temp', '')}.txt")
    return candidate if candidate.exists() else None


def save_upload(upload) -> str:
    """Persist an uploaded image to a temporary file.

    The pipeline is addressed by path rather than by array so that a run from
    the dashboard and a run from the command line take exactly the same code
    path, including the image loading in Module 1.
    """
    suffix = Path(upload.name).suffix or ".jpg"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(upload.getbuffer())
    handle.close()
    return handle.name


st.sidebar.header("Inspection parameters")

samples = list_sample_pairs()
source_options = ["Upload a pair"]
if samples:
    source_options.insert(0, "Sample pair from DeepPCB")
source = st.sidebar.radio("Image source", source_options)

classifier = st.sidebar.selectbox(
    "Stage-two classifier",
    ["connectivity", "descriptor"],
    help=(
        "connectivity reads how each region sits against the surrounding "
        "copper; descriptor is the shape-only baseline retained for comparison."
    ),
)

st.sidebar.subheader("Acceptance criteria")
tolerance = st.sidebar.slider("Defect count tolerance", 0, 10, 0)
max_severity = st.sidebar.slider("Total severity weight allowed", 0, 20, 0)
st.sidebar.caption(
    "A critical defect — an open circuit or a short — fails the board on its "
    "own, whatever these tolerances allow."
)

st.sidebar.subheader("Pipeline settings")
open_kernel = st.sidebar.slider("Morphological opening kernel (px)", 1, 11, 5, step=2)
min_area = st.sidebar.slider("Minimum blob area (px)", 5, 200, 40, step=5)
st.sidebar.caption(
    "Opening at 5 x 5 is the verified default. Lowering it to 3 x 3 drops "
    "precision from 0.93 to 0.28 — see README, finding 3."
)

template_path = test_path = None
board_name = ""
truth_boxes: list[dict] = []

if source == "Sample pair from DeepPCB":
    labels = [name for name, _, _ in samples]
    chosen = st.sidebar.selectbox("Board", labels)
    for name, template, test in samples:
        if name == chosen:
            template_path, test_path, board_name = str(template), str(test), name
            annotation = annotation_for(template)
            if annotation is not None:
                truth_boxes = parse_annotation_file(annotation)
            break
else:
    upload_left, upload_right = st.columns(2)
    with upload_left:
        st.subheader("Reference board (golden template)")
        template_file = st.file_uploader(
            "Template image", type=["jpg", "jpeg", "png"], key="template"
        )
    with upload_right:
        st.subheader("Test board (inspected unit)")
        test_file = st.file_uploader(
            "Test image", type=["jpg", "jpeg", "png"], key="test"
        )
    if template_file and test_file:
        template_path = save_upload(template_file)
        test_path = save_upload(test_file)
        board_name = Path(test_file.name).stem

st.divider()

if not (template_path and test_path):
    st.info(
        "Select a sample pair, or upload both a template and a test image, "
        "to run an inspection."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------
params = {
    "classifier": classifier,
    "max_defects": tolerance,
    "max_severity": max_severity,
    "morph_open_kernel": open_kernel,
    "morph_close_kernel": open_kernel,
    "min_blob_area": min_area,
}

with st.spinner("Running the inspection pipeline…"):
    report = inspect_pair(template_path, test_path, params)

template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
ssim_score = calculate_board_ssim(template_img, test_img)

# --- Verdict banner -------------------------------------------------------
banner = st.success if report.verdict == "PASS" else st.error
banner(f"### Board {report.verdict}\n{report.verdict_reason}")

metrics = st.columns(6)
metrics[0].metric("Defects", len(report.defects))
metrics[1].metric("Severity weight", report.verdict_detail.get("total_severity", 0))
metrics[2].metric("Critical", report.verdict_detail.get("critical_count", 0))
metrics[3].metric("Board SSIM", f"{ssim_score:.4f}")
metrics[4].metric("Alignment residual", f"{report.align_residual:.3f} px")
metrics[5].metric("Runtime", f"{report.runtime_s:.3f} s")

if report.runtime_s > 3.0:
    st.warning(
        "Runtime exceeds the three seconds per board set by SMART Objective 3."
    )
if report.align_residual > 5.0:
    st.warning(
        "Alignment residual is high, so the difference image may contain "
        "registration artefacts rather than genuine defects."
    )


# --- Annotated overlay ----------------------------------------------------
def annotate(base: np.ndarray, show_truth: bool) -> np.ndarray:
    """Draw the detections, and optionally the answer key, over the test board."""
    canvas = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    if show_truth:
        for box in truth_boxes:
            x, y, w, h = box["bbox"]
            cv2.rectangle(canvas, (x, y), (x + w, y + h),
                          GROUND_TRUTH_COLOUR, 1, lineType=cv2.LINE_AA)

    for defect in report.defects:
        x, y, w, h = defect.bbox
        colour = POLARITY_COLOUR.get(str(defect.polarity), (0, 0, 255))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, 2, lineType=cv2.LINE_AA)

        label = f"#{defect.id} {CONTRACT_TO_DISPLAY.get(defect.defect_class, defect.defect_class)}"
        # Drawn twice, a thick dark pass then a thin light pass, so the text
        # stays readable over both copper and substrate.
        for thickness, text_colour in ((3, (0, 0, 0)), (1, (255, 255, 255))):
            cv2.putText(canvas, label, (x, max(12, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_colour, thickness,
                        lineType=cv2.LINE_AA)
    return canvas


show_truth = bool(truth_boxes) and st.checkbox(
    "Overlay the ground-truth annotations", value=bool(truth_boxes)
)
overlay = annotate(test_img, show_truth)

image_left, image_right = st.columns(2)
with image_left:
    st.subheader("Reference template")
    show_image(template_img, channels="GRAY")
with image_right:
    st.subheader("Annotated inspection result")
    show_image(overlay)

legend = "Red: copper removed.  Blue: copper added."
if show_truth:
    legend += "  Green: ground-truth annotation."
st.caption(legend)

st.divider()


# ---------------------------------------------------------------------------
# Summary panels — task 3.6
# ---------------------------------------------------------------------------
rows = [
    {
        "ID": defect.id,
        "Class": CONTRACT_TO_DISPLAY.get(defect.defect_class, defect.defect_class),
        "Polarity": str(defect.polarity).capitalize(),
        "Severity": SEVERITY_WORD.get(defect.severity, "Minor"),
        "Length (mm)": round(defect.width_mm, 3),
        "Width (mm)": round(defect.height_mm, 3),
        "Area (mm²)": round(defect.area_mm2, 4),
        "Confidence": defect.confidence,
        "Decided by": defect.decided_by,
        "x": defect.bbox[0],
        "y": defect.bbox[1],
    }
    for defect in report.defects
]
frame = pd.DataFrame(rows)

table_col, chart_col = st.columns([6, 4])

with table_col:
    st.subheader("Defect log")
    if rows:
        st.dataframe(
            frame.sort_values("Area (mm²)", ascending=False),
            hide_index=True,
            **{"use_container_width": True},
        )
        st.caption(
            "Sort any column by clicking its header. Length and width come "
            "from the rotated minimum-area rectangle, so a diagonal defect is "
            "measured along its own axis rather than along the image axes."
        )
    else:
        st.info("No defects detected on this board.")

with chart_col:
    st.subheader("Defects by class")
    if rows:
        st.bar_chart(frame["Class"].value_counts())
        st.subheader("Defective area distribution")
        st.bar_chart(
            frame.groupby("Class")["Area (mm²)"].sum().sort_values(ascending=False)
        )
    else:
        st.info("Nothing to plot.")

if rows:
    with st.expander("Severity breakdown and classifier provenance"):
        left, right = st.columns(2)
        with left:
            st.write("**Defects by severity**")
            st.dataframe(
                frame["Severity"].value_counts().rename("Count"),
                **{"use_container_width": True},
            )
        with right:
            st.write("**Which rule set decided**")
            st.dataframe(
                frame["Decided by"].value_counts().rename("Count"),
                **{"use_container_width": True},
            )
        st.caption(
            "The connectivity classifier declines to rule on a region whose "
            "copper context cannot be read and hands it to the descriptor "
            "baseline. A high fallback count means registration, not "
            "classification, is the limiting factor on this board."
        )

st.divider()


# ---------------------------------------------------------------------------
# Live scoring against the answer key — task 3.7
# ---------------------------------------------------------------------------
if truth_boxes:
    st.subheader("Scored against the DeepPCB answer key")

    evaluation = Evaluation(iou_threshold=0.5)
    evaluation.add_board(
        board_name,
        predictions=[
            (to_evaluation_bbox(defect.bbox, test_img.shape,
                                padding=DEEPPCB_PADDING_PX),
             defect.defect_class)
            for defect in report.defects
        ],
        truths=[(box["bbox"], box["label"]) for box in truth_boxes],
        runtime_s=report.runtime_s,
    )
    summary = evaluation.summary()

    score_cols = st.columns(5)
    score_cols[0].metric("Ground-truth defects", len(truth_boxes))
    score_cols[1].metric("Localised", evaluation.localisation.true_positives)
    score_cols[2].metric("Localisation F1", f"{summary['loc_f1']:.3f}")
    score_cols[3].metric("Classification F1", f"{summary['cls_f1']:.3f}")
    score_cols[4].metric("Class accuracy", f"{summary['class_accuracy']:.3f}")

    st.caption(
        f"Single board, IoU 0.5. Predicted boxes carry the "
        f"{DEEPPCB_PADDING_PX} px annotation margin the dataset uses, applied "
        "for scoring only and never to a physical measurement. One board is an "
        "illustration, not a result — Chapter 4 quotes the figures from "
        "experiments/benchmark_module3.py over the held-out test split."
    )

    misclassified = [
        row for row in evaluation.confusion_rows()
        if any(row[key] and key not in ("truth", row["truth"])
               for key in row if key != "truth")
    ]
    if misclassified:
        with st.expander("Confusion matrix for this board"):
            st.dataframe(
                pd.DataFrame(evaluation.confusion_rows()).set_index("truth"),
                **{"use_container_width": True},
            )

    st.divider()


# ---------------------------------------------------------------------------
# Report export
# ---------------------------------------------------------------------------
st.subheader("Export inspection report")

st.download_button(
    label="Download inspection report (PDF)",
    data=generate_pdf_report(
        report,
        ssim_score=ssim_score,
        board_name=board_name,
        annotated_image=annotate(test_img, show_truth=False),
        params=params,
    ),
    file_name=f"PCB_Inspection_{board_name or 'board'}_{report.verdict}.pdf",
    mime="application/pdf",
)

if rows:
    st.download_button(
        label="Download defect log (CSV)",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=f"PCB_Defects_{board_name or 'board'}.csv",
        mime="text/csv",
    )
