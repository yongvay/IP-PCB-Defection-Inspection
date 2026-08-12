"""Streamlit inspection dashboard — tasks 3.5 and 3.6.

Owner: Ng Zhi Xuan.

Run from the repository root so that the ``src`` package is importable:

    streamlit run dashboard.py

This is the application entry point only. Every algorithm it displays lives in
``src/``; nothing is computed here that the pipeline does not already compute,
so what the marker sees on screen is the same result the evaluation harness
scores. That is the whole reason the mock-data path is no longer the default.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.module1.ground_truth import POLARITY
from src.module1.preprocess import DEEPPCB_PIXELS_PER_MM
from src.module3.descriptors import calculate_board_ssim
from src.module3.pdf_report import generate_pdf_report
from src.pipeline import inspect_pair

REPO_ROOT = Path(__file__).resolve().parent
PCB_DATA = REPO_ROOT / "data" / "DeepPCB" / "PCBData"

# Copper removed is drawn red, copper added blue. The polarity is not carried
# on the Defect contract, because it is fully determined by the class, so it is
# recovered from the same table the ground-truth parser uses rather than being
# duplicated as a field.
POLARITY_COLOUR = {"removed": (0, 0, 255), "added": (255, 0, 0)}

st.set_page_config(
    page_title="PCB Defect Inspection Dashboard",
    page_icon="🔍",
    layout="wide",
)

st.title("Automated PCB Defect Inspection Dashboard")
st.caption(
    "BMDS2133 Image Processing · Mode B · "
    "Module 3: Defect Classification, Measurement & Analysis"
)


# --------------------------------------------------------------------------
# Input selection
# --------------------------------------------------------------------------
def list_sample_pairs(limit: int = 40) -> list[tuple[str, Path, Path]]:
    """Find template-test pairs already present in the DeepPCB folder."""
    if not PCB_DATA.exists():
        return []
    pairs = []
    for template in sorted(PCB_DATA.glob("*/*/*_temp.jpg"))[:limit]:
        test = template.with_name(template.name.replace("_temp.jpg", "_test.jpg"))
        if test.exists():
            pairs.append((template.stem.replace("_temp", ""), template, test))
    return pairs


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

tolerance = st.sidebar.slider("Defect tolerance (maximum allowed)", 0, 10, 0)
open_kernel = st.sidebar.slider("Morphological opening kernel (px)", 1, 11, 5, step=2)
min_area = st.sidebar.slider("Minimum blob area (px)", 5, 200, 40, step=5)

st.sidebar.caption(
    "Opening at 5 x 5 is the verified default. Lowering it to 3 x 3 drops "
    "precision from 0.93 to 0.28 — see README, finding 3."
)

template_path = test_path = None

if source == "Sample pair from DeepPCB":
    labels = [name for name, _, _ in samples]
    chosen = st.sidebar.selectbox("Board", labels)
    for name, template, test in samples:
        if name == chosen:
            template_path, test_path = str(template), str(test)
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

st.divider()


# --------------------------------------------------------------------------
# Inspection
# --------------------------------------------------------------------------
if not (template_path and test_path):
    st.info(
        "Select a sample pair, or upload both a template and a test image, "
        "to run an inspection."
    )
    st.stop()

params = {
    "max_defects": tolerance,
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
headline = (
    f"Inspection verdict: {report.verdict} · "
    f"{len(report.defects)} defect(s) · tolerance {tolerance}"
)
(st.success if report.verdict == "PASS" else st.error)(f"### {headline}")

metric_cols = st.columns(5)
metric_cols[0].metric("Total defects", len(report.defects))
metric_cols[1].metric("Board SSIM", f"{ssim_score:.4f}")
metric_cols[2].metric("Alignment residual", f"{report.align_residual:.3f} px")
metric_cols[3].metric("Runtime", f"{report.runtime_s:.3f} s")
metric_cols[4].metric("Status", report.verdict)

if report.runtime_s > 3.0:
    st.warning(
        "Runtime exceeds the 3 s per board budget set by SMART Objective 3."
    )

# --- Annotated overlay ----------------------------------------------------
overlay = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
for defect in report.defects:
    x, y, w, h = defect.bbox
    polarity = POLARITY[defect.defect_class]
    colour = POLARITY_COLOUR[polarity]

    cv2.rectangle(overlay, (x, y), (x + w, y + h), colour, 2, lineType=cv2.LINE_AA)
    label = f"#{defect.id} {defect.defect_class}"
    # Drawn twice: a thick dark pass then a thin light pass, so the text stays
    # readable over both copper and substrate.
    for thickness, text_colour in ((3, (0, 0, 0)), (1, (255, 255, 255))):
        cv2.putText(
            overlay, label, (x, max(12, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_colour, thickness,
            lineType=cv2.LINE_AA,
        )

image_left, image_right = st.columns(2)
with image_left:
    st.subheader("Reference template")
    st.image(template_img, use_container_width=True)
with image_right:
    st.subheader("Annotated inspection result")
    st.image(overlay, channels="BGR", use_container_width=True)

st.caption("Red: copper removed.  Blue: copper added.")

st.divider()

# --- Defect table and distribution ---------------------------------------
table_col, chart_col = st.columns([6, 4])

rows = [
    {
        "ID": defect.id,
        "Class": defect.defect_class,
        "Polarity": POLARITY[defect.defect_class],
        "Area (mm²)": round(defect.area_mm2, 4),
        "x": defect.bbox[0],
        "y": defect.bbox[1],
        "Width (px)": defect.bbox[2],
        "Height (px)": defect.bbox[3],
    }
    for defect in report.defects
]

with table_col:
    st.subheader("Defect log")
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No defects detected on this board.")

with chart_col:
    st.subheader("Defect class distribution")
    if rows:
        st.bar_chart(pd.DataFrame(rows)["Class"].value_counts())
    else:
        st.info("Nothing to plot.")

st.divider()

# --- Report export --------------------------------------------------------
st.subheader("Export inspection report")

# Module 1's calibration factor is applied inside the pipeline and is not
# carried on the report, so the same constant is used here for the linear
# dimensions. Task 1.7 replaces the constant with a derived factor, at which
# point exposing it on InspectionReport is a contract amendment to raise at the
# weekly checkpoint.
MM_PER_PX = 1.0 / DEEPPCB_PIXELS_PER_MM

pdf_defects = [
    {
        "ID": row["ID"],
        "Class": row["Class"],
        "Polarity": row["Polarity"],
        "Area (mm²)": row["Area (mm²)"],
        # Bounding-box extent, not the rotated minimum-area box: the contract
        # carries the axis-aligned box only.
        "Width (mm)": round(row["Width (px)"] * MM_PER_PX, 3),
        "Height (mm)": round(row["Height (px)"] * MM_PER_PX, 3),
    }
    for row in rows
]

st.download_button(
    label="Download inspection report (PDF)",
    data=generate_pdf_report(
        pdf_defects, report.verdict, ssim_score, MM_PER_PX, tolerance
    ),
    file_name=f"PCB_Inspection_Report_{report.verdict}.pdf",
    mime="application/pdf",
)
