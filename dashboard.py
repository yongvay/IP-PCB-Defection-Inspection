import streamlit as st
import cv2
import numpy as np
import pandas as pd

from classifier import process_blobs, calculate_board_ssim
from pdf_report import generate_pdf_report

# Page Configuration
st.set_page_config(
    page_title="PCB Defect Inspection Dashboard",
    page_icon="🔍",
    layout="wide"
)

st.title("🛡️ Automated PCB Defect Inspection Dashboard")
st.caption("Mode B: Innovative Solution Development | Module 3: Defect Classification, Measurement & Analysis")

# Sidebar Controls
st.sidebar.header("⚙️ Inspection Parameters")
tolerance = st.sidebar.slider("Defect Tolerance (Max Allowed)", 0, 10, 0)
mm_per_px = st.sidebar.number_input("Spatial Scale (mm/px)", value=0.020, step=0.001, format="%.4f")
min_area_filter = st.sidebar.slider("Noise Area Filter (px)", 5, 100, 15)

use_mock_data = st.sidebar.checkbox("Use Mock Data Mode (Standalone Test)", value=True)

# Main Inspection Area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Reference Board (Golden Template)")
    template_file = st.file_uploader("Upload Template Image", type=["jpg", "png", "jpeg"], key="template")

with col2:
    st.subheader("Test Board (Inspected Unit)")
    test_file = st.file_uploader("Upload Test Image", type=["jpg", "png", "jpeg"], key="test")

st.divider()

# Load images or generate synthetic fallback boards
template_img = None
test_img = None

if template_file and test_file:
    file_bytes_temp = np.asarray(bytearray(template_file.read()), dtype=np.uint8)
    file_bytes_test = np.asarray(bytearray(test_file.read()), dtype=np.uint8)
    template_img = cv2.imdecode(file_bytes_temp, cv2.IMREAD_COLOR)
    test_img = cv2.imdecode(file_bytes_test, cv2.IMREAD_COLOR)
elif use_mock_data:
    # Generate synthetic 640x640 PCB board images for standalone demonstration
    template_img = np.ones((640, 640, 3), dtype=np.uint8) * 230
    cv2.rectangle(template_img, (100, 100), (540, 540), (180, 180, 180), -1)
    # Draw traces
    cv2.line(template_img, (150, 200), (450, 200), (0, 140, 0), 12)
    cv2.line(template_img, (150, 350), (450, 350), (0, 140, 0), 12)
    cv2.circle(template_img, (250, 200), 20, (0, 100, 0), -1)

    test_img = template_img.copy()
    # Introduce synthetic defects: 1. Open circuit (removed), 2. Spurious copper (added)
    cv2.rectangle(test_img, (280, 194), (330, 206), (180, 180, 180), -1)  # Broken trace
    cv2.circle(test_img, (250, 450), 15, (0, 140, 0), -1)  # Extra copper spot

# Mock Blobs Data Contract matching Module 2 output
mock_blobs = [
    {
        "id": 1,
        "bbox": (280, 194, 50, 12),
        "contour": np.array([[[280, 194]], [[330, 194]], [[330, 206]], [[280, 206]]]),
        "area_px": 600,
        "polarity": "removed"
    },
    {
        "id": 2,
        "bbox": (235, 435, 30, 30),
        "contour": np.array([[[235, 435]], [[265, 435]], [[265, 465]], [[235, 465]]]),
        "area_px": 700,
        "polarity": "added"
    }
]

if template_img is not None and test_img is not None:
    # Calculate SSIM (Practical 9)[cite: 4]
    ssim_score = calculate_board_ssim(template_img, test_img)
    
    # Process Blobs (Task 3.1, 3.2, 3.3, 3.4)[cite: 3]
    defects, verdict = process_blobs(mock_blobs, mm_per_px, tolerance, min_area_filter)

    # Verdict Banner
    if verdict == "PASS":
        st.success(f"### 🎉 Inspection Verdict: {verdict} | Total Defects: {len(defects)} (Tolerance: {tolerance})")
    else:
        st.error(f"### ⚠️ Inspection Verdict: {verdict} | Total Defects: {len(defects)} (Tolerance: {tolerance})")

    # Metrics Summary Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Defects", len(defects))
    m2.metric("Board SSIM Score", f"{ssim_score}")
    m3.metric("Scale Factor", f"{mm_per_px} mm/px")
    m4.metric("Status", verdict)

    st.spacer = st.empty()

    # Draw Annotated Overlays with Rotated Bounding Boxes (Practical 1 & 8)[cite: 4]
    overlay_img = test_img.copy()
    for d in defects:
        rect = d["rect"]
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Red for removed copper, Blue for added copper
        color = (0, 0, 255) if d["Polarity"] == "removed" else (255, 0, 0)
        cv2.drawContours(overlay_img, [box], 0, color, 2, lineType=cv2.LINE_AA)

        cx, cy = int(rect[0][0]), int(rect[0][1])
        cv2.putText(
            overlay_img,
            f"#{d['ID']}: {d['Class']}",
            (cx - 30, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
            lineType=cv2.LINE_AA
        )
        cv2.putText(
            overlay_img,
            f"#{d['ID']}: {d['Class']}",
            (cx - 30, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA
        )

    # Display Visual Comparison
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.subheader("Reference Template Board")
        st.image(template_img, channels="BGR", use_container_width=True)

    with img_col2:
        st.subheader("Annotated Inspection Result")
        st.image(overlay_img, channels="BGR", use_container_width=True)

    st.divider()

    # Sortable Defect Log Table & Analytics
    tbl_col, chart_col = st.columns([6, 4])

    with tbl_col:
        st.subheader("📋 Sortable Defect Log")
        if defects:
            df_defects = pd.DataFrame(defects)
            # Remove complex object columns for table display
            df_display = df_defects.drop(columns=["bbox", "rect", "contour"])
            st.dataframe(df_display, use_container_width=True)
        else:
            st.info("No defects present on this board.")

    with chart_col:
        st.subheader("📊 Defect Class Distribution")
        if defects:
            class_counts = df_defects["Class"].value_counts()
            st.bar_chart(class_counts)
        else:
            st.info("Distribution clear.")

    st.divider()

    # Automated PDF Report Download (Extra Effort Feature)[cite: 1, 3]
    st.subheader("📄 Export Inspection Report")
    pdf_bytes = generate_pdf_report(defects, verdict, ssim_score, mm_per_px, tolerance)
    
    st.download_button(
        label="📥 Download Official Inspection Report (PDF)",
        data=pdf_bytes,
        file_name=f"PCB_Inspection_Report_{verdict}.pdf",
        mime="application/pdf"
    )

else:
    st.info("Please upload both a Template and Test image or enable 'Use Mock Data Mode' in the sidebar.")