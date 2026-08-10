import streamlit as st
import cv2
import numpy as np
from classifier import process_blobs

# Page Config
st.set_page_config(page_title="PCB Defect Inspection", layout="wide")
st.title("Automated PCB Defect Inspection Dashboard")

# Sidebar Controls
st.sidebar.header("Inspection Controls")
tolerance = st.sidebar.slider("Defect Tolerance (Max allowed)", 0, 10, 0)
mm_per_px = st.sidebar.number_input("Calibration (mm per px)", value=0.02, step=0.005, format="%.3f")

# Mock blobs (simulating Module 2 output)
mock_blobs = [
    {
        "id": 1,
        "bbox": (150, 120, 80, 15),
        "contour": np.array([[[150, 120]], [[230, 120]], [[230, 135]], [[150, 135]]]),
        "area_px": 1200,
        "polarity": "removed"
    },
    {
        "id": 2,
        "bbox": (300, 300, 25, 25),
        "contour": np.array([[[300, 300]], [[325, 300]], [[325, 325]], [[300, 325]]]),
        "area_px": 625,
        "polarity": "added"
    }
]

# Process defects
defects, verdict = process_blobs(mock_blobs, mm_per_px, tolerance)

# Task 3.4 & 3.6: Banner & Metrics
if verdict == "PASS":
    st.success(f"### Verdict: {verdict} (Total Defects: {len(defects)} | Max Allowed: {tolerance})")
else:
    st.error(f"### Verdict: {verdict} (Total Defects: {len(defects)} | Max Allowed: {tolerance})")

m1, m2, m3 = st.columns(3)
m1.metric("Total Defects", len(defects))
m2.metric("Scale Factor", f"{mm_per_px} mm/px")
m3.metric("Status", verdict)

st.divider()

# Create dummy base image & draw annotated overlays
blank_image = np.ones((500, 500, 3), dtype=np.uint8) * 240
overlay_image = blank_image.copy()

for d in defects:
    x, y, w, h = d["bbox"]
    # Red for 'removed', Blue for 'added'
    color = (0, 0, 255) if d["Polarity"] == "removed" else (255, 0, 0)
    cv2.rectangle(overlay_image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(overlay_image, f"{d['ID']}: {d['Class']}", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Display Side-by-Side Images
col1, col2 = st.columns(2)
with col1:
    st.subheader("Raw Test Board")
    st.image(blank_image, caption="Uploaded Board Image", use_container_width=True)

with col2:
    st.subheader("Annotated Defect Overlay")
    st.image(overlay_image, caption="Detected Defects (Red=Removed, Blue=Added)", use_container_width=True)

st.divider()

# Task 3.6: Sortable Defect Table
st.subheader("Detailed Defect Log")
st.dataframe(defects, use_container_width=True)