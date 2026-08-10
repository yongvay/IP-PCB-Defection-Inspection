import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def extract_descriptors(blob: dict) -> dict:
    """
    Extracts geometric region descriptors from a blob contour.
    Utilises cv2.minAreaRect for rotated bounding box accuracy (Practical 8)[cite: 4].
    """
    contour = blob["contour"]
    area_px = blob["area_px"]

    # Minimum-area rotated bounding box (Practical 8)[cite: 4]
    rect = cv2.minAreaRect(contour)  # ((cx, cy), (width, height), angle)
    (cx, cy), (w, h), angle = rect

    # Calculate oriented dimensions and aspect ratio
    major_axis = max(w, h)
    minor_axis = min(w, h) if min(w, h) > 0 else 1.0
    aspect_ratio = float(major_axis) / minor_axis

    # Extent: ratio of contour area to bounding box area
    bounding_box_area = w * h if (w * h) > 0 else 1.0
    extent = float(area_px) / bounding_box_area

    # Convex Hull & Solidity: ratio of contour area to convex hull area
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    hull_area = hull_area if hull_area > 0 else 1.0
    solidity = float(area_px) / hull_area

    # Image Moments & Hu Moments for shape invariance
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()

    return {
        "area_px": area_px,
        "rect": rect,
        "width_px": major_axis,
        "height_px": minor_axis,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
        "hu_moments": hu_moments,
    }


def classify_defect(polarity: str, aspect_ratio: float, solidity: float) -> str:
    """
    Two-stage rule-based classifier categorising defects into the six standard classes[cite: 3]:
    Stage 1: Polarity split (copper-removed vs copper-added)[cite: 3].
    Stage 2: Geometric descriptor rules[cite: 3].
    """
    if polarity == "removed":
        if aspect_ratio > 2.5:
            return "Open circuit"
        elif solidity > 0.85 and (0.7 <= aspect_ratio <= 1.3):
            return "Missing hole/pin-hole"
        else:
            return "Mouse bite"
    else:  # polarity == "added"
        if aspect_ratio > 2.5:
            return "Short"
        elif solidity > 0.85 and (0.7 <= aspect_ratio <= 1.3):
            return "Spurious copper"
        else:
            return "Spur"


def calculate_board_ssim(template_img: np.ndarray, test_img: np.ndarray) -> float:
    """
    Calculates the global Structural Similarity Index (SSIM) between template and test board (Practical 9)[cite: 4].
    """
    gray_temp = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY) if len(template_img.shape) == 3 else template_img
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY) if len(test_img.shape) == 3 else test_img

    # Ensure identical dimensions for SSIM calculation
    if gray_temp.shape != gray_test.shape:
        gray_test = cv2.resize(gray_test, (gray_temp.shape[1], gray_temp.shape[0]))

    score, _ = ssim(gray_temp, gray_test, full=True)
    return float(round(score, 4))


def process_blobs(blobs: list, mm_per_px: float, tolerance: int, min_area_px: int = 15) -> tuple:
    """
    Processes raw blobs, filters noise, calculates physical measurements in mm/mm²,
    classifies defects, and outputs the board inspection verdict[cite: 3].
    """
    processed_defects = []

    for blob in blobs:
        # Filter noise below minimum pixel area threshold (Practical 8)[cite: 4]
        if blob["area_px"] < min_area_px:
            continue

        desc = extract_descriptors(blob)
        defect_type = classify_defect(blob["polarity"], desc["aspect_ratio"], desc["solidity"])

        # Physical measurement conversion (Task 3.3)[cite: 3]
        area_mm2 = desc["area_px"] * (mm_per_px ** 2)
        width_mm = desc["width_px"] * mm_per_px
        height_mm = desc["height_px"] * mm_per_px

        processed_defects.append({
            "ID": blob["id"],
            "Class": defect_type,
            "Polarity": blob["polarity"],
            "Area (mm²)": round(area_mm2, 4),
            "Width (mm)": round(width_mm, 3),
            "Height (mm)": round(height_mm, 3),
            "bbox": blob["bbox"],
            "rect": desc["rect"],
            "contour": blob["contour"]
        })

    # Board verdict decision (Task 3.4)[cite: 3]
    verdict = "PASS" if len(processed_defects) <= tolerance else "FAIL"

    return processed_defects, verdict