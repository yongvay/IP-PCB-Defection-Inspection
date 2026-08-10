import cv2
import numpy as np

def extract_descriptors(blob):
    contour = blob["contour"]
    x, y, w, h = blob["bbox"]
    area = blob["area_px"]
    
    perimeter = cv2.arcLength(contour, True)
    aspect_ratio = float(w) / h if h != 0 else 0
    bounding_box_area = w * h
    extent = float(area) / bounding_box_area if bounding_box_area != 0 else 0
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area != 0 else 0

    return {
        "area_px": area,
        "perimeter_px": perimeter,
        "width_px": w,
        "height_px": h,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
    }

def classify_defect(polarity, aspect_ratio, solidity):
    if polarity == "removed":
        if aspect_ratio > 3.0:
            return "Open circuit"
        elif solidity > 0.9 and 0.8 < aspect_ratio < 1.2:
            return "Missing hole/pin-hole"
        else:
            return "Mouse bite"
    else:  # polarity == "added"
        if aspect_ratio > 3.0:
            return "Short"
        elif solidity > 0.85 and 0.8 < aspect_ratio < 1.2:
            return "Spurious copper"
        else:
            return "Spur"

def process_blobs(blobs, mm_per_px, tolerance):
    """Processes raw blobs, classifies them, converts to mm/mm², and decides Pass/Fail."""
    processed_defects = []
    
    for blob in blobs:
        desc = extract_descriptors(blob)
        defect_type = classify_defect(blob["polarity"], desc["aspect_ratio"], desc["solidity"])
        
        # Task 3.3: Physical Measurement Conversion
        area_mm2 = desc["area_px"] * (mm_per_px ** 2)
        width_mm = desc["width_px"] * mm_per_px
        height_mm = desc["height_px"] * mm_per_px
        
        processed_defects.append({
            "ID": blob["id"],
            "Class": defect_type,
            "Polarity": blob["polarity"],
            "Area (mm²)": round(area_mm2, 3),
            "Width (mm)": round(width_mm, 2),
            "Height (mm)": round(height_mm, 2),
            "bbox": blob["bbox"]
        })
        
    # Task 3.4: Board Verdict Logic
    verdict = "PASS" if len(processed_defects) <= tolerance else "FAIL"
    
    return processed_defects, verdict