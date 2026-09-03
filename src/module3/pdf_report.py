"""Automated PDF inspection report — extra-effort reporting requirement.

Owner: Ng Zhi Xuan (Module 3).

The assignment lists automated export of results into PDF under extra efforts.
This module turns one ``InspectionReport`` into a document an operator could
file or hand to a production engineer, which means it has to answer more than
"how many defects": it states the verdict, the condition that produced it, the
physical size of every defect in millimetres, and the settings the inspection
ran under, so that a report read six months later can still be interpreted.

The report is built from the ``InspectionReport`` contract object rather than
from dictionaries assembled by the dashboard. That matters: it means the PDF
and the screen cannot disagree, and it means a batch run can export reports
without a dashboard being involved at all.
"""

from __future__ import annotations

import io
from collections import Counter
from datetime import datetime

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.contracts import InspectionReport
from src.module3.classify import CONTRACT_TO_DISPLAY

NAVY = colors.HexColor("#1E3A8A")
GREY_TEXT = colors.HexColor("#374151")
GREY_RULE = colors.HexColor("#D1D5DB")
GREY_FILL = colors.HexColor("#F3F4F6")
GREY_STRIPE = colors.HexColor("#F9FAFB")
PASS_GREEN = colors.HexColor("#10B981")
FAIL_RED = colors.HexColor("#EF4444")

# Severity weight to a word, so that the reader is not asked to interpret a
# bare integer. The mapping mirrors SEVERITY in classify.py.
SEVERITY_WORD = {3: "Critical", 2: "Major", 1: "Minor"}


def generate_pdf_report(report: InspectionReport,
                        ssim_score: float | None = None,
                        board_name: str = "",
                        annotated_image: np.ndarray | None = None,
                        params: dict | None = None) -> bytes:
    """Render one inspection as a PDF and return it as bytes.

    Bytes rather than a file path because the dashboard streams the result
    straight into a download button, and a batch export writes it wherever the
    caller chooses. Nothing here touches the file system.
    """
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
        title="PCB Inspection Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Heading1"], fontSize=18, leading=22,
        textColor=NAVY, alignment=1, spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"], fontSize=9, leading=12,
        textColor=GREY_TEXT, alignment=1, spaceAfter=14,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=9, leading=13,
        textColor=GREY_TEXT,
    )
    heading_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], fontSize=12, leading=15,
        textColor=NAVY, spaceBefore=12, spaceAfter=6,
    )

    elements = [
        Paragraph("Automated PCB Defect Inspection Report", title_style),
        Paragraph(
            "BMDS2133 Image Processing &nbsp;·&nbsp; Golden-template inspection "
            "&nbsp;·&nbsp; classical image processing only",
            subtitle_style,
        ),
        _verdict_banner(report),
        Spacer(1, 10),
        _metadata_table(report, ssim_score, board_name, body_style),
    ]

    if annotated_image is not None:
        elements.append(Paragraph("Annotated inspection result", heading_style))
        elements.append(_image_flowable(annotated_image))

    elements.append(Paragraph("Defect summary by class", heading_style))
    elements.append(_summary_table(report))

    elements.append(Paragraph("Detailed defect breakdown", heading_style))
    elements.append(_detail_table(report, body_style))

    elements.append(Paragraph("Inspection settings", heading_style))
    elements.append(Paragraph(_settings_line(report, params), body_style))

    document.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def _verdict_banner(report: InspectionReport) -> Table:
    """The verdict and the condition that produced it, in one block.

    The reason is printed beside the verdict rather than omitted, because a
    board marked FAIL with no stated cause tells an operator nothing about what
    to do next.
    """
    passed = report.verdict == "PASS"
    style = ParagraphStyle(
        "Verdict", fontSize=14, leading=17, textColor=colors.white,
        fontName="Helvetica-Bold",
    )
    reason_style = ParagraphStyle(
        "VerdictReason", fontSize=9, leading=12, textColor=colors.white,
    )

    cell = [
        Paragraph(f"BOARD {report.verdict}", style),
        Paragraph(report.verdict_reason or "no reason recorded", reason_style),
    ]
    table = Table([[cell]], colWidths=[document_width()])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PASS_GREEN if passed else FAIL_RED),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return table


def _metadata_table(report: InspectionReport,
                    ssim_score: float | None,
                    board_name: str,
                    style: ParagraphStyle) -> Table:
    timestamp = datetime.now().strftime("%d %B %Y, %H:%M:%S")
    total_area = sum(defect.area_mm2 for defect in report.defects)

    left = [
        f"<b>Inspected:</b> {timestamp}",
        f"<b>Board:</b> {board_name or 'not recorded'}",
        f"<b>Total defects:</b> {len(report.defects)}",
        f"<b>Total defective area:</b> {total_area:.4f} mm&sup2;",
    ]
    right = [
        f"<b>Calibration:</b> {report.mm_per_px:.6f} mm/px",
        f"<b>Alignment residual:</b> {report.align_residual:.3f} px",
        f"<b>Runtime:</b> {report.runtime_s:.3f} s",
        f"<b>Board SSIM:</b> {ssim_score:.4f}" if ssim_score is not None
        else "<b>Board SSIM:</b> not computed",
    ]

    rows = [[Paragraph(a, style), Paragraph(b, style)]
            for a, b in zip(left, right)]
    table = Table(rows, colWidths=[document_width() / 2] * 2)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), GREY_FILL),
        ("BOX", (0, 0), (-1, -1), 0.75, GREY_RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def _summary_table(report: InspectionReport) -> Table:
    """Counts and total area per class, ordered by severity then count.

    An operator reads this before the detailed table: three shorts and one spur
    is a different production problem from one short and three spurs, and the
    per-defect list makes that hard to see at a glance.
    """
    counts = Counter(defect.defect_class for defect in report.defects)
    areas: dict[str, float] = {}
    severities: dict[str, int] = {}
    for defect in report.defects:
        areas[defect.defect_class] = areas.get(defect.defect_class, 0.0) + defect.area_mm2
        severities[defect.defect_class] = defect.severity

    if not counts:
        return _empty_table("No defects detected on this board.")

    ordered = sorted(
        counts, key=lambda name: (-severities.get(name, 1), -counts[name])
    )
    rows = [["Defect class", "Severity", "Count", "Total area (mm\u00b2)"]]
    for name in ordered:
        rows.append([
            CONTRACT_TO_DISPLAY.get(name, name),
            SEVERITY_WORD.get(severities.get(name, 1), "Minor"),
            str(counts[name]),
            f"{areas[name]:.4f}",
        ])

    table = Table(rows, colWidths=[document_width() * fraction
                                   for fraction in (0.40, 0.20, 0.15, 0.25)])
    table.setStyle(_grid_style())
    return table


def _detail_table(report: InspectionReport, style: ParagraphStyle) -> Table:
    """One row per defect, with the physical measurements task 3.3 produces."""
    if not report.defects:
        return _empty_table("No defects detected on this board.")

    rows = [["ID", "Class", "Polarity", "Length (mm)", "Width (mm)",
             "Area (mm\u00b2)", "Conf.", "Decided by"]]
    for defect in sorted(report.defects, key=lambda item: -item.severity):
        rows.append([
            str(defect.id),
            CONTRACT_TO_DISPLAY.get(defect.defect_class, defect.defect_class),
            str(defect.polarity).capitalize(),
            f"{defect.width_mm:.3f}",
            f"{defect.height_mm:.3f}",
            f"{defect.area_mm2:.4f}",
            f"{defect.confidence:.2f}",
            defect.decided_by or "n/a",
        ])

    table = Table(rows, repeatRows=1,
                  colWidths=[document_width() * fraction for fraction in
                             (0.06, 0.22, 0.11, 0.13, 0.12, 0.14, 0.09, 0.13)])
    table.setStyle(_grid_style())
    return table


def _settings_line(report: InspectionReport, params: dict | None) -> str:
    """Record what the inspection ran under, so the result is reproducible."""
    params = params or {}
    detail = report.verdict_detail or {}
    parts = [
        f"Classifier: {report.classifier or 'not recorded'}",
        f"Defect tolerance: {detail.get('max_defects', 'not set')}",
        f"Total severity weight: {detail.get('total_severity', 0)}",
    ]
    for key in ("morph_open_kernel", "min_blob_area", "binarise", "denoise"):
        if key in params:
            parts.append(f"{key.replace('_', ' ')}: {params[key]}")
    return " &nbsp;·&nbsp; ".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def document_width() -> float:
    """Usable width of the page body, in points."""
    return A4[0] - 36 * mm


def _grid_style() -> TableStyle:
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, GREY_RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GREY_STRIPE]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ])


def _empty_table(message: str) -> Table:
    table = Table([[message]], colWidths=[document_width()])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), GREY_FILL),
        ("BOX", (0, 0), (-1, -1), 0.75, GREY_RULE),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), GREY_TEXT),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return table


def _image_flowable(image: np.ndarray, max_height: float = 95 * mm) -> Image:
    """Embed an OpenCV BGR or greyscale array without writing a temporary file.

    Encoded to PNG in memory. JPEG would be smaller but it would introduce
    compression artefacts into a document whose purpose is to record exactly
    what the inspection saw.
    """
    import cv2

    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("could not encode the annotated image")

    stream = io.BytesIO(encoded.tobytes())
    height, width = image.shape[:2]
    scale = min(document_width() / width, max_height / height)
    return Image(stream, width=width * scale, height=height * scale)
