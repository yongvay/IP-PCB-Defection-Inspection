import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_pdf_report(defects: list, verdict: str, ssim_score: float, mm_per_px: float, tolerance: int) -> bytes:
    """
    Generates an automated PDF inspection summary report (Extra Effort requirement).
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()

    # Custom Styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1E3A8A"),
        alignment=1,
        spaceAfter=15,
    )
    
    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#374151"),
    )

    elements = []

    # Title & Header
    elements.append(Paragraph("Automated PCB Defect Inspection Report", title_style))
    elements.append(Spacer(1, 10))

    # Metadata Table
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    verdict_color = colors.HexColor("#10B981") if verdict == "PASS" else colors.HexColor("#EF4444")
    
    meta_data = [
        [Paragraph(f"<b>Date/Time:</b> {timestamp}", meta_style), Paragraph(f"<b>Calibration:</b> {mm_per_px} mm/px", meta_style)],
        [Paragraph(f"<b>Board Verdict:</b> <font color='{verdict_color.hexval()}'><b>{verdict}</b></font>", meta_style), Paragraph(f"<b>Defect Tolerance:</b> {tolerance}", meta_style)],
        [Paragraph(f"<b>Total Defects:</b> {len(defects)}", meta_style), Paragraph(f"<b>Board SSIM Index:</b> {ssim_score}", meta_style)],
    ]

    meta_table = Table(meta_data, colWidths=[260, 260])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F3F4F6")),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#D1D5DB")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 20))

    # Detailed Defect Table
    elements.append(Paragraph("<b>Detailed Defect Breakdown</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    if defects:
        table_data = [["ID", "Defect Class", "Polarity", "Area (mm²)", "Width (mm)", "Height (mm)"]]
        for d in defects:
            table_data.append([
                str(d["ID"]),
                d["Class"],
                d["Polarity"].capitalize(),
                f"{d['Area (mm²)']:.4f}",
                f"{d['Width (mm)']:.3f}",
                f"{d['Height (mm)']:.3f}"
            ])

        defect_table = Table(table_data, colWidths=[40, 130, 90, 90, 85, 85])
        defect_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
        ]))
        elements.append(defect_table)
    else:
        elements.append(Paragraph("No defects detected on this board.", meta_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()