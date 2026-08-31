"""
report_pdf.py
--------------
Generates a downloadable PDF compliance report from the dict produced by
compliance_checker.check_fields().
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime


def generate_pdf_report(report: dict, product_name: str, output_path: str):
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=18)
    elements = []

    elements.append(Paragraph("Legal Metrology Compliance Report", title_style))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(f"Product: {product_name}", styles["Normal"]))
    elements.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}", styles["Normal"]
        )
    )
    elements.append(Spacer(1, 12))

    status = "COMPLIANT" if report["overall_compliant"] else "NON-COMPLIANT"
    status_color = colors.HexColor("#0F6E56") if report["overall_compliant"] else colors.HexColor("#993C1D")
    status_style = ParagraphStyle(
        "Status", parent=styles["Heading2"], textColor=status_color
    )
    elements.append(Paragraph(f"Overall status: {status}", status_style))
    elements.append(Spacer(1, 12))

    # Field table
    table_data = [["Declaration", "Status", "Detected text"]]
    for key, res in report["fields"].items():
        status_cell = "Present" if res["present"] else "MISSING"
        table_data.append(
            [res["description"], status_cell, res["matched_text"] or "-"]
        )

    table = Table(table_data, colWidths=[75 * mm, 25 * mm, 60 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C2C2A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1EFE8")]),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 12))

    if report["readability_flags"]:
        elements.append(Paragraph("Readability concerns:", styles["Heading3"]))
        for f in report["readability_flags"]:
            elements.append(Paragraph(f"- '{f['text']}': {f['reason']}", styles["Normal"]))
        elements.append(Spacer(1, 8))

    if report["notes"]:
        elements.append(Paragraph("Notes:", styles["Heading3"]))
        for n in report["notes"]:
            elements.append(Paragraph(f"- {n}", styles["Normal"]))

    doc.build(elements)
    return output_path
