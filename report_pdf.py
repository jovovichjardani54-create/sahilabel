"""Presentation-ready PDF reports for structured compliance decisions."""

import os
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image as ReportImage, LongTable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

STATUS_COLORS = {"PASS": colors.HexColor("#0F6E56"), "REVIEW": colors.HexColor("#B45309"), "VIOLATION": colors.HexColor("#993C1D")}
DECLARATION_NAMES = {"manufacturer_or_packer": "Manufacturer / packer details", "generic_product_name": "Generic product name", "net_quantity": "Net quantity", "mrp": "Maximum retail price (MRP)", "manufacture_or_packing_date": "Manufacture / packing date", "consumer_care": "Consumer-care contact"}


def _safe(value, fallback="Not detected"):
    return fallback if value is None or value == "" else str(value)


def _decisions(report):
    structured = list(report.get("field_decisions", {}).values())
    if structured:
        return structured
    return [{"field_key": key, "declaration": field.get("description", key), "rule_id": "-", "status": "PASS" if field.get("present") else "VIOLATION", "extracted_value": field.get("matched_text"), "confidence": None, "reason": "Present in the legacy report." if field.get("present") else "Missing in the legacy report."} for key, field in report.get("fields", {}).items()]


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D6D3D1")); canvas.line(doc.leftMargin, 12 * mm, A4[0] - doc.rightMargin, 12 * mm)
    canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor("#57534E"))
    canvas.drawString(doc.leftMargin, 7 * mm, "SahiLabel - Decision support only; inspector confirmation may be required.")
    canvas.drawRightString(A4[0] - doc.rightMargin, 7 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _visual_evidence_section(image_path, section, cell):
    """Use the existing annotated image as-is; never recreate annotations here."""
    elements = [PageBreak(), Paragraph("Visual Evidence", section), Spacer(1, 3 * mm)]
    if not image_path or not os.path.exists(image_path):
        return elements + [Paragraph("Visual evidence unavailable.", cell)]
    try:
        source_width, source_height = ImageReader(image_path).getSize()
        scale = min((180 * mm) / source_width, (205 * mm) / source_height)
        image = ReportImage(image_path, width=source_width * scale, height=source_height * scale, hAlign="CENTER")
        elements.extend([image, Spacer(1, 3 * mm), Paragraph("Highlighted regions show the label evidence used for each declaration decision.", cell), Spacer(1, 3 * mm)])
        legend = Table([["", "Green: PASS", "", "Amber: REVIEW", "", "Red: VIOLATION"]], colWidths=[6 * mm, 35 * mm, 6 * mm, 39 * mm, 6 * mm, 38 * mm], hAlign="LEFT")
        legend.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), STATUS_COLORS["PASS"]), ("BACKGROUND", (2, 0), (2, 0), STATUS_COLORS["REVIEW"]), ("BACKGROUND", (4, 0), (4, 0), STATUS_COLORS["VIOLATION"]), ("FONTNAME", (0, 0), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
        elements.append(legend)
    except Exception:
        elements.append(Paragraph("Visual evidence unavailable.", cell))
    return elements


def generate_pdf_report(report: dict, product_name: str, output_path: str, evidence_image_path: str | None = None):
    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm, bottomMargin=18 * mm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=colors.HexColor("#1C1917"), spaceAfter=2)
    meta = ParagraphStyle("Meta", parent=styles["Normal"], fontSize=10, leading=14, textColor=colors.HexColor("#57534E"))
    cell = ParagraphStyle("Cell", parent=styles["BodyText"], fontSize=8, leading=10, wordWrap="CJK")
    section = ParagraphStyle("Section", parent=styles["Heading2"], fontSize=12, leading=15, textColor=colors.HexColor("#1C1917"), spaceBefore=12, spaceAfter=6)
    review = report.get("inspector_decision") or {}
    overall = report.get("final_decision") or report.get("overall_result", "PASS" if report.get("overall_compliant") else "VIOLATION")
    color = STATUS_COLORS.get(overall, STATUS_COLORS["REVIEW"])
    message = _safe(report.get("decision_support_message"), "Automated result requires review where indicated.")
    elements = [Paragraph("SahiLabel", title), Paragraph("Legal Metrology Compliance Assessment", meta), Spacer(1, 5 * mm), Paragraph(f"<b>Product:</b> {_safe(product_name, 'Uploaded image')}", meta), Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%d %b %Y, %H:%M')}", meta), Spacer(1, 4 * mm)]
    status = Table([[Paragraph(f"<b>Overall result: {overall}</b>", ParagraphStyle("Status", parent=styles["Heading2"], fontSize=15, leading=19, textColor=color)), Paragraph(message, cell)]], colWidths=[48 * mm, 132 * mm])
    status.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F5F4")), ("BOX", (0, 0), (-1, -1), .75, color), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    elements.extend([status, Spacer(1, 7 * mm), Paragraph("Declaration decisions", section)])
    if review.get("review_status") == "RESOLVED":
        elements.insert(-1, Paragraph(
            f"<b>Inspector decision:</b> {review['final_decision']} by {_safe(review.get('reviewer_name'))} "
            f"({_safe(review.get('inspector_id'))}). Reason: {_safe(review.get('inspector_reason'))}. "
            "Automated decision: REVIEW; the original field evidence remains below.", meta,
        ))
    decisions = _decisions(report)
    table_data = [[Paragraph("<b>Declaration</b>", cell), Paragraph("<b>Rule</b>", cell), Paragraph("<b>Decision</b>", cell), Paragraph("<b>Extracted value</b>", cell), Paragraph("<b>Confidence</b>", cell)]]
    styles_to_apply = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#292524")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#D6D3D1")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]
    for index, decision in enumerate(decisions, 1):
        decision_status = decision.get("status", "REVIEW")
        name = decision.get("declaration") or DECLARATION_NAMES.get(decision.get("field_key"), decision.get("field_key", "Declaration"))
        confidence = decision.get("confidence")
        confidence = f"{float(confidence):.1f}%" if isinstance(confidence, (int, float)) else "Not available"
        decision_style = ParagraphStyle(f"Decision{index}", parent=cell, textColor=STATUS_COLORS.get(decision_status, STATUS_COLORS["REVIEW"]))
        table_data.append([Paragraph(_safe(name), cell), Paragraph(_safe(decision.get("rule_id"), "-"), cell), Paragraph(f"<b>{decision_status}</b>", decision_style), Paragraph(_safe(decision.get("extracted_value")), cell), Paragraph(confidence, cell)])
    table = LongTable(table_data, colWidths=[45 * mm, 21 * mm, 25 * mm, 58 * mm, 31 * mm], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle(styles_to_apply)); elements.append(table)
    review_items = [d for d in decisions if d.get("status") in {"REVIEW", "VIOLATION"}]
    if review_items:
        elements.extend([Spacer(1, 5 * mm), Paragraph("Review notes", section)])
        for decision in review_items:
            name = DECLARATION_NAMES.get(decision.get("field_key"), decision.get("field_key", "Declaration"))
            elements.extend([Paragraph(f"<b>{_safe(name)} ({decision.get('status', 'REVIEW')}):</b> {_safe(decision.get('reason'), 'No reason provided.')}", cell), Spacer(1, 2 * mm)])
    elements.extend(_visual_evidence_section(evidence_image_path, section, cell))
    doc.build(elements, onFirstPage=_footer, onLaterPages=_footer)
    return output_path
