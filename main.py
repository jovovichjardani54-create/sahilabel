"""
main.py
-------
FastAPI backend for the Legal Metrology Compliance Checker.

Endpoints:
  POST /check         - upload an image, get back a JSON compliance report
  GET  /report/{id}/pdf - download the PDF for a previously-checked item
  GET  /history        - list all previously checked items (for the dashboard)
  GET  /                - serves the simple upload UI

Run with:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import uuid
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ocr_pipeline import extract_text
from compliance_checker import check_fields
from report_pdf import generate_pdf_report

# --- PLUGIN IMPORTS (new, additive only) ---
from plugins.scoring import compute_score
from plugins.highlighting import annotate_image
from plugins.explanations import build_full_explanation_set
from plugins.dashboard_stats import compute_dashboard_stats

app = FastAPI(title="Legal Metrology Compliance Checker")

UPLOAD_DIR = "uploads"
REPORT_DIR = "reports"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Simple in-memory history (swap for SQLite before final submission -
# this is fine for a hackathon demo, but resets on server restart)
HISTORY = []

# --- PLUGIN STORAGE (new, additive only) ---
# Kept separate from HISTORY/record dicts on purpose, so the existing
# /history endpoint's response shape never changes. Plugins look up
# what they need here by item_id.
_PLUGIN_STORE = {}  # item_id -> {"image_path": str, "ocr_words": list}


@app.post("/check")
async def check_label(
    file: UploadFile = File(...),
    latitude: float = Form(None),   # Feature: Geotagged Inspections
    longitude: float = Form(None),  # optional - None if not sent/denied
    inspector_id: str = Form(None), # optional - wire up once auth exists
):
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file")

    filename = file.filename or "uploaded_image.jpg"
    item_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(filename)[1] or ".jpg"
    image_path = os.path.join(UPLOAD_DIR, f"{item_id}{ext}")

    with open(image_path, "wb") as f:
        f.write(await file.read())

    ocr_result = extract_text(image_path)
    report = check_fields(ocr_result)

    # Generate evidence once from the already-computed OCR/report data so the
    # PDF and /annotated endpoint reuse the same decision-specific image.
    try:
        annotated_image_path = annotate_image(image_path, ocr_result["words"], report, item_id)
    except Exception:
        annotated_image_path = None

    pdf_path = os.path.join(REPORT_DIR, f"{item_id}.pdf")
    generate_pdf_report(report, filename, pdf_path, annotated_image_path)

    record = {
        "id": item_id,
        "filename": filename,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "overall_compliant": report["overall_compliant"],
        "overall_result": report.get(
            "overall_result", "PASS" if report["overall_compliant"] else "VIOLATION"
        ),
        "location": (
            {"latitude": latitude, "longitude": longitude}
            if latitude is not None and longitude is not None else None
        ),
        "inspector_id": inspector_id,
        "report": report,
    }
    HISTORY.append(record)

    # PLUGIN: stash what scoring/highlighting/explanations will need,
    # without touching `record` or HISTORY's shape at all
    _PLUGIN_STORE[item_id] = {
        "image_path": image_path,
        "ocr_words": ocr_result["words"],
        "annotated_image_path": annotated_image_path,
    }

    return JSONResponse(record)


@app.get("/report/{item_id}/pdf")
async def download_pdf(item_id: str):
    pdf_path = os.path.join(REPORT_DIR, f"{item_id}.pdf")
    if not os.path.exists(pdf_path):
        raise HTTPException(404, "Report not found")
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"compliance_{item_id}.pdf")


@app.get("/history")
async def get_history():
    # Return newest first, without the full report body (keep it light)
    return [
        {k: v for k, v in r.items() if k != "report"} for r in reversed(HISTORY)
    ]


# ============================================================
# NEW PLUGIN ENDPOINTS (Features 2, 3, 4, 8)
# None of these modify the routes above. Each is independently
# removable by deleting its @app.get/@app.post block below.
# ============================================================

def _find_record(item_id: str) -> dict:
    record = next((r for r in HISTORY if r["id"] == item_id), None)
    if not record:
        raise HTTPException(404, "Item not found")
    return record


@app.get("/score/{item_id}")
async def get_score(item_id: str):
    """Feature 3: Compliance Score"""
    record = _find_record(item_id)
    return compute_score(record["report"])


@app.get("/annotated/{item_id}")
async def get_annotated_image(item_id: str):
    """Feature 2: Violation Highlighting - returns the annotated image file"""
    record = _find_record(item_id)
    plugin_data = _PLUGIN_STORE.get(item_id)
    if not plugin_data:
        raise HTTPException(404, "No stored OCR data for this item")

    output_path = plugin_data.get("annotated_image_path")
    if not output_path or not os.path.exists(output_path):
        output_path = annotate_image(
            plugin_data["image_path"], plugin_data["ocr_words"], record["report"], item_id
        )
    return FileResponse(output_path, media_type="image/png")


@app.get("/explanations/{item_id}")
async def get_explanations(item_id: str):
    """Feature 8: AI Violation Explanation (template-based)"""
    record = _find_record(item_id)
    return build_full_explanation_set(record["report"])


@app.get("/dashboard/stats")
async def get_dashboard_stats():
    """Feature 4: Advanced Dashboard - aggregate stats, existing /history untouched"""
    return compute_dashboard_stats(HISTORY)


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r") as f:
        return f.read()


app.mount("/static", StaticFiles(directory="static"), name="static")
