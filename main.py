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

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ocr_pipeline import extract_text
from compliance_checker import check_fields
from report_pdf import generate_pdf_report

app = FastAPI(title="Legal Metrology Compliance Checker")

UPLOAD_DIR = "uploads"
REPORT_DIR = "reports"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Simple in-memory history (swap for SQLite before final submission -
# this is fine for a hackathon demo, but resets on server restart)
HISTORY = []


@app.post("/check")
async def check_label(file: UploadFile = File(...)):
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


    pdf_path = os.path.join(REPORT_DIR, f"{item_id}.pdf")
    generate_pdf_report(report, filename, pdf_path)

    

    

    record = {
        "id": item_id,
        "filename": filename,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "overall_compliant": report["overall_compliant"],
        "report": report,
    }
    HISTORY.append(record)

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


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r") as f:
        return f.read()


app.mount("/static", StaticFiles(directory="static"), name="static")
