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
from report_pdf import generate_pdf_report
from history_analytics_router import create_history_analytics_router
from history_store import HistoryStore
from inspector_review import create_pending_review
from inspector_review_router import router as inspector_review_router
from product_evidence import evaluate_product_record
from product_record import OCRResult, ProductRecord
from quality_gate import assess_image_quality

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
os.makedirs("db", exist_ok=True)

# In-memory records retain session-only artifacts such as image paths. SQLite
# persists the compact, searchable history and analytics summaries.
HISTORY = []
HISTORY_STORE = HistoryStore(os.path.join("db", "history.sqlite3"))
PRODUCT_RECORDS = {}
app.include_router(inspector_review_router)
app.include_router(create_history_analytics_router(HISTORY_STORE))

# --- PLUGIN STORAGE (new, additive only) ---
# Kept separate from HISTORY/record dicts on purpose, so the existing
# /history endpoint's response shape never changes. Plugins look up
# what they need here by item_id.
_PLUGIN_STORE = {}  # item_id -> {"image_path": str, "ocr_words": list}


def _quality_gated_ocr(image_path: str, quality: dict) -> dict:
    """Run OCR only after the already-recorded quality gate permits it."""
    if quality.get("recommendation") != "PROCEED":
        return {"full_text": "", "words": []}
    return extract_text(image_path)


def _report_for_source(report: dict, source_label: str) -> dict:
    """Keep annotations in one image's coordinate space.

    Product aggregation preserves source provenance for every selected field.
    An annotation for a back or side image must therefore never draw a bbox
    selected from the front image.
    """
    decisions = report.get("field_decisions", {})
    source_decisions = {
        field_key: decision
        for field_key, decision in decisions.items()
        if decision.get("source_evidence", {}).get("source_label") == source_label
    }
    return {**report, "field_decisions": source_decisions}


def _preferred_evidence_label(report: dict, labels: list[str]) -> str | None:
    """Select the source with most selected declarations for the PDF image."""
    if not labels:
        return None
    counts = {label: 0 for label in labels}
    for decision in report.get("field_decisions", {}).values():
        source_label = decision.get("source_evidence", {}).get("source_label")
        if source_label in counts:
            counts[source_label] += 1
    return max(enumerate(labels), key=lambda item: (counts[item[1]], -item[0]))[1]


@app.post("/check")
async def check_label(
    file: UploadFile | None = File(None),
    files: list[UploadFile] | None = File(None),
    image_labels: list[str] | None = Form(None),
    latitude: float = Form(None),   # Feature: Geotagged Inspections
    longitude: float = Form(None),  # optional - None if not sent/denied
    inspector_id: str = Form(None), # optional - wire up once auth exists
):
    uploads = files if files else ([file] if file is not None else [])
    if not uploads:
        raise HTTPException(400, "Please upload at least one image file")
    if image_labels is not None and len(image_labels) != len(uploads):
        raise HTTPException(400, "image_labels must have one label for each uploaded file")
    for upload in uploads:
        if not (upload.content_type or "").startswith("image/"):
            raise HTTPException(400, "Please upload image files only")

    filename = uploads[0].filename or "uploaded_image.jpg"
    item_id = str(uuid.uuid4())[:8]
    default_labels = ("front", "back", "side")
    labels = image_labels or [
        default_labels[index] if index < len(default_labels) else f"image-{index + 1}"
        for index in range(len(uploads))
    ]
    product_record = ProductRecord(product_id=item_id, session_id=item_id)
    image_paths, quality_assessments, ocr_results = [], [], []
    try:
        for index, (upload, label) in enumerate(zip(uploads, labels)):
            uploaded_name = upload.filename or f"uploaded_image_{index + 1}.jpg"
            ext = os.path.splitext(uploaded_name)[1] or ".jpg"
            path = os.path.join(UPLOAD_DIR, f"{item_id}_{index + 1}{ext}")
            with open(path, "wb") as destination:
                destination.write(await upload.read())
            quality = assess_image_quality(path)
            ocr_result = _quality_gated_ocr(path, quality)
            normalized_label = label.strip().lower()
            product_record.add_image(
                normalized_label,
                path,
                [OCRResult(ocr_result.get("full_text", ""), ocr_result.get("words", []))],
                {"quality_assessment": quality},
            )
            image_paths.append(path)
            quality_assessments.append({"label": normalized_label, **quality})
            ocr_results.append(ocr_result)
    except (TypeError, ValueError) as error:
        raise HTTPException(400, str(error)) from error

    # Product evidence is aggregated only through the tested, source-preserving
    # integration contract. It derives coverage and retains each bbox in the
    # image where that declaration was actually observed.
    evaluation = evaluate_product_record(product_record)
    report = dict(evaluation.report)
    report["quality_assessment"] = quality_assessments[0]
    if len(quality_assessments) > 1:
        report["additional_quality_assessments"] = quality_assessments[1:]

    # Generate one annotation per source image. The source-specific report
    # excludes declarations selected from other images, so coordinates are not
    # merged or projected onto the wrong package side.
    annotated_image_paths = {}
    for image, ocr_result in zip(product_record.image_list, ocr_results):
        try:
            annotated_image_paths[image.label] = annotate_image(
                image.source,
                ocr_result["words"],
                _report_for_source(report, image.label),
                f"{item_id}_{image.label}",
            )
        except Exception:
            # Visual evidence remains optional; an annotation failure never
            # changes the extraction or compliance decision.
            continue
    evidence_label = _preferred_evidence_label(
        report, [image.label for image in product_record.image_list]
    )
    annotated_image_path = annotated_image_paths.get(evidence_label)

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
        "product_record": {
            "image_labels": [image.label for image in product_record.image_list],
            "coverage_status": product_record.coverage_status.value,
        },
        "report": report,
    }
    if report["overall_result"] == "REVIEW" and inspector_id:
        record["inspector_review"] = create_pending_review(
            item_id,
            report["overall_result"],
            inspector_id=inspector_id,
        )
    if annotated_image_paths:
        record["annotated_evidence"] = {
            label: f"/annotated/{item_id}/{label}"
            for label in annotated_image_paths
        }
    HISTORY.append(record)
    HISTORY_STORE.save_report(
        item_id, filename, report,
        reviewer_status=(record.get("inspector_review") or {}).get("inspector_decision"),
        created_at=record["timestamp"],
    )
    PRODUCT_RECORDS[item_id] = product_record

    # PLUGIN: stash what scoring/highlighting/explanations will need,
    # without touching `record` or HISTORY's shape at all
    _PLUGIN_STORE[item_id] = {
        "image_path": image_paths[0],
        "ocr_words": ocr_results[0]["words"],
        "annotated_image_path": annotated_image_path,
        "annotated_image_paths": annotated_image_paths,
        "source_images": {
            image.label: {"image_path": image.source, "ocr_words": ocr_result["words"]}
            for image, ocr_result in zip(product_record.image_list, ocr_results)
        },
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
    # Return persisted, compact metadata only; uploaded images and PDFs remain
    # filesystem artifacts and are intentionally not copied into SQLite.
    active_records = {record["id"]: record for record in HISTORY}
    history = []
    for item in HISTORY_STORE.list_recent_reports():
        active_record = active_records.get(item["product_session_id"])
        if active_record is not None:
            history.append({key: value for key, value in active_record.items() if key != "report"})
            continue
        history.append({
            "id": item["product_session_id"],
            "filename": item["filename"],
            "timestamp": item["created_at"],
            "overall_compliant": item["overall_result"] == "PASS",
            "overall_result": item["overall_result"],
            "location": None,
            "inspector_id": None,
            "reviewer_status": item["reviewer_status"],
        })
    return history


@app.get("/history/search")
async def search_history(
    query: str | None = None,
    status: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """Search compact persisted summaries without changing /history."""
    try:
        return HISTORY_STORE.search_reports(
            query, overall_result=status, start_date=start_date, end_date=end_date
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(400, str(error)) from error


@app.get("/history/analytics")
async def history_analytics(start_date: str | None = None, end_date: str | None = None):
    """Return lightweight result totals for the requested date range."""
    try:
        return HISTORY_STORE.analytics_counts(start_date=start_date, end_date=end_date)
    except (TypeError, ValueError) as error:
        raise HTTPException(400, str(error)) from error


@app.get("/history/{item_id}")
async def get_history_summary(item_id: str):
    """Retrieve one compact persisted report summary."""
    summary = HISTORY_STORE.get_report_summary(item_id)
    if summary is None:
        raise HTTPException(404, "History record not found")
    return summary


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
    """Return the primary source annotation for legacy clients."""
    record = _find_record(item_id)
    plugin_data = _PLUGIN_STORE.get(item_id)
    if not plugin_data:
        raise HTTPException(404, "No stored OCR data for this item")

    output_path = plugin_data.get("annotated_image_path")
    if not output_path or not os.path.exists(output_path):
        source_images = plugin_data.get("source_images", {})
        source_label = _preferred_evidence_label(record["report"], list(source_images))
        source = source_images.get(source_label) if source_label else None
        if source is None:
            output_path = annotate_image(
                plugin_data["image_path"], plugin_data["ocr_words"], record["report"], item_id
            )
        else:
            output_path = annotate_image(
                source["image_path"],
                source["ocr_words"],
                _report_for_source(record["report"], source_label),
                f"{item_id}_{source_label}",
            )
        plugin_data["annotated_image_path"] = output_path
    return FileResponse(output_path, media_type="image/png")


@app.get("/annotated/{item_id}/{image_label}")
async def get_source_annotated_image(item_id: str, image_label: str):
    """Return the annotation for one labelled source image, if available."""
    _find_record(item_id)
    plugin_data = _PLUGIN_STORE.get(item_id)
    output_path = (plugin_data or {}).get("annotated_image_paths", {}).get(
        image_label.strip().lower()
    )
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(404, "Annotated evidence for this image is not available")
    return FileResponse(output_path, media_type="image/png")


@app.get("/explanations/{item_id}")
async def get_explanations(item_id: str):
    """Feature 8: AI Violation Explanation (template-based)"""
    record = _find_record(item_id)
    return build_full_explanation_set(record["report"])


@app.get("/dashboard/stats")
async def get_dashboard_stats():
    """Feature 4: Advanced Dashboard - aggregate stats, existing /history untouched"""
    # Keep the legacy dashboard payload intact and add the durable aggregate
    # calculated by the tested SQLite-backed analytics service/router.
    dashboard = compute_dashboard_stats(HISTORY)
    dashboard["persistent_analytics"] = HISTORY_STORE.analytics_summary()
    return dashboard


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r") as f:
        return f.read()


app.mount("/static", StaticFiles(directory="static"), name="static")
