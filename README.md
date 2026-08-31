# Legal Metrology Compliance Checker (SIH26034)

## What this is
A starter pipeline that:
1. Takes a photo of a packaged product label
2. Runs OCR to extract all visible text (with word-level positions + confidence)
3. Checks the extracted text against the mandatory declarations required
   under Rule 6 of the Legal Metrology (Packaged Commodities) Rules, 2011
4. Produces a pass/fail compliance report explaining exactly what's
   missing and why

## Files
- `rules_config.py` - the checklist of mandatory fields as regex patterns.
  **Your team should verify these against the official Gazette text**
  before final submission.
- `ocr_pipeline.py` - wraps Tesseract OCR, returns full text + per-word
  bounding boxes/confidence.
- `compliance_checker.py` - runs the rules against OCR output, produces
  the report. Includes a readability heuristic (relative text height +
  OCR confidence) as a placeholder for the exact Rule 8 font-size mm
  thresholds - swap this out once your team confirms the real numbers.
- `make_test_labels.py` - generates synthetic test images (compliant +
  non-compliant) so you can test the pipeline before you have real photos.

## How to run
```bash
pip install -r requirements.txt
sudo apt-get install tesseract-ocr   # if not already installed

python3 make_test_labels.py                       # generate test images
python3 compliance_checker.py compliant_label.png
python3 compliance_checker.py noncompliant_label.png
```

## Known gaps / TODO for your team
1. **Exact Rule 8 font-size thresholds (mm) not yet confirmed** - the
   readability check currently uses a relative heuristic + OCR
   confidence. Replace with the real mm-based schedule once located in
   the official Rules PDF.
2. **Test on real product photos**, not just synthetic ones - real
   labels have curved surfaces, glare, multiple languages, and
   inconsistent layouts. Expect to tune `preprocess_image()` in
   ocr_pipeline.py.
3. **Next build steps**: wrap this in a FastAPI backend + simple
   upload UI, add PDF report export, add a SQLite-based history table.

## Web app (NEW)

Run the full web app locally:
```bash
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart reportlab
sudo apt-get install tesseract-ocr

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Then open http://localhost:8000 in your browser. Upload a label photo,
see the live compliance report, and download a PDF.

Verified working end-to-end (tested via curl):
- POST /check   -> returns full JSON compliance report
- GET /report/{id}/pdf -> returns a real downloadable PDF
- GET /history  -> lists all checks done this session

Current limitation: history is stored in-memory (resets on server
restart). Swap for SQLite before final submission if you want persistence
across demo restarts.
