# Legal Metrology Compliance Checker — v2 (extended)

## What changed from your working version

**Fixed a real bug**: `main.py` had two dead imports (`compliance_score`,
`annotate`) pointing to files that don't exist - your server would have
failed to start. Removed; the working equivalents already live in
`plugins/`.

**Verified working end-to-end** (all tested via curl, not just written):
`/check` → `/score/{id}` → `/explanations/{id}` → `/annotated/{id}` →
`/dashboard/stats`, plus geotagging fields and backward compatibility.

## Folder structure

```
legal_metrology_checker/
├── main.py                    # FastAPI app - MODIFIED (dead imports removed,
│                               #   geotag fields added, None-safety fix)
├── ocr_pipeline.py             # unchanged
├── rules_config.py             # unchanged
├── compliance_checker.py       # unchanged
├── report_pdf.py               # unchanged
├── auth.py                     # NEW - opt-in JWT auth, not wired in yet
├── requirements.txt            # MODIFIED - new deps appended, nothing removed
├── PATCH_multilingual_ocr.txt  # instructions to add Hindi/Tamil/Telugu OCR
├── static/
│   ├── index.html              # MODIFIED - camera UI + geotag checkbox merged in
│   ├── camera.js                # camera capture logic (reuses /check as-is)
│   └── camera_ui_snippet.html   # reference copy of the raw snippet (kept for docs)
├── plugins/                     # ALL NEW - additive only, main.py imports these
│   ├── scoring.py                # Feature: Compliance Score
│   ├── highlighting.py           # Feature: Violation Highlighting (OCR boxes, not YOLO - see note)
│   ├── explanations.py           # Feature: AI Violation Explanation (template-based)
│   └── dashboard_stats.py        # Feature: Inspection Dashboard
├── db/
│   └── database.py               # NEW - opt-in PostgreSQL layer, NOT imported by main.py
├── migrations/
│   └── 001_initial_schema.sql    # NEW - schema for users/products/inspections/violations
├── uploads/, reports/, annotated/  # existing runtime folders, unchanged behavior
```

## On YOLO specifically

The original ask was YOLO-based visual field detection. **This isn't
built, and honestly shouldn't be attempted this week** — training a
usable YOLO model needs a labeled dataset of hundreds/thousands of
product labels with bounding boxes drawn around each declaration type.
That dataset doesn't exist for this problem and building one is its own
multi-week project.

`plugins/highlighting.py` gets you the same *visual outcome* (boxes
drawn on the image, color-coded by violation type) using the OCR word
bounding boxes you already compute in `ocr_pipeline.py`. It's a
legitimate engineering substitution, not a hack — explain this trade-off
to judges directly if asked; "we used the OCR positions we already had
instead of training a model with no available dataset" is a strong,
honest answer.

## API endpoints (new, alongside your existing ones)

| Endpoint | Method | Purpose |
|---|---|---|
| `/check` | POST | unchanged route, now accepts optional `latitude`, `longitude`, `inspector_id` form fields |
| `/score/{item_id}` | GET | 0-100 compliance score + grade + category breakdown |
| `/annotated/{item_id}` | GET | PNG with violation boxes drawn on it |
| `/explanations/{item_id}` | GET | rule/evidence/suggested-fix per violation |
| `/dashboard/stats` | GET | total scanned, compliance rate, top violations, monthly trend |
| `/auth/login` | POST | *(opt-in, not required for anything above)* returns a JWT |

## Testing strategy

**What's already verified** (this conversation, via curl):
- `/check` on both a compliant and non-compliant synthetic label
- `/score` producing sensible numbers (100/A vs 27/F)
- `/explanations` citing correct rule sub-clauses with evidence
- `/annotated` producing a real, viewable PNG with correct violation boxes
- `/dashboard/stats` aggregating correctly
- Geotag fields round-tripping correctly, and working when omitted (backward compat)
- `auth.py`'s password hashing/verification and token creation, standalone

**What you should test next, in order**:
1. Run all of the above again on your machine, then on 5-10 real product
   photos (not synthetic) - OCR behaves differently on real images
2. Camera capture on an actual phone (getUserMedia requires HTTPS or
   localhost - `http://127.0.0.1:8000` is fine, a raw IP like
   `http://192.168.x.x:8000` on your phone will NOT get camera access
   without HTTPS)
3. Geotagging on a phone (desktop geolocation is often inaccurate/denied;
   test on mobile where it matters)
4. If you adopt Postgres: run the migration, call `save_inspection()`
   from a test script before wiring it into `/check`
5. If you adopt auth: test `/auth/login` with each of the 3 demo
   roles, then apply `Depends(require_role(...))` to ONE endpoint first
   and confirm both allowed and denied cases before rolling out further

## Deployment guide (realistic version for a hackathon demo)

**For the actual judging demo**: run it locally exactly as you have been
(`uvicorn main:app --host 0.0.0.0 --port 8000`) - this is the most
reliable option, zero network dependency, zero deploy-day surprises.

**If you want a public URL as a backup/bonus**:
1. Simplest: `ngrok http 8000` - gives you a temporary public HTTPS URL
   in one command, no config. Good enough for judges to try it on their
   own phones.
2. More permanent: deploy to Render.com or Railway.app (both have free
   tiers, both support FastAPI directly, both give you HTTPS automatically
   which you need for camera access on phones). Push this folder to a
   GitHub repo, connect it, set the start command to
   `uvicorn main:app --host 0.0.0.0 --port $PORT`.
3. Only add Postgres in deployment if you've actually adopted
   `db/database.py` beforehand and tested it - don't wire up a database
   for the first time under deployment pressure.

**Do not attempt**: Docker, Kubernetes, CI/CD pipelines, or cloud IAM
setup for this - genuinely not worth the time this week, and none of it
is visible in a 5-minute demo.
