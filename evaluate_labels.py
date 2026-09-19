"""Evaluate genuine local package photos against a manually transcribed JSON manifest.

Usage: python evaluate_labels.py validation_samples/manifest.json

The manifest has a ``products`` list. Each product has an ``id``, ``images``
(``path``, ``label``, and ``visible_text`` phrases), and ``expected`` fields.
Use null for a field not visible on the supplied sides, never a guessed value.
The tool writes no uploads, model files, or reports to the repository.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean

from ocr_pipeline import extract_text
from product_evidence import evaluate_product_record
from product_record import OCRResult, ProductRecord
from quality_gate import assess_image_quality

FIELDS = ("generic_product_name", "mrp", "net_quantity", "manufacturer_or_packer",
          "manufacture_or_packing_date", "consumer_care")


def _plain(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").casefold())


def evaluate(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    rows = []
    providers = {"paddle": 0, "tesseract": 0}
    visible_total = visible_hit = eligible_visible_total = eligible_visible_hit = 0
    expected_total = expected_hit = false_positives = 0
    times = []
    for product in manifest["products"]:
        record = ProductRecord(product_id=str(product["id"]))
        image_rows = []
        all_text = []
        for image in product["images"]:
            path = (root / image["path"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            quality = assess_image_quality(str(path))
            ocr = extract_text(str(path))
            metadata = ocr["metadata"]
            times.append(float(metadata["elapsed_seconds"]))
            for provider in metadata["providers_used"]:
                if provider in providers:
                    providers[provider] += 1
            expected_phrases = image.get("visible_text", [])
            found_phrases = [phrase for phrase in expected_phrases
                             if _plain(phrase) in _plain(ocr["full_text"])]
            visible_total += len(expected_phrases)
            visible_hit += len(found_phrases)
            if metadata["providers_used"]:
                eligible_visible_total += len(expected_phrases)
                eligible_visible_hit += len(found_phrases)
            all_text.append(ocr["full_text"])
            record.add_image(image["label"], str(path),
                             [OCRResult(ocr["full_text"], ocr["words"], metadata)],
                             {"quality_assessment": quality})
            image_rows.append({"path": str(path), "label": image["label"],
                               "quality": quality["recommendation"],
                               "metadata": metadata, "ocr_text": ocr["full_text"],
                               "visible_text_found": found_phrases,
                               "visible_text_missed": [phrase for phrase in expected_phrases
                                                       if phrase not in found_phrases]})
        assessment = evaluate_product_record(record)
        fields = {}
        joined_ocr = _plain(" ".join(all_text))
        for key in FIELDS:
            expected = product.get("expected", {}).get(key)
            selected = assessment.evidence.declarations[key]
            actual = selected.value
            if expected is None:
                category = "absent_on_supplied_sides" if actual is None else "false_positive"
                false_positives += actual is not None
            else:
                expected_total += 1
                correct = bool(actual and (_plain(expected) in _plain(actual) or
                                           _plain(actual) in _plain(expected)))
                expected_hit += correct
                if correct:
                    category = "correct"
                elif not any(image["metadata"]["providers_used"] for image in image_rows):
                    category = "quality_gate_blocked"
                elif _plain(expected) not in joined_ocr:
                    category = "visible_but_ocr_missed"
                elif selected.status == "UNCERTAIN":
                    category = "detected_but_low_confidence"
                else:
                    category = "detected_but_extraction_missed_or_ambiguous"
            fields[key] = {"expected": expected, "actual": actual,
                           "status": selected.status, "category": category,
                           "source_label": selected.source_label,
                           "provider": selected.provider}
        rows.append({"id": product["id"], "images": image_rows,
                     "fields": fields, "decision": assessment.report["overall_result"]})
    return {"product_count": len(rows), "image_count": sum(len(row["images"]) for row in rows),
            "paddle_initialization_success": providers["paddle"] > 0,
            "paddle_usage_count": providers["paddle"],
            "tesseract_fallback_count": providers["tesseract"],
            "visible_text_recall": {"found": visible_hit, "total": visible_total,
                                    "rate": visible_hit / visible_total if visible_total else None},
            "ocr_processed_visible_text_recall": {
                "found": eligible_visible_hit, "total": eligible_visible_total,
                "rate": eligible_visible_hit / eligible_visible_total if eligible_visible_total else None},
            "required_field_accuracy": {"correct": expected_hit, "total": expected_total,
                                        "rate": expected_hit / expected_total if expected_total else None},
            "false_positive_fields": false_positives,
            "average_processing_seconds_per_image": mean(times) if times else 0,
            "products": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--summary", action="store_true", help="Print aggregate and concise per-product results")
    args = parser.parse_args()
    result = evaluate(args.manifest)
    if args.summary:
        result["products"] = [{"id": row["id"], "decision": row["decision"],
                               "images": [{"label": image["label"], "quality": image["quality"],
                                           "providers": image["metadata"]["providers_used"],
                                           "seconds": image["metadata"]["elapsed_seconds"],
                                           "visible_text_found": image["visible_text_found"],
                                           "visible_text_missed": image["visible_text_missed"]}
                                          for image in row["images"]],
                               "fields": row["fields"]} for row in result["products"]]
    print(json.dumps(result, ensure_ascii=False, indent=2))
