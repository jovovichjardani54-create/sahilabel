"""
compliance_checker.py
----------------------
Takes OCR output (from ocr_pipeline.py) and checks it against the
mandatory declarations defined in rules_config.py.

Produces a structured compliance report:
    {
        "overall_compliant": bool,
        "fields": {
            "manufacturer_info": {"present": bool, "matched_text": str|None, ...},
            ...
        },
        "readability_flags": [...],
        "notes": [...]
    }
"""

from typing import Any, Mapping

from field_extractor import FOUND, NOT_FOUND, UNCERTAIN, extract_fields
from rules_config import MANDATORY_FIELDS

# ---- Font-size / readability heuristic ----
# NOTE: This is a RELATIVE heuristic, not the exact Rule 8 mm thresholds.
# Once your team confirms the real schedule (area of principal display
# panel -> minimum letter height in mm), replace this function's logic
# with an absolute check using image DPI to convert pixel height -> mm.
READABILITY_RATIO_THRESHOLD = 0.35  # a field's text height must be at
# least 35% of the largest text height on the label, else flagged


FIELD_MAPPING = {
    "manufacturer_or_packer": ("manufacturer_info", "Rule 6(1)(a)"),
    "generic_product_name": ("product_name", "Rule 6(1)(b)"),
    "net_quantity": ("net_quantity", "Rule 6(1)(c)"),
    "mrp": ("mrp", "Rule 6(1)(e)"),
    "manufacture_or_packing_date": ("mfg_date", "Rule 6(1)(d)"),
    "consumer_care": ("customer_care", "Rule 6(2)"),
}


def _has_sufficient_ocr_quality(ocr_result: dict) -> bool:
    """A conservative quality gate: an OCR failure can never prove absence."""
    words = ocr_result.get("words", [])
    confidences = [float(word.get("conf", -1)) for word in words if float(word.get("conf", -1)) >= 0]
    return bool(confidences) and sum(confidences) / len(confidences) >= 60.0


def _decision_for(extracted: dict, package_sides_complete: bool, ocr_quality_sufficient: bool) -> tuple[str, str]:
    if extracted["status"] == FOUND:
        return "PASS", extracted["reason"]
    if extracted["status"] == UNCERTAIN:
        return "REVIEW", f"{extracted['reason']} Inspector confirmation is required."
    if not package_sides_complete:
        return "REVIEW", "Declaration was not found, but package-side coverage is incomplete; it may appear on another side."
    if not ocr_quality_sufficient:
        return "REVIEW", "Declaration was not found, but OCR/image quality is insufficient to confirm an absence."
    return "VIOLATION", "Potential violation pending inspector confirmation: declaration was not found after complete package coverage and sufficient OCR quality."


def check_extracted_fields(
    extracted_fields: Mapping[str, Mapping[str, Any]],
    package_sides_complete: bool = False,
    ocr_quality_sufficient: bool = False,
    source_evidence: Mapping[str, Mapping[str, Any]] | None = None,
    readability_words: list[dict] | tuple[dict, ...] = (),
) -> dict:
    """Assess already-extracted declarations without changing rule semantics.

    ``source_evidence`` is optional provenance supplied by a caller that
    extracted fields from more than one source image. It is returned alongside
    each decision and is deliberately opaque to this checker: legal decisions
    still depend only on the existing extraction status, coverage and OCR
    quality inputs.
    """
    decisions, legacy_fields = {}, {}
    for field_key, extracted in extracted_fields.items():
        legacy_key, rule_id = FIELD_MAPPING[field_key]
        status, reason = _decision_for(
            extracted, package_sides_complete, ocr_quality_sufficient
        )
        decisions[field_key] = {
            "field_key": field_key, "rule_id": rule_id, "status": status,
            "extracted_value": extracted["value"], "confidence": extracted["confidence"],
            "evidence_text": extracted["evidence_text"], "bbox": extracted["bbox"],
            "reason": reason,
            "address": extracted.get("address"),
        }
        if source_evidence and field_key in source_evidence:
            decisions[field_key]["source_evidence"] = dict(source_evidence[field_key])
        definition = MANDATORY_FIELDS[legacy_key]
        legacy_fields[legacy_key] = {
            "present": status == "PASS", "matched_text": extracted["value"],
            "description": definition["description"], "severity": definition["severity"],
        }

    statuses = [decision["status"] for decision in decisions.values()]
    overall_result = "VIOLATION" if "VIOLATION" in statuses else "REVIEW" if "REVIEW" in statuses else "PASS"
    readability_flags = _check_readability(list(readability_words))
    notes = ["Decision support only: any potential violation requires inspector confirmation."]
    if not package_sides_complete:
        notes.append("Package-side coverage is incomplete by default for the single-image workflow.")
    if readability_flags:
        notes.append("Experimental readability observations are review notes only and do not create legal violations.")
    return {
        "overall_compliant": overall_result == "PASS", "overall_result": overall_result,
        "decision_support_message": "Potential violation pending inspector confirmation." if overall_result == "VIOLATION" else "Inspector review is required." if overall_result == "REVIEW" else "All extracted declarations passed automated checks.",
        "field_decisions": decisions, "fields": legacy_fields,
        "readability_flags": readability_flags, "notes": notes,
    }


def check_fields(ocr_result: dict, package_sides_complete: bool = False,
                 ocr_quality_sufficient: bool | None = None) -> dict:
    """Assess structured OCR evidence as decision support, not a legal finding."""
    words = ocr_result.get("words", [])
    quality = _has_sufficient_ocr_quality(ocr_result) if ocr_quality_sufficient is None else ocr_quality_sufficient
    return check_extracted_fields(
        extract_fields(ocr_result), package_sides_complete, quality,
        readability_words=words,
    )


OCR_CONFIDENCE_THRESHOLD = 70.0  # below this, treat as "hard to read"


def _check_readability(words: list) -> list:
    """Flags text that is either:
      (a) disproportionately small compared to the largest text on the
          label (proxy for the brand name / most prominent text), OR
      (b) read with low OCR confidence - which in practice often signals
          the same real-world problem (tiny/blurry/low-contrast print)
          as an explicit font-size violation, since a human would
          struggle to read it too.
    This is a placeholder for the real Rule 8 mm-based check - see the
    note at the top of this file."""
    if not words:
        return []

    max_height = max(w["height"] for w in words)
    flags = []
    for w in words:
        is_small = max_height > 0 and (w["height"] / max_height) < READABILITY_RATIO_THRESHOLD
        is_low_conf = w["conf"] < OCR_CONFIDENCE_THRESHOLD
        # Only flag words that look like they might be part of a mandatory
        # declaration (rough heuristic: contains a digit, since MRP/net-qty/
        # date all contain digits)
        looks_like_declaration = any(ch.isdigit() for ch in w["text"])

        if looks_like_declaration and (is_small or is_low_conf):
            reason = []
            if is_small:
                reason.append("text height below threshold vs. largest text on label")
            if is_low_conf:
                reason.append(f"low OCR confidence ({w['conf']}%) - likely hard to read")
            flags.append(
                {
                    "text": w["text"],
                    "height_px": w["height"],
                    "ratio_to_max": round(w["height"] / max_height, 2) if max_height else None,
                    "ocr_confidence": w["conf"],
                    "reason": "; ".join(reason),
                }
            )
    return flags


def print_report(report: dict):
    print("=" * 60)
    print("COMPLIANCE REPORT")
    print("=" * 60)
    status = report.get("overall_result", "PASS" if report["overall_compliant"] else "NON-COMPLIANT")
    print(f"Overall status: {status}\n")

    print("Field-by-field check:")
    if "field_decisions" in report:
        for decision in report["field_decisions"].values():
            print(f"  {decision['status']} {decision['field_key']} ({decision['rule_id']})")
            print(f"      -> value: '{decision['extracted_value']}'")
            print(f"      -> {decision['reason']}")
    else:
        for key, res in report["fields"].items():
            mark = "✅" if res["present"] else "❌"
            print(f"  {mark} {res['description']}")
            if res["present"]:
                print(f"      -> matched: '{res['matched_text']}'")

    if report["readability_flags"]:
        print("\n⚠️  Readability concerns:")
        for f in report["readability_flags"]:
            print(f"  - '{f['text']}' -> {f['reason']}")

    if report["notes"]:
        print("\n📝 Notes:")
        for n in report["notes"]:
            print(f"  - {n}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    from ocr_pipeline import extract_text

    if len(sys.argv) < 2:
        print("Usage: python compliance_checker.py <image_path>")
        sys.exit(1)

    ocr_result = extract_text(sys.argv[1])
    report = check_fields(ocr_result)
    print_report(report)
