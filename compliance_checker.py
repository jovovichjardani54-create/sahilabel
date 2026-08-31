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

from rules_config import MANDATORY_FIELDS, QUALIFYING_WORDS

# ---- Font-size / readability heuristic ----
# NOTE: This is a RELATIVE heuristic, not the exact Rule 8 mm thresholds.
# Once your team confirms the real schedule (area of principal display
# panel -> minimum letter height in mm), replace this function's logic
# with an absolute check using image DPI to convert pixel height -> mm.
READABILITY_RATIO_THRESHOLD = 0.35  # a field's text height must be at
# least 35% of the largest text height on the label, else flagged


def check_fields(ocr_result: dict) -> dict:
    full_text = ocr_result["full_text"]
    words = ocr_result["words"]

    field_results = {}
    for field_key, field_def in MANDATORY_FIELDS.items():
        match = field_def["pattern"].search(full_text)
        field_results[field_key] = {
            "present": bool(match),
            "matched_text": match.group(0) if match else None,
            "description": field_def["description"],
            "severity": field_def["severity"],
        }

    # Explanation I check: manufacturer info present WITHOUT a qualifying word
    notes = []
    if field_results["manufacturer_info"]["present"]:
        if not QUALIFYING_WORDS.search(full_text):
            notes.append(
                "Name/address found without a qualifying word "
                "('Mfd by'/'Packed by'/etc). Per Explanation I of Rule 6(1)(a), "
                "this is presumed to be the manufacturer's address - flagging "
                "for manual review."
            )

    readability_flags = _check_readability(words)

    overall_compliant = all(
        f["present"] for f in field_results.values() if f["severity"] == "critical"
    ) and not readability_flags

    return {
        "overall_compliant": overall_compliant,
        "fields": field_results,
        "readability_flags": readability_flags,
        "notes": notes,
    }


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
    status = "COMPLIANT ✅" if report["overall_compliant"] else "NON-COMPLIANT ❌"
    print(f"Overall status: {status}\n")

    print("Field-by-field check:")
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
