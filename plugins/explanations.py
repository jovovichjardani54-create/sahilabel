"""
plugins/explanations.py
-------------------------
PLUGIN — Feature 8: AI Violation Explanation (lightweight version)

Generates a human-readable explanation for each violation: which rule
was broken, what evidence was found (or not found), and a suggested
corrective action.

DESIGN NOTE: this uses templates, not a live LLM call. For a hackathon
demo, a template-based explainer is MORE reliable (no API latency, no
network dependency, no risk of a bad generation live on stage) while
still fully satisfying "human-readable explanation with rule + evidence
+ corrective action". If you want to upgrade to an LLM-generated version
later, swap generate_explanation()'s body for an API call - the calling
code in main.py doesn't need to change.
"""

CORRECTIVE_ACTIONS = {
    "manufacturer_info": "Add the manufacturer's, packer's, or importer's "
    "full name and address, prefixed with 'Mfd by' / 'Packed by' / "
    "'Marketed by' as applicable.",
    "product_name": "Print the common or generic name of the product "
    "clearly on the principal display panel.",
    "net_quantity": "Declare the net quantity in standard units "
    "(g, kg, ml, l) in a clearly visible position.",
    "mrp": "Print the Maximum Retail Price, inclusive of all taxes, "
    "prefixed with 'MRP' or the ₹ symbol.",
    "mfg_date": "Add the month and year of manufacture, packing, or "
    "import in MM/YYYY or Month YYYY format.",
    "customer_care": "Include a customer care contact - phone number, "
    "email, or helpline - for consumer complaints.",
}


def generate_explanation(field_key: str, field_result: dict) -> dict:
    """
    Args:
        field_key: e.g. "mrp"
        field_result: the dict for that field from report["fields"]

    Returns:
        {"rule_violated", "evidence", "suggested_action"}
    """
    if field_result["present"]:
        return {
            "rule_violated": None,
            "evidence": f"Found: '{field_result['matched_text']}'",
            "suggested_action": "No action needed - declaration is present.",
        }

    return {
        "rule_violated": field_result["description"],
        "evidence": "No matching text found anywhere on the scanned label.",
        "suggested_action": CORRECTIVE_ACTIONS.get(
            field_key, "Review the Legal Metrology Rules for this declaration."
        ),
    }


def generate_readability_explanation(flag: dict) -> dict:
    return {
        "rule_violated": "Rule 8 - font size / readability requirement",
        "evidence": flag["reason"],
        "suggested_action": "Increase the font size of this declaration "
        "so it is clearly legible relative to the rest of the label.",
    }


def build_full_explanation_set(report: dict) -> dict:
    """Convenience wrapper: builds explanations for every non-compliant
    field and every readability flag in one call."""
    explanations = {}
    for field_key, field_result in report["fields"].items():
        if not field_result["present"]:
            explanations[field_key] = generate_explanation(field_key, field_result)

    readability_explanations = [
        generate_readability_explanation(f) for f in report.get("readability_flags", [])
    ]

    return {
        "field_explanations": explanations,
        "readability_explanations": readability_explanations,
    }
