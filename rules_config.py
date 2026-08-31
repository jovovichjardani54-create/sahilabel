"""
rules_config.py
----------------
Encodes the mandatory declarations required under Rule 6 of the
Legal Metrology (Packaged Commodities) Rules, 2011 as machine-checkable
regex patterns.

IMPORTANT (read before demo/pitch):
This is a STARTING checklist derived from the publicly available text of
Rule 6. Your team should verify the exact wording against the official
Gazette copy of the Rules (via indiacode.nic.in or the Ministry of Consumer
Affairs) before the final submission, and update patterns as needed.
The FONT_SIZE section below uses a RELATIVE heuristic (not the exact legal
mm thresholds) until your team confirms the precise Rule 8 schedule -
see the note in compliance_checker.py.
"""

import re

# Each entry: field_key -> {pattern, description, severity}
MANDATORY_FIELDS = {
    "manufacturer_info": {
        "pattern": re.compile(
            r"(mfd\.?\s*by|manufactured\s*by|marketed\s*by|packed\s*by|imported\s*by)",
            re.IGNORECASE,
        ),
        "description": "Name & address of manufacturer/packer/importer (Rule 6(1)(a))",
        "severity": "critical",
    },
    "product_name": {
        # Heuristic: at least one line of mostly-alphabetic text near the top
        # of the label. Real check done in compliance_checker.py using OCR
        # line position, not just regex.
        "pattern": re.compile(r"[A-Za-z]{3,}"),
        "description": "Common/generic name of the commodity (Rule 6(1)(b))",
        "severity": "critical",
    },
    "net_quantity": {
        "pattern": re.compile(
            r"\b\d+\.?\d*\s?(g|gm|gms|gram|grams|kg|kgs|ml|mL|l|ltr|litre|litres|N)\b",
            re.IGNORECASE,
        ),
        "description": "Net quantity in standard units (Rule 6(1)(c))",
        "severity": "critical",
    },
    "mrp": {
        "pattern": re.compile(
            r"(₹|rs\.?|inr|mrp)\s?\d+(\.\d{1,2})?", re.IGNORECASE
        ),
        "description": "Maximum Retail Price, inclusive of all taxes (Rule 6(1)(e))",
        "severity": "critical",
    },
    "mfg_date": {
        "pattern": re.compile(
            r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*[\.\s-]?\d{2,4}"
            r"|\b\d{1,2}[\/\-]\d{2,4}\b",
            re.IGNORECASE,
        ),
        "description": "Month & year of manufacture/pre-packing/import (Rule 6(1)(d))",
        "severity": "critical",
    },
    "customer_care": {
        "pattern": re.compile(
            r"(customer\s*care|consumer\s*care|helpline|toll[\s-]?free)"
            r"|\b\d{10}\b|[\w\.-]+@[\w\.-]+\.\w+",
            re.IGNORECASE,
        ),
        "description": "Consumer care / complaint contact details (Rule 6(2))",
        "severity": "moderate",
    },
}

# Exemptions noted in the Rules (extend as your team verifies more)
EXEMPT_FROM_MFG_DATE = ["bidi", "incense stick", "agarbatti"]

# Ambiguous-attribution check (Explanation I under Rule 6(1)(a)):
# If a name/address appears WITHOUT a qualifying word like "mfd by" /
# "packed by", it is presumed to be the manufacturer's - flag this as
# a "needs review" note, not necessarily a hard violation.
QUALIFYING_WORDS = re.compile(
    r"(mfd\.?\s*by|manufactured\s*by|marketed\s*by|packed\s*by|imported\s*by)",
    re.IGNORECASE,
)
