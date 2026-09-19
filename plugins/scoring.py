"""
plugins/scoring.py
--------------------
PLUGIN — Feature 3: Compliance Score

Takes the EXISTING report dict produced by compliance_checker.check_fields()
and derives a 0-100 score plus category-wise breakdown. Does not modify
compliance_checker.py at all — this is a pure read-only transformation
that can be enabled/disabled independently.

Scoring model (documented so you can defend it to judges):
  - Each "critical" field present = worth points proportional to how many
    critical fields exist (they sum to 80 total).
  - Each "moderate" field present = worth points from the remaining 20.
  - Every readability flag deducts a fixed penalty (5 pts each, capped).
  - Every "notes" entry (e.g. ambiguous attribution) deducts a small
    penalty (2 pts each, capped) since it's a soft warning, not a hard miss.

This is intentionally simple and explainable — a judge can ask "why 72?"
and you can answer in one sentence, which matters more than a fancier
weighting scheme you can't justify on stage.
"""

CRITICAL_POOL = 80   # total points split across all "critical" fields
MODERATE_POOL = 20   # total points split across all "moderate" fields
READABILITY_PENALTY = 5
NOTE_PENALTY = 2
MAX_READABILITY_DEDUCTION = 20
MAX_NOTE_DEDUCTION = 10


def compute_score(report: dict) -> dict:
    """
    Args:
        report: the dict returned by compliance_checker.check_fields()
                (unchanged shape — fields/readability_flags/notes)

    Returns:
        {
            "score": int (0-100),
            "grade": str,
            "category_breakdown": {field_key: {"points_earned", "points_possible"}},
            "deductions": {"readability": int, "notes": int}
        }
    """
    # REVIEW means the evidence or image quality needs a human decision.  A
    # numeric zero and grade F imply a failed legal check, so neither is a
    # meaningful score until that review is complete.
    if report.get("overall_result") == "REVIEW":
        return {
            "score": None,
            "grade": None,
            "pending_review": True,
            "message": "Pending review",
            "category_breakdown": {},
            "deductions": {"readability": 0, "notes": 0},
        }

    fields = report["fields"]

    critical_keys = [k for k, v in fields.items() if v["severity"] == "critical"]
    moderate_keys = [k for k, v in fields.items() if v["severity"] == "moderate"]

    critical_each = CRITICAL_POOL / len(critical_keys) if critical_keys else 0
    moderate_each = MODERATE_POOL / len(moderate_keys) if moderate_keys else 0

    category_breakdown = {}
    base_score = 0.0

    for key in critical_keys:
        earned = critical_each if fields[key]["present"] else 0
        base_score += earned
        category_breakdown[key] = {
            "points_earned": round(earned, 1),
            "points_possible": round(critical_each, 1),
        }

    for key in moderate_keys:
        earned = moderate_each if fields[key]["present"] else 0
        base_score += earned
        category_breakdown[key] = {
            "points_earned": round(earned, 1),
            "points_possible": round(moderate_each, 1),
        }

    readability_deduction = min(
        len(report.get("readability_flags", [])) * READABILITY_PENALTY,
        MAX_READABILITY_DEDUCTION,
    )
    notes_deduction = min(
        len(report.get("notes", [])) * NOTE_PENALTY, MAX_NOTE_DEDUCTION
    )

    final_score = max(0, round(base_score - readability_deduction - notes_deduction))

    if final_score >= 90:
        grade = "A"
    elif final_score >= 75:
        grade = "B"
    elif final_score >= 60:
        grade = "C"
    elif final_score >= 40:
        grade = "D"
    else:
        grade = "F"

    return {
        "score": final_score,
        "grade": grade,
        "pending_review": False,
        "category_breakdown": category_breakdown,
        "deductions": {
            "readability": readability_deduction,
            "notes": notes_deduction,
        },
    }
