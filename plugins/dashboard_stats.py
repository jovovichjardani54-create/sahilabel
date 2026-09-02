"""
plugins/dashboard_stats.py
----------------------------
PLUGIN — Feature 4: Advanced Dashboard

Computes aggregate statistics from the existing HISTORY list (or later,
the database) without changing how history is stored. Called on-demand
by a new /dashboard/stats endpoint - the existing /history endpoint is
untouched.
"""

from collections import Counter
from datetime import datetime


def compute_dashboard_stats(history_records: list) -> dict:
    """
    Args:
        history_records: list of dicts, each shaped like the existing
            HISTORY entries in main.py (must include "overall_compliant",
            "timestamp", and ideally "report" with "fields").

    Returns:
        {
            "total_scanned": int,
            "compliant_count": int,
            "compliance_rate": float (0-100),
            "most_common_violations": [(field_key, count), ...],
            "monthly_trend": {"YYYY-MM": {"total": int, "compliant": int}}
        }
    """
    total = len(history_records)
    compliant = sum(1 for r in history_records if r.get("overall_compliant"))
    compliance_rate = round((compliant / total * 100), 1) if total else 0.0

    violation_counter = Counter()
    monthly_trend = {}

    for record in history_records:
        report = record.get("report")
        if report:
            for field_key, field_result in report.get("fields", {}).items():
                if not field_result.get("present"):
                    violation_counter[field_key] += 1

        ts = record.get("timestamp")
        if ts:
            try:
                month_key = datetime.fromisoformat(ts).strftime("%Y-%m")
            except ValueError:
                continue
            bucket = monthly_trend.setdefault(month_key, {"total": 0, "compliant": 0})
            bucket["total"] += 1
            if record.get("overall_compliant"):
                bucket["compliant"] += 1

    return {
        "total_scanned": total,
        "compliant_count": compliant,
        "compliance_rate": compliance_rate,
        "most_common_violations": violation_counter.most_common(5),
        "monthly_trend": monthly_trend,
    }
