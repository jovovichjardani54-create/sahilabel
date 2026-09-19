"""Reusable inspector-review record that never mutates automated results."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


VALID_INSPECTOR_DECISIONS = frozenset({"CONFIRMED", "OVERRIDDEN", "PENDING"})
DECISIONS_REQUIRING_REASON = frozenset({"CONFIRMED", "OVERRIDDEN"})


class InspectorReviewError(ValueError):
    """Raised when an inspector review cannot be recorded."""


class InspectorReview:
    """Audit record that preserves the original automated decision separately.

    The automated result is stored as ``automated_decision`` and is never
    replaced by ``inspector_decision``. Serialization always emits both.
    """

    def __init__(
        self,
        *,
        reviewer: str,
        automated_decision: Any,
        inspector_decision: str,
        reason: str | None = None,
        field_confirmations: dict[str, Any] | None = None,
        reviewed_at: datetime | None = None,
    ) -> None:
        reviewer_id = "" if reviewer is None else str(reviewer).strip()
        if not reviewer_id:
            raise InspectorReviewError("reviewer name or ID is required")

        decision = str(inspector_decision or "").strip().upper()
        if decision not in VALID_INSPECTOR_DECISIONS:
            raise InspectorReviewError(
                "inspector_decision must be CONFIRMED, OVERRIDDEN, or PENDING"
            )

        reason_text = reason.strip() if isinstance(reason, str) else reason
        if decision in DECISIONS_REQUIRING_REASON and not reason_text:
            raise InspectorReviewError("reviewer reason is required")

        if field_confirmations is not None and not isinstance(field_confirmations, dict):
            raise InspectorReviewError("field_confirmations must be a dictionary")

        self.reviewer = reviewer_id
        self.reviewed_at = reviewed_at or datetime.now(timezone.utc)
        self.automated_decision = deepcopy(automated_decision)
        self.inspector_decision = decision
        self.reason = reason_text or None
        self.field_confirmations = deepcopy(field_confirmations) if field_confirmations else None

    def to_dict(self) -> dict[str, Any]:
        """Return an audit-friendly snapshot without mutating stored values."""
        reviewed_at = self.reviewed_at
        timestamp = reviewed_at.isoformat() if isinstance(reviewed_at, datetime) else str(reviewed_at)
        return {
            "reviewer": self.reviewer,
            "reviewed_at": timestamp,
            "automated_decision": deepcopy(self.automated_decision),
            "inspector_decision": self.inspector_decision,
            "reason": self.reason,
            "field_confirmations": deepcopy(self.field_confirmations),
        }


def create_inspector_review(
    *,
    reviewer: str,
    automated_decision: Any,
    inspector_decision: str,
    reason: str | None = None,
    field_confirmations: dict[str, Any] | None = None,
    reviewed_at: datetime | None = None,
) -> InspectorReview:
    """Build a review record. The automated decision is copied, not rewritten."""
    return InspectorReview(
        reviewer=reviewer,
        automated_decision=automated_decision,
        inspector_decision=inspector_decision,
        reason=reason,
        field_confirmations=field_confirmations,
        reviewed_at=reviewed_at,
    )
