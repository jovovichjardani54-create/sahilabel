"""Inspector-review workflow with an audit trail that never rewrites AI results.

AI extraction and automated rule checks are evidence only. They never become a
final legal decision. Human inspection is required solely for REVIEW cases;
PASS and VIOLATION/FAIL results stay as-is and do not enter this workflow.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


VALID_INSPECTOR_DECISIONS = frozenset({"CONFIRMED", "OVERRIDDEN", "PENDING"})
REVIEW_REQUIRED_STATUSES = frozenset({"REVIEW"})
PASS_STATUSES = frozenset({"PASS"})
VIOLATION_STATUSES = frozenset({"VIOLATION", "FAIL"})
MISSING_REVIEW_MESSAGE = "No inspector review exists for this item"
REVIEW_NOT_REQUIRED_MESSAGE = (
    "PASS and VIOLATION results do not require inspector review"
)


class InspectorReviewError(ValueError):
    """Raised when an inspector review cannot be recorded."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def system_status(automated_decision: Any) -> str:
    """Return the automated legal-status label without altering the original."""
    if isinstance(automated_decision, dict):
        for key in ("overall_status", "status", "automated_status", "overall"):
            raw = automated_decision.get(key)
            if raw is not None and str(raw).strip():
                return str(raw).strip().upper()
    if automated_decision is None:
        return ""
    return str(automated_decision).strip().upper()


def requires_human_inspection(automated_decision: Any) -> bool:
    """Only REVIEW cases need an inspector. PASS and VIOLATION stay unchanged."""
    return system_status(automated_decision) in REVIEW_REQUIRED_STATUSES


def _normalize_decision(inspector_decision: Any) -> str:
    decision = str(inspector_decision or "").strip().upper()
    if decision not in VALID_INSPECTOR_DECISIONS:
        raise InspectorReviewError(
            "inspector_decision must be CONFIRMED, OVERRIDDEN, or PENDING"
        )
    return decision


def _require_inspector_identity(inspector_id: Any, reviewer_name: Any) -> tuple[str, str]:
    inspector = _clean_text(inspector_id)
    name = _clean_text(reviewer_name)
    if not inspector:
        raise InspectorReviewError("inspector ID is required")
    if not name:
        raise InspectorReviewError("reviewer name is required")
    return inspector, name


class InspectorReview:
    """Audit record that preserves the original automated decision separately.

    ``automated_decision`` is copied at creation time and is never replaced by
    ``inspector_decision``. Serialization always emits both.
    """

    def __init__(
        self,
        *,
        reviewer: str | None = None,
        automated_decision: Any,
        inspector_decision: str = "PENDING",
        reason: str | None = None,
        field_confirmations: dict[str, Any] | None = None,
        reviewed_at: datetime | None = None,
        inspector_id: str | None = None,
        reviewer_name: str | None = None,
        review_id: str | None = None,
    ) -> None:
        decision = _normalize_decision(inspector_decision)
        reason_text = _clean_text(reason)
        if decision == "OVERRIDDEN" and not reason_text:
            raise InspectorReviewError("override reason is required")

        if field_confirmations is not None and not isinstance(field_confirmations, dict):
            raise InspectorReviewError("field_confirmations must be a dictionary")

        identity = _clean_text(inspector_id) or _clean_text(reviewer)
        name = _clean_text(reviewer_name) or _clean_text(reviewer)
        if decision != "PENDING":
            identity, name = _require_inspector_identity(
                identity or inspector_id, name or reviewer_name
            )
        elif not identity and not name:
            raise InspectorReviewError("inspector ID or reviewer name is required")

        created_at = reviewed_at or _now()
        self.review_id = _clean_text(review_id)
        self.inspector_id = identity
        self.reviewer_name = name
        self.reviewer = name or identity
        self.created_at = created_at
        self.reviewed_at = created_at
        # Frozen copy: confirm/override must never replace this original result.
        self._automated_decision = deepcopy(automated_decision)
        self.inspector_decision = decision
        self.reason = reason_text
        self.field_confirmations = deepcopy(field_confirmations) if field_confirmations else None
        self.audit_trail: list[dict[str, Any]] = [
            self._event("created", created_at, reason=reason_text)
        ]

    @property
    def automated_decision(self) -> Any:
        return deepcopy(self._automated_decision)

    def _event(self, action: str, at: datetime, reason: str | None = None) -> dict[str, Any]:
        return {
            "action": action,
            "inspector_decision": self.inspector_decision,
            "inspector_id": self.inspector_id,
            "reviewer_name": self.reviewer_name,
            "reason": reason,
            "at": _iso(at),
            "automated_decision": self.automated_decision,
        }

    def confirm(
        self,
        *,
        inspector_id: str,
        reviewer_name: str,
        reason: str | None = None,
        confirmed_at: datetime | None = None,
    ) -> InspectorReview:
        self.inspector_id, self.reviewer_name = _require_inspector_identity(
            inspector_id, reviewer_name
        )
        self.reviewer = self.reviewer_name
        self.inspector_decision = "CONFIRMED"
        self.reason = _clean_text(reason)
        self.reviewed_at = confirmed_at or _now()
        self.audit_trail.append(self._event("confirmed", self.reviewed_at, self.reason))
        return self

    def override(
        self,
        *,
        inspector_id: str,
        reviewer_name: str,
        reason: str,
        overridden_at: datetime | None = None,
    ) -> InspectorReview:
        reason_text = _clean_text(reason)
        if not reason_text:
            raise InspectorReviewError("override reason is required")
        self.inspector_id, self.reviewer_name = _require_inspector_identity(
            inspector_id, reviewer_name
        )
        self.reviewer = self.reviewer_name
        self.inspector_decision = "OVERRIDDEN"
        self.reason = reason_text
        self.reviewed_at = overridden_at or _now()
        self.audit_trail.append(self._event("overridden", self.reviewed_at, reason_text))
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return an audit-friendly snapshot without mutating stored values."""
        return {
            "review_id": self.review_id,
            "inspector_id": self.inspector_id,
            "reviewer_name": self.reviewer_name,
            "reviewer": self.reviewer,
            "created_at": _iso(self.created_at),
            "reviewed_at": _iso(self.reviewed_at),
            "automated_decision": self.automated_decision,
            "system_status": system_status(self._automated_decision),
            "inspector_decision": self.inspector_decision,
            "reason": self.reason,
            "field_confirmations": deepcopy(self.field_confirmations),
            "audit_trail": deepcopy(self.audit_trail),
            "review_required": True,
            "found": True,
        }


def create_inspector_review(
    *,
    reviewer: str | None = None,
    automated_decision: Any,
    inspector_decision: str,
    reason: str | None = None,
    field_confirmations: dict[str, Any] | None = None,
    reviewed_at: datetime | None = None,
    inspector_id: str | None = None,
    reviewer_name: str | None = None,
    review_id: str | None = None,
) -> InspectorReview:
    """Build a review record. The automated decision is copied, not rewritten."""
    return InspectorReview(
        reviewer=reviewer,
        automated_decision=automated_decision,
        inspector_decision=inspector_decision,
        reason=reason,
        field_confirmations=field_confirmations,
        reviewed_at=reviewed_at,
        inspector_id=inspector_id,
        reviewer_name=reviewer_name,
        review_id=review_id,
    )


def missing_review_response(review_id: str | None = None) -> dict[str, Any]:
    return {
        "found": False,
        "review_id": review_id,
        "review": None,
        "inspector_decision": None,
        "automated_decision": None,
        "reason": None,
        "audit_trail": [],
        "message": MISSING_REVIEW_MESSAGE,
    }


class InspectorReviewRegistry:
    """In-memory pending/completed reviews keyed by inspection id."""

    def __init__(self) -> None:
        self._reviews: dict[str, InspectorReview] = {}

    def clear(self) -> None:
        self._reviews.clear()

    def create_pending(
        self,
        review_id: str,
        automated_decision: Any,
        *,
        inspector_id: str | None = None,
        reviewer_name: str | None = None,
    ) -> dict[str, Any]:
        item_id = _clean_text(review_id)
        if not item_id:
            raise InspectorReviewError("review id is required")
        if not requires_human_inspection(automated_decision):
            return {
                "created": False,
                "found": False,
                "review_id": item_id,
                "review_required": False,
                "inspector_decision": None,
                "automated_decision": deepcopy(automated_decision),
                "message": REVIEW_NOT_REQUIRED_MESSAGE,
            }
        review = InspectorReview(
            review_id=item_id,
            automated_decision=automated_decision,
            inspector_decision="PENDING",
            inspector_id=inspector_id,
            reviewer_name=reviewer_name,
            reviewer=inspector_id or reviewer_name,
        )
        self._reviews[item_id] = review
        payload = review.to_dict()
        payload["created"] = True
        payload["message"] = "Inspector review is pending"
        return payload

    def get(self, review_id: str) -> dict[str, Any]:
        item_id = _clean_text(review_id)
        if not item_id or item_id not in self._reviews:
            return missing_review_response(item_id)
        payload = self._reviews[item_id].to_dict()
        payload["review"] = deepcopy(payload)
        payload["message"] = None
        return payload

    def _require(self, review_id: str) -> InspectorReview:
        item_id = _clean_text(review_id)
        if not item_id or item_id not in self._reviews:
            raise InspectorReviewError(MISSING_REVIEW_MESSAGE)
        return self._reviews[item_id]

    def confirm(
        self,
        review_id: str,
        *,
        inspector_id: str,
        reviewer_name: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        review = self._require(review_id)
        review.confirm(inspector_id=inspector_id, reviewer_name=reviewer_name, reason=reason)
        return review.to_dict()

    def override(
        self,
        review_id: str,
        *,
        inspector_id: str,
        reviewer_name: str,
        reason: str,
    ) -> dict[str, Any]:
        review = self._require(review_id)
        review.override(inspector_id=inspector_id, reviewer_name=reviewer_name, reason=reason)
        return review.to_dict()


REGISTRY = InspectorReviewRegistry()


def create_pending_review(
    review_id: str,
    automated_decision: Any,
    *,
    inspector_id: str | None = None,
    reviewer_name: str | None = None,
) -> dict[str, Any]:
    return REGISTRY.create_pending(
        review_id,
        automated_decision,
        inspector_id=inspector_id,
        reviewer_name=reviewer_name,
    )


def get_review_status(review_id: str) -> dict[str, Any]:
    return REGISTRY.get(review_id)


def confirm_review(
    review_id: str,
    *,
    inspector_id: str,
    reviewer_name: str,
    reason: str | None = None,
) -> dict[str, Any]:
    return REGISTRY.confirm(
        review_id,
        inspector_id=inspector_id,
        reviewer_name=reviewer_name,
        reason=reason,
    )


def override_review(
    review_id: str,
    *,
    inspector_id: str,
    reviewer_name: str,
    reason: str,
) -> dict[str, Any]:
    return REGISTRY.override(
        review_id,
        inspector_id=inspector_id,
        reviewer_name=reviewer_name,
        reason=reason,
    )
