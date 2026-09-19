"""Standalone inspector-review HTTP router and request handlers.

These handlers expose the inspector workflow without changing OCR, extraction,
quality-gate, or legal-rule code. FastAPI wiring lives here so `main.py` only
needs to include the router.
"""

from __future__ import annotations

from typing import Any

from inspector_review import (
    InspectorReviewError,
    confirm_review,
    create_pending_review,
    get_review_status,
    list_pending_reviews,
    override_review,
)


def _error(message: str, status_code: int = 400) -> tuple[int, dict[str, Any]]:
    return status_code, {
        "ok": False,
        "error": message,
        "found": False,
        "created": False,
    }


def _ok(payload: Any, status_code: int = 200) -> tuple[int, Any]:
    return status_code, payload


def handle_create_pending_review(body: dict[str, Any]) -> tuple[int, Any]:
    try:
        payload = create_pending_review(
            str(body.get("review_id") or ""),
            body.get("automated_decision"),
            inspector_id=body.get("inspector_id"),
            reviewer_name=body.get("reviewer_name"),
        )
    except InspectorReviewError as exc:
        return _error(str(exc), 400)
    if not payload.get("created"):
        return _ok(payload, 200)
    return _ok(payload, 201)


def handle_list_pending_reviews() -> tuple[int, Any]:
    pending = list_pending_reviews()
    return _ok({"count": len(pending), "reviews": pending})


def handle_get_review(review_id: str) -> tuple[int, Any]:
    payload = get_review_status(review_id)
    if not payload.get("found"):
        return _ok(payload, 200)
    return _ok(payload)


def handle_confirm_review(review_id: str, body: dict[str, Any]) -> tuple[int, Any]:
    try:
        payload = confirm_review(
            review_id,
            inspector_id=body.get("inspector_id"),
            reviewer_name=body.get("reviewer_name"),
            reason=body.get("reason"),
            field_confirmations=body.get("field_confirmations"),
        )
    except InspectorReviewError as exc:
        return _error(str(exc), 400)
    return _ok(payload)


def handle_override_review(review_id: str, body: dict[str, Any]) -> tuple[int, Any]:
    try:
        payload = override_review(
            review_id,
            inspector_id=body.get("inspector_id"),
            reviewer_name=body.get("reviewer_name"),
            reason=body.get("reason"),
            field_confirmations=body.get("field_confirmations"),
        )
    except InspectorReviewError as exc:
        return _error(str(exc), 400)
    return _ok(payload)


try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - FastAPI is present when the app runs
    router = None
else:

    class CreatePendingReviewBody(BaseModel):
        review_id: str
        automated_decision: Any
        inspector_id: str | None = None
        reviewer_name: str | None = None

    class CompleteReviewBody(BaseModel):
        inspector_id: str
        reviewer_name: str
        reason: str | None = None
        field_confirmations: dict[str, Any] | None = Field(default=None)

    class OverrideReviewBody(BaseModel):
        inspector_id: str
        reviewer_name: str
        reason: str
        field_confirmations: dict[str, Any] | None = Field(default=None)

    router = APIRouter(prefix="/inspector-reviews", tags=["inspector-review"])

    def _as_dict(body: Any) -> dict[str, Any]:
        if hasattr(body, "model_dump"):
            return body.model_dump()
        return body.dict()

    def _raise_if_error(status_code: int, payload: Any) -> Any:
        if status_code >= 400:
            raise HTTPException(status_code, payload.get("error") if isinstance(payload, dict) else str(payload))
        return payload

    @router.post("")
    @router.post("/")
    def create_pending_review_endpoint(body: CreatePendingReviewBody):
        status_code, payload = handle_create_pending_review(_as_dict(body))
        return _raise_if_error(status_code, payload)

    @router.get("")
    @router.get("/")
    def list_pending_reviews_endpoint():
        status_code, payload = handle_list_pending_reviews()
        return _raise_if_error(status_code, payload)

    @router.get("/{review_id}")
    def get_review_endpoint(review_id: str):
        status_code, payload = handle_get_review(review_id)
        return _raise_if_error(status_code, payload)

    @router.post("/{review_id}/confirm")
    def confirm_review_endpoint(review_id: str, body: CompleteReviewBody):
        status_code, payload = handle_confirm_review(review_id, _as_dict(body))
        return _raise_if_error(status_code, payload)

    @router.post("/{review_id}/override")
    def override_review_endpoint(review_id: str, body: OverrideReviewBody):
        status_code, payload = handle_override_review(review_id, _as_dict(body))
        return _raise_if_error(status_code, payload)
