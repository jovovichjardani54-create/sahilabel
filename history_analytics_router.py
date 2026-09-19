"""Optional FastAPI router for the persistent Central Analytics Hub.

Register it with ``app.include_router(create_history_analytics_router(HISTORY_STORE))``.
The default prefix intentionally avoids the existing ``/history`` endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from history_analytics_service import HistoryAnalyticsService
from history_store import HistoryStore


def create_history_analytics_router(
    history_store: HistoryStore, prefix: str = "/analytics/history"
) -> APIRouter:
    """Create an independently registerable router backed by SQLite history."""
    service = HistoryAnalyticsService(history_store)
    router = APIRouter(prefix=prefix, tags=["history-analytics"])

    @router.get("/reports")
    def list_reports(
        query: str | None = None,
        status: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ):
        try:
            return service.list_reports(query, status, start_date, end_date, limit, offset)
        except (TypeError, ValueError) as error:
            raise HTTPException(400, str(error)) from error

    @router.get("/analytics")
    def get_analytics(
        start_date: str | None = None,
        end_date: str | None = None,
        trend_days: int = 7,
    ):
        try:
            return service.analytics(start_date, end_date, trend_days)
        except (TypeError, ValueError) as error:
            raise HTTPException(400, str(error)) from error

    @router.get("/reports/{product_session_id}")
    def get_report(product_session_id: str):
        report = service.get_report(product_session_id)
        if report is None:
            raise HTTPException(404, "History record not found")
        return report

    return router
