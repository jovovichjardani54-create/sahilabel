"""Persistent query and analytics service backed solely by ``HistoryStore``."""

from __future__ import annotations

from typing import Any

from history_store import HistoryStore


class HistoryAnalyticsService:
    """Provide Central Analytics Hub data without relying on in-memory records."""

    def __init__(self, history_store: HistoryStore) -> None:
        self.history_store = history_store

    def list_reports(
        self,
        query: str | None = None,
        status: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Return a stable paginated result set and matching total."""
        return {
            "items": self.history_store.list_reports(
                query=query,
                overall_result=status,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                offset=offset,
            ),
            "total": self.history_store.count_reports(
                query=query,
                overall_result=status,
                start_date=start_date,
                end_date=end_date,
            ),
            "limit": limit,
            "offset": offset,
        }

    def get_report(self, product_session_id: str) -> dict[str, Any] | None:
        """Return one compact report summary, or ``None`` when it is absent."""
        return self.history_store.get_report_summary(product_session_id)

    def analytics(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        trend_days: int = 7,
    ) -> dict[str, Any]:
        """Return persisted totals, rates, counts, and daily trends."""
        return self.history_store.analytics_summary(start_date, end_date, trend_days)
