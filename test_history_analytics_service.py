import tempfile
import unittest
from pathlib import Path

from history_analytics_service import HistoryAnalyticsService
from history_store import HistoryStore


def report(result: str) -> dict:
    return {"overall_result": result, "field_decisions": {}}


class HistoryAnalyticsServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = HistoryStore(Path(self.temp_dir.name) / "history.sqlite3")
        self.service = HistoryAnalyticsService(self.store)
        self.store.save_report(
            "tea-1", "tea.png", report("PASS"), product_name="Masala Tea",
            created_at="2026-09-18T09:00:00+00:00",
        )
        self.store.save_report(
            "tea-2", "tea-review.png", report("REVIEW"), product_name="Masala Tea",
            created_at="2026-09-19T10:00:00+00:00",
        )
        self.store.save_report(
            "coffee-1", "coffee.png", report("VIOLATION"),
            created_at="2026-09-19T11:00:00+00:00",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_lists_paginated_reports_and_total_from_sqlite(self):
        page = self.service.list_reports(query="tea", limit=1, offset=1)

        self.assertEqual(page["total"], 2)
        self.assertEqual(page["limit"], 1)
        self.assertEqual(page["offset"], 1)
        self.assertEqual([item["product_session_id"] for item in page["items"]], ["tea-1"])

    def test_retrieves_report_and_empty_results_are_safe(self):
        self.assertEqual(self.service.get_report("coffee-1")["overall_result"], "VIOLATION")
        self.assertIsNone(self.service.get_report("missing"))
        self.assertEqual(self.service.list_reports(query="absent")["items"], [])

    def test_analytics_uses_persisted_records_and_daily_trends(self):
        analytics = self.service.analytics(start_date="2026-09-19", trend_days=2)

        self.assertEqual(analytics["total_inspections"], 2)
        self.assertEqual(analytics["counts"], {"PASS": 0, "REVIEW": 1, "VIOLATION": 1})
        self.assertEqual(analytics["review_rate"], 0.5)
        self.assertEqual(analytics["violation_rate"], 0.5)
        self.assertEqual(analytics["daily_trends"][0]["date"], "2026-09-19")

    def test_invalid_filters_are_controlled(self):
        with self.assertRaises(ValueError):
            self.service.list_reports(status="NOT_A_STATUS")
        with self.assertRaises(ValueError):
            self.service.list_reports(start_date="2026-09-20", end_date="2026-09-19")
        with self.assertRaises(ValueError):
            self.service.list_reports(limit=101)

    def test_empty_database_analytics_are_safe(self):
        empty_store = HistoryStore(Path(self.temp_dir.name) / "empty.sqlite3")
        analytics = HistoryAnalyticsService(empty_store).analytics()

        self.assertEqual(analytics["total_inspections"], 0)
        self.assertEqual(analytics["counts"], {"PASS": 0, "REVIEW": 0, "VIOLATION": 0})
        self.assertEqual(analytics["review_rate"], 0.0)
        self.assertEqual(analytics["violation_rate"], 0.0)
        self.assertEqual(analytics["daily_trends"], [])


if __name__ == "__main__":
    unittest.main()
