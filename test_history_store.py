import sqlite3
import tempfile
import unittest
from pathlib import Path

from history_store import HistoryStore


def report(result: str, **decisions: str) -> dict:
    return {
        "overall_result": result,
        "field_decisions": {
            field: {"status": status} for field, status in decisions.items()
        },
    }


class HistoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "history.sqlite3"
        self.store = HistoryStore(self.database_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_creates_database_and_saves_only_summary_metadata(self):
        saved = self.store.save_report(
            "session-1",
            "tea-label.png",
            report("REVIEW", mrp="PASS", net_quantity="REVIEW"),
            reviewer_status="pending",
            created_at="2026-09-19T10:00:00+00:00",
        )

        self.assertTrue(self.database_path.exists())
        self.assertEqual(saved["product_session_id"], "session-1")
        self.assertEqual(saved["overall_result"], "REVIEW")
        self.assertEqual(
            saved["field_decision_summary"],
            {"mrp": "PASS", "net_quantity": "REVIEW"},
        )
        self.assertEqual(saved["reviewer_status"], "pending")

        connection = sqlite3.connect(self.database_path)
        try:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(report_history)")
            }
        finally:
            connection.close()
        self.assertNotIn("image", columns)
        self.assertNotIn("pdf", columns)

    def test_lists_newest_reports_and_searches_filename_or_result(self):
        self.store.save_report(
            "first", "tea-label.png", report("PASS", mrp="PASS"),
            created_at="2026-09-19T09:00:00+00:00",
        )
        self.store.save_report(
            "second", "coffee-label.png", report("VIOLATION", mrp="VIOLATION"),
            created_at="2026-09-19T10:00:00+00:00",
        )

        self.assertEqual(
            [item["product_session_id"] for item in self.store.list_recent_reports()],
            ["second", "first"],
        )
        self.assertEqual(
            [item["product_session_id"] for item in self.store.search_reports("TEA")],
            ["first"],
        )
        self.assertEqual(
            [item["product_session_id"] for item in self.store.search_reports(overall_result="violation")],
            ["second"],
        )

    def test_missing_rows_are_safe(self):
        self.assertIsNone(self.store.get_report("missing"))
        self.assertEqual(self.store.list_recent_reports(), [])
        self.assertEqual(self.store.search_reports("nothing"), [])

    def test_uses_reviewer_status_from_report_when_present(self):
        saved = self.store.save_report(
            "session-2",
            "milk-label.png",
            {**report("PASS", mrp="PASS"), "reviewer_status": "approved"},
        )
        self.assertEqual(saved["reviewer_status"], "approved")

    def test_rejects_invalid_limits(self):
        with self.assertRaises(ValueError):
            self.store.list_recent_reports(0)
        with self.assertRaises(ValueError):
            self.store.search_reports(limit=True)


if __name__ == "__main__":
    unittest.main()
