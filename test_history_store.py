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
        self.assertIsNone(saved["product_name"])
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

    def test_retrieves_a_previously_saved_summary(self):
        self.store.save_report(
            "session-previous", "tea-label.png", report("PASS", mrp="PASS"),
            product_name="Masala Tea",
        )

        summary = self.store.get_report_summary("session-previous")

        self.assertEqual(summary["product_name"], "Masala Tea")
        self.assertEqual(summary["field_decision_summary"], {"mrp": "PASS"})

    def test_searches_product_name_filename_and_identifier(self):
        self.store.save_report(
            "tea-identifier", "tea-label.png", report("PASS", mrp="PASS"),
            created_at="2026-09-19T09:00:00+00:00",
            product_name="Masala Tea",
        )
        self.store.save_report(
            "coffee-identifier", "coffee-label.png", report("VIOLATION", mrp="VIOLATION"),
            created_at="2026-09-19T10:00:00+00:00",
        )

        self.assertEqual(
            [item["product_session_id"] for item in self.store.list_recent_reports()],
            ["coffee-identifier", "tea-identifier"],
        )
        self.assertEqual(
            [item["product_session_id"] for item in self.store.search_reports("TEA")],
            ["tea-identifier"],
        )
        self.assertEqual(
            [item["product_session_id"] for item in self.store.search_reports("identifier")],
            ["coffee-identifier", "tea-identifier"],
        )

    def test_filters_by_each_supported_status(self):
        for result in ("PASS", "REVIEW", "VIOLATION"):
            self.store.save_report(result.lower(), f"{result}.png", report(result))

        for result in ("PASS", "REVIEW", "VIOLATION"):
            self.assertEqual(
                [item["overall_result"] for item in self.store.search_reports(overall_result=result)],
                [result],
            )
        with self.assertRaises(ValueError):
            self.store.search_reports(overall_result="UNKNOWN")

    def test_filters_by_single_date_and_date_range(self):
        self.store.save_report(
            "first", "first.png", report("PASS"), created_at="2026-09-18T23:00:00+00:00"
        )
        self.store.save_report(
            "second", "second.png", report("REVIEW"), created_at="2026-09-19T12:00:00+00:00"
        )
        self.store.save_report(
            "third", "third.png", report("VIOLATION"), created_at="2026-09-20T01:00:00+00:00"
        )

        self.assertEqual(
            [item["product_session_id"] for item in self.store.search_reports(
                start_date="2026-09-19", end_date="2026-09-19"
            )],
            ["second"],
        )
        self.assertEqual(
            [item["product_session_id"] for item in self.store.search_reports(
                start_date="2026-09-19", end_date="2026-09-20"
            )],
            ["third", "second"],
        )

    def test_missing_rows_are_safe(self):
        self.assertIsNone(self.store.get_report("missing"))
        self.assertIsNone(self.store.get_report_summary("missing"))
        self.assertEqual(self.store.list_recent_reports(), [])
        self.assertEqual(self.store.search_reports("nothing"), [])
        self.assertEqual(self.store.analytics_counts(), {"PASS": 0, "REVIEW": 0, "VIOLATION": 0})

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

    def test_analytics_counts_multiple_records_and_persist_across_instances(self):
        self.store.save_report("pass-1", "pass.png", report("PASS"))
        self.store.save_report("pass-2", "pass-again.png", report("PASS"))
        self.store.save_report("review-1", "review.png", report("REVIEW"))
        self.store.save_report("violation-1", "violation.png", report("VIOLATION"))

        reopened_store = HistoryStore(self.database_path)

        self.assertEqual(
            reopened_store.analytics_counts(),
            {"PASS": 2, "REVIEW": 1, "VIOLATION": 1},
        )
        self.assertEqual(reopened_store.get_report_summary("pass-1")["filename"], "pass.png")

    def test_upgrades_an_existing_database_without_losing_records(self):
        legacy_database = Path(self.temp_dir.name) / "legacy.sqlite3"
        connection = sqlite3.connect(legacy_database)
        try:
            connection.execute(
                """
                CREATE TABLE report_history (
                    product_session_id TEXT PRIMARY KEY, filename TEXT NOT NULL,
                    created_at TEXT NOT NULL, overall_result TEXT NOT NULL,
                    field_decision_summary TEXT NOT NULL, reviewer_status TEXT
                )
                """
            )
            connection.execute(
                "INSERT INTO report_history VALUES (?, ?, ?, ?, ?, ?)",
                ("legacy", "legacy.png", "2026-09-19T10:00:00+00:00", "PASS", "{}", None),
            )
            connection.commit()
        finally:
            connection.close()

        upgraded_store = HistoryStore(legacy_database)

        self.assertEqual(upgraded_store.get_report_summary("legacy")["filename"], "legacy.png")
        self.assertIsNone(upgraded_store.get_report_summary("legacy")["product_name"])


if __name__ == "__main__":
    unittest.main()
