"""API tests for the optional history analytics router.

These need FastAPI's test client.  They are skipped (not failed) in
environments where FastAPI is unavailable; the underlying service is covered
separately by ``test_history_analytics_service``.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from history_store import HistoryStore

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from history_analytics_router import create_history_analytics_router

    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the local environment
    FASTAPI_AVAILABLE = False

PREFIX = "/analytics/history"


def report(result: str, product: str | None = None) -> dict:
    decisions = {"mrp": {"status": result}}
    if product:
        decisions["generic_product_name"] = {"status": "PASS", "extracted_value": product}
    return {"overall_result": result, "field_decisions": decisions}


def make_client(store: HistoryStore, prefix: str = PREFIX) -> "TestClient":
    app = FastAPI()
    app.include_router(create_history_analytics_router(store, prefix=prefix))
    return TestClient(app)


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI is not installed")
class HistoryAnalyticsRouterTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "history.sqlite3"
        self.store = HistoryStore(self.database_path)
        self.store.save_report(
            "tea-1", "tea.png", report("PASS", "Masala Tea"),
            created_at="2026-09-18T09:00:00+00:00",
        )
        self.store.save_report(
            "tea-2", "tea-review.png", report("REVIEW", "Masala Tea"),
            created_at="2026-09-19T10:00:00+00:00",
        )
        self.store.save_report(
            "coffee-1", "coffee.png", report("VIOLATION"),
            created_at="2026-09-19T11:00:00+00:00",
        )
        self.client = make_client(self.store)

    def tearDown(self):
        self.temp_dir.cleanup()

    def ids(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        return [item["product_session_id"] for item in response.json()["items"]]

    def test_lists_reports_newest_first_with_total(self):
        response = self.client.get(f"{PREFIX}/reports")

        self.assertEqual(self.ids(response), ["coffee-1", "tea-2", "tea-1"])
        self.assertEqual(response.json()["total"], 3)
        self.assertEqual((response.json()["limit"], response.json()["offset"]), (50, 0))

    def test_searches_by_filename_product_and_identifier(self):
        for query, expected in [
            ("coffee", ["coffee-1"]),
            ("masala", ["tea-2", "tea-1"]),  # product name, derived from the report
            ("tea-review", ["tea-2"]),
            ("tea-1", ["tea-1"]),
            ("does-not-exist", []),
        ]:
            with self.subTest(query=query):
                self.assertEqual(
                    self.ids(self.client.get(f"{PREFIX}/reports", params={"query": query})),
                    expected,
                )

    def test_filters_by_status_and_date_range(self):
        for status in ("PASS", "REVIEW", "VIOLATION"):
            with self.subTest(status=status):
                items = self.client.get(f"{PREFIX}/reports", params={"status": status}).json()["items"]
                self.assertEqual(len(items), 1)
                self.assertEqual(items[0]["overall_result"], status)

        day = self.client.get(
            f"{PREFIX}/reports", params={"start_date": "2026-09-19", "end_date": "2026-09-19"}
        )
        self.assertEqual(self.ids(day), ["coffee-1", "tea-2"])
        self.assertEqual(day.json()["total"], 2)

    def test_pagination_is_stable_and_total_ignores_the_page(self):
        first = self.client.get(f"{PREFIX}/reports", params={"limit": 2, "offset": 0})
        second = self.client.get(f"{PREFIX}/reports", params={"limit": 2, "offset": 2})

        self.assertEqual(self.ids(first), ["coffee-1", "tea-2"])
        self.assertEqual(self.ids(second), ["tea-1"])
        self.assertEqual(first.json()["total"], 3)
        self.assertEqual(second.json()["total"], 3)

    def test_retrieves_one_report_and_unknown_id_is_404(self):
        found = self.client.get(f"{PREFIX}/reports/tea-1")
        missing = self.client.get(f"{PREFIX}/reports/missing")

        self.assertEqual(found.status_code, 200)
        self.assertEqual(found.json()["product_name"], "Masala Tea")
        self.assertEqual(found.json()["field_decision_summary"], {
            "generic_product_name": "PASS", "mrp": "PASS",
        })
        self.assertEqual(missing.status_code, 404)

    def test_analytics_reports_counts_rates_and_daily_trends(self):
        data = self.client.get(f"{PREFIX}/analytics").json()

        self.assertEqual(data["total_inspections"], 3)
        self.assertEqual(data["counts"], {"PASS": 1, "REVIEW": 1, "VIOLATION": 1})
        self.assertAlmostEqual(data["review_rate"], 1 / 3)
        self.assertAlmostEqual(data["violation_rate"], 1 / 3)
        by_day = {row["date"]: row for row in data["daily_trends"]}
        self.assertEqual(by_day["2026-09-19"]["total"], 2)
        self.assertEqual(by_day["2026-09-18"]["total"], 1)

    def test_analytics_respects_date_filters(self):
        data = self.client.get(
            f"{PREFIX}/analytics", params={"start_date": "2026-09-19", "end_date": "2026-09-19"}
        ).json()

        self.assertEqual(data["total_inspections"], 2)
        self.assertEqual(data["counts"], {"PASS": 0, "REVIEW": 1, "VIOLATION": 1})

    def test_invalid_filters_return_controlled_client_errors(self):
        bad_requests = [
            (f"{PREFIX}/reports", {"status": "BOGUS"}),
            (f"{PREFIX}/reports", {"start_date": "not-a-date"}),
            (f"{PREFIX}/reports", {"end_date": "2026-13-45"}),
            (f"{PREFIX}/reports", {"start_date": "2026-09-20", "end_date": "2026-09-01"}),
            (f"{PREFIX}/reports", {"limit": 0}),
            (f"{PREFIX}/reports", {"limit": 101}),
            (f"{PREFIX}/reports", {"offset": -1}),
            (f"{PREFIX}/reports", {"limit": "many"}),
            (f"{PREFIX}/analytics", {"trend_days": 0}),
            (f"{PREFIX}/analytics", {"trend_days": 9999}),
            (f"{PREFIX}/analytics", {"start_date": "yesterday"}),
        ]
        for path, params in bad_requests:
            with self.subTest(path=path, params=params):
                response = self.client.get(path, params=params)
                self.assertIn(response.status_code, (400, 422), response.text)
                self.assertIn("detail", response.json())

    def test_query_text_is_treated_as_data_not_sql(self):
        # None of the seeded records contain these characters, so a literal
        # match returns nothing (wildcard behaviour would return everything).
        for hostile in ("'; DROP TABLE report_history; --", "%", "_", "\\"):
            with self.subTest(query=hostile):
                response = self.client.get(f"{PREFIX}/reports", params={"query": hostile})
                self.assertEqual(self.ids(response), [])
                self.assertEqual(response.json()["total"], 0)
        self.assertEqual(self.client.get(f"{PREFIX}/analytics").json()["total_inspections"], 3)

    def test_empty_database_is_safe(self):
        empty = make_client(HistoryStore(Path(self.temp_dir.name) / "empty.sqlite3"))

        listing = empty.get(f"{PREFIX}/reports")
        analytics = empty.get(f"{PREFIX}/analytics")

        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.json()["items"], [])
        self.assertEqual(listing.json()["total"], 0)
        self.assertEqual(analytics.status_code, 200)
        self.assertEqual(analytics.json()["total_inspections"], 0)
        self.assertEqual(analytics.json()["counts"], {"PASS": 0, "REVIEW": 0, "VIOLATION": 0})
        self.assertEqual(analytics.json()["review_rate"], 0.0)
        self.assertEqual(analytics.json()["violation_rate"], 0.0)
        self.assertEqual(empty.get(f"{PREFIX}/reports/anything").status_code, 404)

    def test_data_and_analytics_survive_an_application_restart(self):
        before = self.client.get(f"{PREFIX}/analytics").json()
        self.client.close()
        del self.client, self.store  # discard every in-process object

        # A brand-new store, service, router, and app on the same SQLite file,
        # with no in-memory HISTORY list anywhere.
        restarted = make_client(HistoryStore(self.database_path))

        self.assertEqual(restarted.get(f"{PREFIX}/analytics").json(), before)
        self.assertEqual(
            self.ids(restarted.get(f"{PREFIX}/reports")), ["coffee-1", "tea-2", "tea-1"]
        )
        self.assertEqual(restarted.get(f"{PREFIX}/reports/tea-2").json()["overall_result"], "REVIEW")

    def test_records_written_after_restart_are_added_to_existing_history(self):
        restarted_store = HistoryStore(self.database_path)
        restarted_store.save_report(
            "after-restart", "new.png", report("PASS"), created_at="2026-09-20T08:00:00+00:00"
        )
        restarted = make_client(restarted_store)

        self.assertEqual(restarted.get(f"{PREFIX}/analytics").json()["total_inspections"], 4)
        self.assertEqual(self.ids(restarted.get(f"{PREFIX}/reports"))[0], "after-restart")

    def test_legacy_database_records_stay_readable_through_the_api(self):
        legacy_path = Path(self.temp_dir.name) / "legacy.sqlite3"
        connection = sqlite3.connect(legacy_path)
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
                ("legacy-1", "old-label.png", "2026-09-01T10:00:00+00:00", "VIOLATION",
                 '{"mrp": "VIOLATION"}', "CONFIRMED"),
            )
            connection.commit()
        finally:
            connection.close()

        legacy = make_client(HistoryStore(legacy_path))  # triggers the in-place upgrade

        record = legacy.get(f"{PREFIX}/reports/legacy-1").json()
        self.assertEqual(record["filename"], "old-label.png")
        self.assertIsNone(record["product_name"])
        self.assertEqual(record["reviewer_status"], "CONFIRMED")
        self.assertEqual(self.ids(legacy.get(f"{PREFIX}/reports", params={"query": "old-label"})), ["legacy-1"])
        self.assertEqual(legacy.get(f"{PREFIX}/analytics").json()["counts"]["VIOLATION"], 1)

    def test_sqlite_holds_no_binary_image_or_pdf_content(self):
        connection = sqlite3.connect(self.database_path)
        try:
            columns = connection.execute("PRAGMA table_info(report_history)").fetchall()
            blob_values = connection.execute(
                "SELECT COUNT(*) FROM report_history WHERE "
                + " OR ".join(f"typeof({column[1]}) = 'blob'" for column in columns)
            ).fetchone()[0]
        finally:
            connection.close()

        self.assertFalse([c[1] for c in columns if (c[2] or "").upper() == "BLOB"])
        self.assertEqual(blob_values, 0)

    def test_router_prefix_can_be_customised(self):
        custom = make_client(self.store, prefix="/hub")

        self.assertEqual(custom.get("/hub/analytics").status_code, 200)
        self.assertEqual(custom.get(f"{PREFIX}/analytics").status_code, 404)


if __name__ == "__main__":
    unittest.main()
