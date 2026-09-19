"""Small SQLite-backed storage for compliance report history.

The store deliberately keeps only report metadata and a compact field-decision
summary.  Uploaded images and generated PDFs remain on the filesystem.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


class HistoryStore:
    """Persist and retrieve lightweight compliance report summaries."""

    def __init__(self, database_path: str | Path = "history.db") -> None:
        self.database_path = str(database_path)
        self._create_table()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _create_table(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS report_history (
                    product_session_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    product_name TEXT,
                    created_at TEXT NOT NULL,
                    overall_result TEXT NOT NULL,
                    field_decision_summary TEXT NOT NULL,
                    reviewer_status TEXT
                )
                """
            )
            # Existing installations predate product_name.  Keep their data
            # usable by adding the nullable column in place rather than
            # rebuilding the table.
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(report_history)")
            }
            if "product_name" not in columns:
                connection.execute("ALTER TABLE report_history ADD COLUMN product_name TEXT")
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_report_history_created_at
                ON report_history (created_at DESC)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_report_history_product_name
                ON report_history (product_name)
                """
            )

    def save_report(
        self,
        product_session_id: str,
        filename: str,
        report: dict[str, Any],
        reviewer_status: str | None = None,
        created_at: str | None = None,
        product_name: str | None = None,
    ) -> dict[str, Any]:
        """Save a report summary and return the stored representation.

        ``report`` is expected to follow the existing compliance-checker
        structure.  Only each field decision's status is retained.
        """
        if not product_session_id:
            raise ValueError("product_session_id is required")
        if not filename:
            raise ValueError("filename is required")
        if not isinstance(report, dict):
            raise TypeError("report must be a dictionary")

        overall_result = report.get("overall_result")
        if overall_result is None:
            overall_result = "PASS" if report.get("overall_compliant") else "VIOLATION"
        overall_result = str(overall_result)
        summary = _field_decision_summary(report)
        if reviewer_status is None:
            reviewer_status = report.get("reviewer_status")
        if product_name is None:
            product_name = report.get("product_name")
        product_name = str(product_name) if product_name else None
        created_at = created_at or datetime.now(timezone.utc).isoformat()

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO report_history (
                    product_session_id, filename, product_name, created_at, overall_result,
                    field_decision_summary, reviewer_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(product_session_id) DO UPDATE SET
                    filename = excluded.filename,
                    product_name = excluded.product_name,
                    created_at = excluded.created_at,
                    overall_result = excluded.overall_result,
                    field_decision_summary = excluded.field_decision_summary,
                    reviewer_status = excluded.reviewer_status
                """,
                (
                    product_session_id,
                    filename,
                    product_name,
                    created_at,
                    overall_result,
                    json.dumps(summary, sort_keys=True),
                    reviewer_status,
                ),
            )

        return self.get_report(product_session_id) or {}

    def list_recent_reports(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return newest report summaries first."""
        limit = _valid_limit(limit)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT product_session_id, filename, created_at, overall_result,
                       product_name, field_decision_summary, reviewer_status
                FROM report_history
                ORDER BY created_at DESC, product_session_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_row_to_report(row) for row in rows]

    def search_reports(
        self,
        filename_query: str | None = None,
        overall_result: str | None = None,
        limit: int = 50,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search summaries by product name, filename, or identifier.

        Dates accept ISO timestamps or ``YYYY-MM-DD`` values.  A date-only
        end date includes the whole calendar day.
        """
        limit = _valid_limit(limit)
        clauses: list[str] = []
        parameters: list[Any] = []
        if filename_query:
            clauses.append(
                "(filename LIKE ? COLLATE NOCASE OR product_name LIKE ? COLLATE NOCASE "
                "OR product_session_id LIKE ? COLLATE NOCASE)"
            )
            query_text = f"%{filename_query}%"
            parameters.extend((query_text, query_text, query_text))
        if overall_result:
            overall_result = _valid_result(overall_result)
            clauses.append("overall_result = ? COLLATE NOCASE")
            parameters.append(overall_result)
        if start_date:
            clauses.append("created_at >= ?")
            parameters.append(_date_boundary(start_date, is_end=False))
        if end_date:
            clauses.append("created_at <= ?")
            parameters.append(_date_boundary(end_date, is_end=True))

        query = (
            "SELECT product_session_id, filename, product_name, created_at, overall_result, "
            "field_decision_summary, reviewer_status FROM report_history"
        )
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC, product_session_id DESC LIMIT ?"
        parameters.append(limit)

        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [_row_to_report(row) for row in rows]

    def get_report(self, product_session_id: str) -> dict[str, Any] | None:
        """Return one report summary, or ``None`` when it does not exist."""
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT product_session_id, filename, created_at, overall_result,
                       product_name, field_decision_summary, reviewer_status
                FROM report_history
                WHERE product_session_id = ?
                """,
                (product_session_id,),
            ).fetchone()
        return _row_to_report(row) if row is not None else None

    def get_report_summary(self, product_session_id: str) -> dict[str, Any] | None:
        """Return a previously saved compact summary, or ``None`` if absent."""
        return self.get_report(product_session_id)

    def analytics_counts(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, int]:
        """Return PASS, REVIEW, and VIOLATION totals, including zeroes."""
        clauses: list[str] = []
        parameters: list[Any] = []
        if start_date:
            clauses.append("created_at >= ?")
            parameters.append(_date_boundary(start_date, is_end=False))
        if end_date:
            clauses.append("created_at <= ?")
            parameters.append(_date_boundary(end_date, is_end=True))
        query = "SELECT overall_result, COUNT(*) AS count FROM report_history"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " GROUP BY overall_result"
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        counts = {"PASS": 0, "REVIEW": 0, "VIOLATION": 0}
        for row in rows:
            if row["overall_result"] in counts:
                counts[row["overall_result"]] = row["count"]
        return counts


def _field_decision_summary(report: dict[str, Any]) -> dict[str, str]:
    decisions = report.get("field_decisions", {})
    if not isinstance(decisions, dict):
        return {}
    return {
        str(field): str(decision.get("status", ""))
        for field, decision in decisions.items()
        if isinstance(decision, dict)
    }


def _row_to_report(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "product_session_id": row["product_session_id"],
        "filename": row["filename"],
        "product_name": row["product_name"],
        "created_at": row["created_at"],
        "overall_result": row["overall_result"],
        "field_decision_summary": json.loads(row["field_decision_summary"]),
        "reviewer_status": row["reviewer_status"],
    }


def _valid_limit(limit: int) -> int:
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")
    return limit


def _valid_result(result: str) -> str:
    normalized = str(result).upper()
    if normalized not in {"PASS", "REVIEW", "VIOLATION"}:
        raise ValueError("overall_result must be PASS, REVIEW, or VIOLATION")
    return normalized


def _date_boundary(value: str, is_end: bool) -> str:
    if not isinstance(value, str):
        raise TypeError("dates must be ISO strings")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("dates must be ISO-8601 strings") from error
    if len(value) == 10:
        return f"{value}T23:59:59.999999+00:00" if is_end else f"{value}T00:00:00+00:00"
    return parsed.isoformat()
