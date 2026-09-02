"""
db/database.py
----------------
OPTIONAL PostgreSQL layer using SQLAlchemy. NOT imported by main.py by
default - the app keeps working exactly as-is (in-memory HISTORY, files
on disk) until you deliberately wire this in.

Why opt-in rather than swapped-in now:
  - Your working demo should not depend on a running Postgres instance
    the night before judging.
  - Swapping storage backends is a real change with real risk; do it
    when you have time to test it, not under time pressure.

HOW TO ADOPT (when ready):
  1. `pip install sqlalchemy psycopg2-binary`
  2. Set env var DATABASE_URL, e.g.:
       postgresql://postgres:yourpassword@localhost:5432/legal_metrology
  3. Run the schema: `psql $DATABASE_URL -f migrations/001_initial_schema.sql`
     (or use Alembic if your team wants proper migration versioning)
  4. In main.py's /check endpoint, AFTER the existing HISTORY.append(record)
     line, add:
         from db.database import save_inspection
         save_inspection(record, image_path, pdf_path)
     This runs alongside HISTORY, not instead of it - so if the DB write
     fails for any reason, the demo still works off HISTORY as before.
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/legal_metrology"
)

# echo=False in normal use; flip to True temporarily when debugging queries
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)


def save_inspection(record: dict, image_path: str, pdf_path: str, annotated_path: str = None):
    """
    Persists one /check result into products + inspections + violations.
    Safe to call even if some optional fields (location, inspector_id)
    are None - matches the existing record shape from main.py exactly.
    """
    report = record["report"]
    location = record.get("location") or {}

    with SessionLocal() as session:
        product_row = session.execute(
            text(
                """
                INSERT INTO products (item_id, filename, image_path, annotated_path, pdf_path)
                VALUES (:item_id, :filename, :image_path, :annotated_path, :pdf_path)
                ON CONFLICT (item_id) DO NOTHING
                RETURNING id
                """
            ),
            {
                "item_id": record["id"],
                "filename": record["filename"],
                "image_path": image_path,
                "annotated_path": annotated_path,
                "pdf_path": pdf_path,
            },
        ).fetchone()

        if product_row is None:
            # item_id already existed (shouldn't normally happen - uuid4
            # collision is astronomically unlikely); fetch its id instead
            product_row = session.execute(
                text("SELECT id FROM products WHERE item_id = :item_id"),
                {"item_id": record["id"]},
            ).fetchone()
        product_id = product_row[0]

        inspection_row = session.execute(
            text(
                """
                INSERT INTO inspections
                    (product_id, overall_compliant, latitude, longitude)
                VALUES (:product_id, :overall_compliant, :lat, :lon)
                RETURNING id
                """
            ),
            {
                "product_id": product_id,
                "overall_compliant": report["overall_compliant"],
                "lat": location.get("latitude"),
                "lon": location.get("longitude"),
            },
        ).fetchone()
        inspection_id = inspection_row[0]

        for field_key, field_data in report["fields"].items():
            if not field_data["present"]:
                session.execute(
                    text(
                        """
                        INSERT INTO violations
                            (inspection_id, field_key, rule_violated, severity)
                        VALUES (:iid, :field_key, :rule, :severity)
                        """
                    ),
                    {
                        "iid": inspection_id,
                        "field_key": field_key,
                        "rule": field_data["description"],
                        "severity": field_data["severity"],
                    },
                )

        session.commit()
    return inspection_id


def get_dashboard_stats_from_db() -> dict:
    """DB-backed equivalent of plugins/dashboard_stats.compute_dashboard_stats,
    for once history lives in Postgres instead of the in-memory list."""
    with SessionLocal() as session:
        total = session.execute(text("SELECT COUNT(*) FROM inspections")).scalar()
        compliant = session.execute(
            text("SELECT COUNT(*) FROM inspections WHERE overall_compliant = TRUE")
        ).scalar()
        top_violations = session.execute(
            text(
                """
                SELECT field_key, COUNT(*) as cnt
                FROM violations
                GROUP BY field_key
                ORDER BY cnt DESC
                LIMIT 5
                """
            )
        ).fetchall()

    return {
        "total_scanned": total,
        "compliant_count": compliant,
        "compliance_rate": round(100 * compliant / total, 1) if total else 0,
        "most_common_violations": [{"field": r[0], "count": r[1]} for r in top_violations],
    }
