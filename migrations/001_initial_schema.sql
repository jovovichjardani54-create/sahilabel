-- migrations/001_initial_schema.sql
-- ---------------------------------------------------------------
-- PostgreSQL schema for the Legal Metrology Compliance Checker.
--
-- STATUS: This is an OPT-IN addition. The app currently works fully
-- with in-memory HISTORY + files on disk (uploads/, reports/,
-- annotated/) - that is NOT being removed or replaced by this file.
-- Adopt this schema when you're ready to move from "hackathon demo"
-- to "persists across restarts, queryable, multi-user" - see
-- db/README.md in this same folder for the cutover steps.
-- ---------------------------------------------------------------

CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      VARCHAR(64)  UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    role          VARCHAR(16)  NOT NULL DEFAULT 'viewer',  -- admin | inspector | viewer
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS products (
    id            SERIAL PRIMARY KEY,
    item_id       VARCHAR(16)  UNIQUE NOT NULL,   -- matches the existing short id (e.g. "18aa9f92")
    filename      VARCHAR(255),
    image_path    VARCHAR(512),
    annotated_path VARCHAR(512),
    pdf_path      VARCHAR(512),
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS inspections (
    id                SERIAL PRIMARY KEY,
    product_id        INTEGER REFERENCES products(id) ON DELETE CASCADE,
    inspector_id      INTEGER REFERENCES users(id),
    overall_compliant BOOLEAN NOT NULL,
    compliance_score  SMALLINT,                    -- 0-100, from plugins/scoring.py
    grade             CHAR(1),                      -- A-F
    latitude          DOUBLE PRECISION,
    longitude         DOUBLE PRECISION,
    scanned_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS violations (
    id             SERIAL PRIMARY KEY,
    inspection_id  INTEGER REFERENCES inspections(id) ON DELETE CASCADE,
    field_key      VARCHAR(64) NOT NULL,   -- e.g. "mrp", "manufacturer_info"
    rule_violated  TEXT NOT NULL,
    evidence       TEXT,
    suggested_action TEXT,
    severity       VARCHAR(16)              -- critical | moderate
);

-- Helpful indexes for the dashboard queries (Feature: Inspection Dashboard)
CREATE INDEX IF NOT EXISTS idx_inspections_scanned_at ON inspections(scanned_at);
CREATE INDEX IF NOT EXISTS idx_inspections_compliant ON inspections(overall_compliant);
CREATE INDEX IF NOT EXISTS idx_violations_field_key ON violations(field_key);
