"""CSV -> SQLite ingestion.

Why SQLite and not "just keep the DataFrame in memory":
  * gives the LLM a real, well-known query language (SQL) instead of asking it to
    emit pandas code we would then have to exec();
  * read-only connections are an easy, auditable safety boundary;
  * swapping in Postgres later is a connection-string change, not a rewrite.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from app.config import CSV_PATH, DB_PATH, TABLE_NAME

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "ticket_id",
    "created_at",
    "category",
    "priority",
    "status",
    "response_time_hrs",
    "resolution_time_hrs",
    "agent_id",
    "customer_rating",
    "issue_summary",
]

# Derived columns added at ingest time so the LLM never has to invent date maths.
DERIVED_COLUMNS = ["created_date", "created_year_month", "created_iso_week", "is_open"]


class IngestionError(RuntimeError):
    """Raised when the source CSV cannot be turned into a usable table."""


def _parse_datetimes(series: pd.Series) -> pd.Series:
    """Parse created_at, tolerating both ISO and day-first formats.

    The source file is ISO ('2024-02-05 11:14'), but opening it in Excel and
    saving rewrites dates as '05-02-2024 11:14'. Pandas reads that as US
    month-first and silently returns 2 May instead of 5 February — wrong answers
    with no error. So: try ISO strictly, and only fall back to day-first, never
    month-first ambiguous guessing.
    """
    iso = pd.to_datetime(series, format="ISO8601", errors="coerce")
    if iso.notna().all():
        return iso

    dayfirst = pd.to_datetime(series, dayfirst=True, errors="coerce")
    if dayfirst.notna().sum() > iso.notna().sum():
        logger.warning(
            "created_at is not ISO format — parsed as day-first (DD-MM-YYYY). "
            "Verify this matches your source file."
        )
        return dayfirst
    return iso


def load_dataframe(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Read the CSV, validate its schema, and add derived columns."""
    if not Path(csv_path).exists():
        raise IngestionError(f"CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise IngestionError(f"CSV is missing required columns: {missing}")

    df = df[EXPECTED_COLUMNS].copy()

    df["created_at"] = _parse_datetimes(df["created_at"])
    bad_dates = int(df["created_at"].isna().sum())
    if bad_dates:
        logger.warning("Dropping %s rows with unparseable created_at", bad_dates)
        df = df.dropna(subset=["created_at"])

    for col in ("response_time_hrs", "resolution_time_hrs", "customer_rating"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("category", "priority", "status", "agent_id"):
        df[col] = df[col].astype(str).str.strip()

    df["created_date"] = df["created_at"].dt.strftime("%Y-%m-%d")
    df["created_year_month"] = df["created_at"].dt.strftime("%Y-%m")
    df["created_iso_week"] = df["created_at"].dt.strftime("%G-W%V")
    df["is_open"] = (df["status"].str.lower() != "resolved").astype(int)
    # Store as text so SQLite string comparisons on dates behave predictably.
    df["created_at"] = df["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return df


def build_database(csv_path: Path = CSV_PATH, db_path: Path = DB_PATH) -> dict:
    """(Re)build the SQLite database from the CSV. Idempotent."""
    df = load_dataframe(csv_path)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_status ON {TABLE_NAME}(status)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_priority ON {TABLE_NAME}(priority)"
        )
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_agent ON {TABLE_NAME}(agent_id)")
        conn.commit()

    logger.info("Ingested %s rows into %s", len(df), db_path)
    return {
        "rows_ingested": len(df),
        "db_path": str(db_path),
        "columns": list(df.columns),
    }


def ensure_database(csv_path: Path = CSV_PATH, db_path: Path = DB_PATH) -> None:
    """Build the DB only if it is absent — called on API startup."""
    if not Path(db_path).exists():
        build_database(csv_path, db_path)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    print(build_database())
