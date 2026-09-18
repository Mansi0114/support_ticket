"""Read-only SQL execution layer.

Every statement the LLM produces passes through `validate_sql` before it ever
reaches the database, and the connection itself is opened in SQLite's read-only
URI mode. Two independent layers, because prompt-level instructions are a
request, not a guarantee.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import DB_PATH, MAX_ROWS_RETURNED, TABLE_NAME

FORBIDDEN_TOKENS = {
    "insert", "update", "delete", "drop", "alter", "create", "replace",
    "attach", "detach", "pragma", "vacuum", "reindex", "truncate", "grant",
}


class UnsafeQueryError(ValueError):
    """Raised when generated SQL violates the read-only contract."""


def validate_sql(sql: str) -> str:
    """Return cleaned SQL, or raise if it is not a single read-only SELECT."""
    if not sql or not sql.strip():
        raise UnsafeQueryError("Empty SQL statement.")

    cleaned = sql.strip().rstrip(";").strip()

    # Reject stacked statements ("SELECT 1; DROP TABLE tickets").
    if ";" in cleaned:
        raise UnsafeQueryError("Multiple SQL statements are not allowed.")

    # Strip comments so they cannot smuggle keywords past the token check.
    no_comments = re.sub(r"--[^\n]*", " ", cleaned)
    no_comments = re.sub(r"/\*.*?\*/", " ", no_comments, flags=re.DOTALL)

    lowered = no_comments.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise UnsafeQueryError("Only SELECT / WITH queries are permitted.")

    tokens = set(re.findall(r"[a-z_]+", lowered))
    banned = tokens & FORBIDDEN_TOKENS
    if banned:
        raise UnsafeQueryError(f"Disallowed SQL keyword(s): {sorted(banned)}")

    if TABLE_NAME not in lowered:
        raise UnsafeQueryError(f"Query must read from the '{TABLE_NAME}' table.")

    return cleaned


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"Database missing at {db_path}. Run `python -m app.ingest` first."
        )
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def run_query(sql: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Validate and execute a read-only query, capped at MAX_ROWS_RETURNED."""
    safe_sql = validate_sql(sql)
    with _connect(db_path) as conn:
        df = pd.read_sql_query(safe_sql, conn)
    return df.head(MAX_ROWS_RETURNED)


def load_all(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Full table as a DataFrame — used by the statistical anomaly detectors."""
    with _connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)


def schema_snapshot(db_path: Path = DB_PATH) -> dict[str, Any]:
    """Distinct values and date bounds, injected into the text-to-SQL prompt.

    Giving the model the actual vocabulary of the data ('Escalated', 'AGT-07')
    removes an entire class of failure where it filters on a value that does not
    exist and silently returns zero rows.
    """
    with _connect(db_path) as conn:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({TABLE_NAME})")]
        distinct = {}
        for col in ("category", "priority", "status"):
            rows = conn.execute(
                f"SELECT DISTINCT {col} FROM {TABLE_NAME} ORDER BY 1"
            ).fetchall()
            distinct[col] = [r[0] for r in rows]
        agents = conn.execute(
            f"SELECT COUNT(DISTINCT agent_id) FROM {TABLE_NAME}"
        ).fetchone()[0]
        min_dt, max_dt = conn.execute(
            f"SELECT MIN(created_at), MAX(created_at) FROM {TABLE_NAME}"
        ).fetchone()
        total = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]

    return {
        "columns": cols,
        "distinct_values": distinct,
        "agent_count": agents,
        "min_created_at": min_dt,
        "max_created_at": max_dt,
        "row_count": total,
    }
