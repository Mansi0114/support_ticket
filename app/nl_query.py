"""Natural-language querying: question -> SQL -> rows -> grounded answer.

Design note (the thing worth arguing about in the walkthrough):

    We do NOT ask the LLM to answer questions about the data directly, and we do
    NOT stuff 500 rows into the prompt. The LLM writes *one SQL query*; SQLite
    computes the answer; the LLM then narrates the returned rows.

Why: arithmetic over 500 rows is exactly what language models are worst at and
what databases are best at. This split means counts and averages are always
exact, the answer is auditable (the SQL is returned to the caller), and the
approach does not degrade as the dataset grows from 500 rows to 5 million.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.db import UnsafeQueryError, run_query, schema_snapshot
from app.llm import LLMError, get_client

logger = logging.getLogger(__name__)

MAX_SQL_ATTEMPTS = 2  # one generation + one self-correction on error

SQL_SYSTEM_PROMPT = """You are a careful SQL analyst. You translate questions \
about a customer-support ticket table into a single SQLite SELECT query.

TABLE: tickets
COLUMNS:
  ticket_id            TEXT     unique id, e.g. 'TKT-001'
  created_at           TEXT     'YYYY-MM-DD HH:MM:SS' — compare as a string
  category             TEXT     Billing | Technical | General
  priority             TEXT     Low | Medium | High | Critical
  status               TEXT     Open | Resolved | Escalated
  response_time_hrs    REAL     hours to first agent response
  resolution_time_hrs  REAL     hours to resolution; NULL when not resolved
  agent_id             TEXT     e.g. 'AGT-04'
  customer_rating      INTEGER  1-5; NULL when not resolved
  issue_summary        TEXT     free-text description
  created_date         TEXT     'YYYY-MM-DD'   (pre-computed)
  created_year_month   TEXT     'YYYY-MM'      (pre-computed)
  created_iso_week     TEXT     'YYYY-Www'     (pre-computed)
  is_open              INTEGER  1 when status != 'Resolved', else 0

RULES
1. Output ONLY JSON: {"sql": "...", "assumptions": "..."} — no prose, no fences.
2. Exactly one SELECT (or WITH ... SELECT). Never INSERT/UPDATE/DELETE/DROP/PRAGMA.
3. "Unresolved" / "not resolved" / "still open" means status != 'Resolved'
   (this covers both Open and Escalated). "Open" alone means status = 'Open'.
4. resolution_time_hrs and customer_rating are NULL for unresolved tickets —
   exclude NULLs from averages with a WHERE clause; never treat NULL as 0.
5. String comparisons are case-sensitive in SQLite. Use the exact casing above,
   or wrap both sides in LOWER().
6. This is a historical dataset. "Today"/"now" does NOT mean the real current
   date — use the DATASET_MAX_DATE supplied below as the reference point, and
   say so in "assumptions".
   - "this month"  -> created_year_month = <month of DATASET_MAX_DATE>
   - "this week"   -> created_iso_week   = <ISO week of DATASET_MAX_DATE>
   - "last N days" -> created_at >= date(DATASET_MAX_DATE, '-N days')
7. Always alias aggregates readably: COUNT(*) AS ticket_count.
8. For "top"/"most"/"lowest" questions add ORDER BY and LIMIT.
9. If the question cannot be answered from these columns, return
   {"sql": "", "assumptions": "why it is not answerable"}.
10. Put any interpretation you had to make into "assumptions", briefly."""

ANSWER_SYSTEM_PROMPT = """You explain SQL query results to a support operations \
manager.

You are given the user's question and the exact rows the database returned.

RULES
- Answer in 1-3 sentences, plain English, no markdown headers.
- Use ONLY the numbers present in the rows. Never estimate, extrapolate or
  invent a figure. If the rows do not contain the answer, say so plainly.
- Quote figures exactly as given; round only if you say you are rounding.
- If the result set is empty, say no matching tickets were found and suggest
  what filter might be too narrow.
- If the result was truncated to a row cap, mention that it is a partial list.
- Do not restate the SQL."""


@dataclass
class QueryResult:
    question: str
    answer: str
    sql: str
    row_count: int
    rows: list[dict[str, Any]]
    assumptions: str = ""
    attempts: int = 1
    latency_ms: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sql": self.sql,
            "row_count": self.row_count,
            "rows": self.rows,
            "assumptions": self.assumptions,
            "attempts": self.attempts,
            "latency_ms": self.latency_ms,
            "warnings": self.warnings,
        }


def _context_block() -> str:
    snap = schema_snapshot()
    return (
        f"DATASET_MIN_DATE: {snap['min_created_at']}\n"
        f"DATASET_MAX_DATE: {snap['max_created_at']}\n"
        f"TOTAL_ROWS: {snap['row_count']}\n"
        f"DISTINCT categories: {snap['distinct_values']['category']}\n"
        f"DISTINCT priorities: {snap['distinct_values']['priority']}\n"
        f"DISTINCT statuses: {snap['distinct_values']['status']}\n"
        f"AGENT_COUNT: {snap['agent_count']}\n"
    )


def _rows_for_prompt(df: pd.DataFrame, limit: int = 25) -> str:
    """Serialise rows compactly; large result sets are summarised, not dumped."""
    if df.empty:
        return "(no rows returned)"
    head = df.head(limit)
    text = head.to_csv(index=False)
    if len(df) > limit:
        text += f"\n(showing first {limit} of {len(df)} rows)"
    return text


def generate_sql(question: str) -> tuple[str, str]:
    """Ask the LLM for a SQL query. Returns (sql, assumptions)."""
    client = get_client()
    user_msg = f"{_context_block()}\nQUESTION: {question}"
    payload = client.complete_json(SQL_SYSTEM_PROMPT, user_msg)
    return str(payload.get("sql", "")).strip(), str(payload.get("assumptions", "")).strip()


def answer_question(question: str) -> QueryResult:
    """Full pipeline with one self-correction round on invalid SQL."""
    question = (question or "").strip()
    if not question:
        raise ValueError("Question must not be empty.")
    if len(question) > 500:
        raise ValueError("Question is too long (max 500 characters).")

    started = time.perf_counter()
    client = get_client()
    warnings: list[str] = []
    sql, assumptions = generate_sql(question)

    if not sql:
        return QueryResult(
            question=question,
            answer=(
                "I can't answer that from this dataset. "
                + (assumptions or "The required information is not in the ticket table.")
            ),
            sql="",
            row_count=0,
            rows=[],
            assumptions=assumptions,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    df: pd.DataFrame | None = None
    last_error = ""
    attempts = 0

    for attempt in range(1, MAX_SQL_ATTEMPTS + 1):
        attempts = attempt
        try:
            df = run_query(sql)
            break
        except (UnsafeQueryError, Exception) as exc:  # noqa: BLE001
            last_error = str(exc)
            logger.warning("SQL attempt %s failed: %s", attempt, last_error)
            if attempt == MAX_SQL_ATTEMPTS:
                break
            warnings.append(f"First SQL attempt failed and was retried: {last_error}")
            repair_msg = (
                f"{_context_block()}\nQUESTION: {question}\n"
                f"Your previous SQL was:\n{sql}\n"
                f"It failed with this error:\n{last_error}\n"
                "Return corrected JSON with a working single SELECT."
            )
            try:
                payload = client.complete_json(SQL_SYSTEM_PROMPT, repair_msg)
                sql = str(payload.get("sql", "")).strip()
                assumptions = str(payload.get("assumptions", "")).strip() or assumptions
            except LLMError as llm_exc:
                last_error = str(llm_exc)
                break

    if df is None:
        return QueryResult(
            question=question,
            answer=(
                "I could not build a valid query for that question. "
                f"Last error: {last_error}. Try rephrasing, e.g. "
                "'How many Critical tickets are still unresolved?'"
            ),
            sql=sql,
            row_count=0,
            rows=[],
            assumptions=assumptions,
            attempts=attempts,
            latency_ms=int((time.perf_counter() - started) * 1000),
            warnings=warnings,
        )

    rows = df.where(pd.notnull(df), None).to_dict(orient="records")

    try:
        answer = client.complete(
            ANSWER_SYSTEM_PROMPT,
            f"QUESTION: {question}\n\nSQL RESULT ROWS (CSV):\n{_rows_for_prompt(df)}",
        ).strip()
    except LLMError as exc:
        # Degrade gracefully: the data is correct even when narration fails.
        warnings.append(f"Answer synthesis failed: {exc}")
        answer = f"Query returned {len(df)} row(s). See the table below."

    return QueryResult(
        question=question,
        answer=answer,
        sql=sql,
        row_count=len(df),
        rows=rows,
        assumptions=assumptions,
        attempts=attempts,
        latency_ms=int((time.perf_counter() - started) * 1000),
        warnings=warnings,
    )
