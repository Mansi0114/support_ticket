"""Tests for everything that does not require a live LLM.

The LLM-dependent path is covered by mocking the client, so `pytest` passes with
no API key and no network — which matters when an evaluator clones the repo.
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from app.anomalies import (
    detect_anomalies,
    inconsistent_timestamps,
    resolution_time_outliers,
    stale_high_priority,
)
from app.db import UnsafeQueryError, run_query, schema_snapshot, validate_sql
from app.ingest import build_database, load_dataframe
from app.llm import LLMError, parse_json_response


@pytest.fixture(scope="module", autouse=True)
def database():
    build_database()


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------
def test_ingest_loads_all_rows():
    df = load_dataframe()
    assert len(df) == 500
    assert {"created_date", "created_year_month", "created_iso_week", "is_open"} <= set(
        df.columns
    )


def test_unresolved_tickets_have_null_resolution_time():
    df = load_dataframe()
    unresolved = df[df["status"] != "Resolved"]
    assert unresolved["resolution_time_hrs"].isna().all()


def test_is_open_flag_matches_status():
    df = load_dataframe()
    assert (df.loc[df["status"] == "Resolved", "is_open"] == 0).all()
    assert (df.loc[df["status"] != "Resolved", "is_open"] == 1).all()


# --------------------------------------------------------------------------
# SQL guardrails — the security boundary, so tested hardest
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE tickets",
        "DELETE FROM tickets WHERE 1=1",
        "SELECT * FROM tickets; DROP TABLE tickets",
        "UPDATE tickets SET priority='Low'",
        "PRAGMA table_info(tickets)",
        "INSERT INTO tickets VALUES (1)",
        "SELECT 1",  # does not touch the tickets table
        "",
    ],
)
def test_validate_sql_rejects_unsafe(sql):
    with pytest.raises(UnsafeQueryError):
        validate_sql(sql)


def test_validate_sql_accepts_select_and_cte():
    assert validate_sql("SELECT COUNT(*) FROM tickets;").endswith("tickets")
    assert validate_sql(
        "WITH x AS (SELECT * FROM tickets) SELECT COUNT(*) FROM x"
    ).startswith("WITH")


def test_comment_smuggling_is_caught():
    with pytest.raises(UnsafeQueryError):
        validate_sql("SELECT * FROM tickets /* */ DROP")


def test_run_query_returns_expected_count():
    df = run_query("SELECT COUNT(*) AS n FROM tickets")
    assert df.iloc[0]["n"] == 500


def test_schema_snapshot_exposes_real_vocabulary():
    snap = schema_snapshot()
    assert set(snap["distinct_values"]["status"]) <= {"Open", "Resolved", "Escalated"}
    assert snap["row_count"] == 500
    assert snap["min_created_at"] < snap["max_created_at"]


# --------------------------------------------------------------------------
# Anomalies — deterministic, so assertions can be exact
# --------------------------------------------------------------------------
def test_detect_anomalies_structure():
    report = detect_anomalies(include_summary=False)
    assert report["tickets_scanned"] == 500
    assert report["total"] == len(report["anomalies"]) + len(report["agent_anomalies"]) or report["truncated"]
    for finding in report["anomalies"]:
        assert finding["severity"] in {"high", "medium", "low"}
        assert finding["detail"]


def test_severity_filter_is_respected():
    report = detect_anomalies(severity="high", include_summary=False)
    assert all(f["severity"] == "high" for f in report["anomalies"])


def test_stale_high_priority_only_flags_unresolved():
    df = pd.DataFrame(
        {
            "ticket_id": ["T1", "T2", "T3"],
            "created_at": ["2024-01-01 00:00:00"] * 3,
            "created_at_dt": pd.to_datetime(["2024-01-01"] * 3),
            "category": ["Technical"] * 3,
            "priority": ["Critical", "Critical", "Low"],
            "status": ["Open", "Resolved", "Open"],
            "agent_id": ["AGT-01"] * 3,
            "response_time_hrs": [1.0] * 3,
            "resolution_time_hrs": [None, 2.0, None],
            "customer_rating": [None, 5, None],
        }
    )
    found = stale_high_priority(df, pd.Timestamp("2024-01-05"))
    assert [f["ticket_id"] for f in found] == ["T1"]


def test_outliers_are_scoped_per_priority():
    """A 30h Low-priority ticket is an outlier; a 30h Critical one need not be."""
    rows = []
    for i in range(20):
        rows.append(("Low", 5.0 + i * 0.1))
        rows.append(("Critical", 25.0 + i * 0.5))
    rows.append(("Low", 300.0))
    df = pd.DataFrame(
        {
            "ticket_id": [f"T{i}" for i in range(len(rows))],
            "priority": [r[0] for r in rows],
            "resolution_time_hrs": [r[1] for r in rows],
            "created_at": ["2024-01-01 00:00:00"] * len(rows),
            "category": ["Billing"] * len(rows),
            "status": ["Resolved"] * len(rows),
            "agent_id": ["AGT-01"] * len(rows),
            "response_time_hrs": [1.0] * len(rows),
            "customer_rating": [4] * len(rows),
        }
    )
    flagged = {f["ticket_id"] for f in resolution_time_outliers(df)}
    assert f"T{len(rows) - 1}" in flagged


# --------------------------------------------------------------------------
# LLM output handling
# --------------------------------------------------------------------------
def test_parse_json_handles_fences_and_prose():
    assert parse_json_response('```json\n{"sql": "SELECT 1"}\n```')["sql"] == "SELECT 1"
    assert parse_json_response('Sure! {"sql": "SELECT 2"} hope that helps')["sql"] == "SELECT 2"


def test_parse_json_raises_on_garbage():
    with pytest.raises(LLMError):
        parse_json_response("no json anywhere here")


# --------------------------------------------------------------------------
# NL pipeline with a mocked model — no network required
# --------------------------------------------------------------------------
class FakeClient:
    name, model = "fake", "fake"

    def __init__(self, sql):
        self._sql = sql

    def complete(self, system, user, json_mode=False):
        return "Mocked narrative answer."

    def complete_json(self, system, user):
        return {"sql": self._sql, "assumptions": "mocked"}


def test_answer_question_end_to_end_with_mock():
    from app import nl_query

    fake = FakeClient("SELECT COUNT(*) AS n FROM tickets WHERE status != 'Resolved'")
    with patch.object(nl_query, "get_client", return_value=fake):
        result = nl_query.answer_question("How many tickets are unresolved?")
    assert result.row_count == 1
    assert result.rows[0]["n"] == 173
    assert result.answer == "Mocked narrative answer."


def test_unsafe_generated_sql_is_not_executed():
    from app import nl_query

    fake = FakeClient("DROP TABLE tickets")
    with patch.object(nl_query, "get_client", return_value=fake):
        result = nl_query.answer_question("delete everything")
    assert result.row_count == 0
    assert "could not build a valid query" in result.answer.lower()
    # And the table survives.
    assert run_query("SELECT COUNT(*) AS n FROM tickets").iloc[0]["n"] == 500


def test_inconsistent_timestamps_flags_impossible_rows():
    df = pd.DataFrame(
        {
            "ticket_id": ["T1", "T2"],
            "response_time_hrs": [5.0, 1.0],
            "resolution_time_hrs": [2.0, 9.0],  # T1 resolved before it was answered
            "created_at": ["2024-01-01 00:00:00"] * 2,
            "category": ["Billing"] * 2,
            "priority": ["Low"] * 2,
            "status": ["Resolved"] * 2,
            "agent_id": ["AGT-01"] * 2,
            "customer_rating": [4, 4],
        }
    )
    assert [f["ticket_id"] for f in inconsistent_timestamps(df)] == ["T1"]


def test_dayfirst_dates_are_not_misread_as_month_first(tmp_path):
    """An Excel-resaved CSV ('05-02-2024') must parse as 5 Feb, not 2 May."""
    from app.ingest import _parse_datetimes

    parsed = _parse_datetimes(pd.Series(["05-02-2024 11:14", "13-02-2024 12:09"]))
    assert parsed.iloc[0].month == 2 and parsed.iloc[0].day == 5
    assert parsed.iloc[1].month == 2 and parsed.iloc[1].day == 13


def test_iso_dates_still_parse_correctly():
    from app.ingest import _parse_datetimes

    parsed = _parse_datetimes(pd.Series(["2024-02-05 11:14"]))
    assert parsed.iloc[0].month == 2 and parsed.iloc[0].day == 5
