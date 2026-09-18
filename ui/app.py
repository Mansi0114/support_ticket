"""Streamlit UI.

Calls the FastAPI service over HTTP rather than importing the service modules
directly. That keeps one code path in production: whatever the UI can do, an API
client can do identically, and the UI is a real integration test of the API.
"""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TIMEOUT = 90

st.set_page_config(page_title="Support Ticket Intelligence", page_icon="🎫", layout="wide")

SAMPLE_QUESTIONS = [
    "How many tickets are currently open?",
    "Which agent resolved the most tickets this month?",
    "Show me all Critical tickets not resolved within 12 hours.",
    "What is the average customer rating for Technical category tickets?",
    "Which agent has the lowest average customer rating?",
    "How many Critical tickets are unresolved?",
]


def api_get(path: str, **params):
    r = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict):
    r = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=TIMEOUT)
    if r.status_code >= 400:
        detail = r.json().get("detail", r.text) if r.content else r.text
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json()


# --------------------------------------------------------------------------
# Sidebar: health + dataset stats
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("System status")
    try:
        health = api_get("/health")
        if health["status"] == "ok":
            st.success("All systems operational")
        else:
            st.warning("Degraded — see details")
        st.caption(
            f"DB: {health['database'].get('rows', '?')} rows · "
            f"LLM: {health['llm'].get('provider', '?')} "
            f"({'up' if health['llm'].get('reachable') else 'down'})"
        )
        if not health["llm"].get("reachable"):
            st.error(health["llm"].get("error", "LLM unreachable"))
    except Exception as exc:  # noqa: BLE001
        st.error(f"API unreachable at {API_BASE_URL}")
        st.caption(str(exc)[:200])
        st.stop()

    st.divider()
    try:
        stats = api_get("/stats")
        st.metric("Total tickets", stats["total_tickets"])
        st.metric("Unresolved", stats["unresolved"])
        st.metric("Avg rating", stats["avg_customer_rating"])
        st.metric("Avg resolution (hrs)", stats["avg_resolution_time_hrs"])
    except Exception:  # noqa: BLE001
        st.caption("Stats unavailable.")

st.title("🎫 Support Ticket Intelligence")
st.caption(
    "Ask questions in plain English. The LLM writes SQL, SQLite computes the "
    "answer — so every number below is exact and auditable."
)

tab_query, tab_anomaly, tab_data = st.tabs(
    ["Ask a question", "Anomalies", "Browse data"]
)

# --------------------------------------------------------------------------
# Tab 1 — NL query
# --------------------------------------------------------------------------
with tab_query:
    if "question" not in st.session_state:
        st.session_state.question = SAMPLE_QUESTIONS[0]

    st.write("**Try one of these:**")
    cols = st.columns(3)
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if cols[i % 3].button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state.question = q

    question = st.text_input("Your question", key="question")
    ask = st.button("Ask", type="primary")

    if ask and question.strip():
        with st.spinner("Generating SQL and querying..."):
            try:
                result = api_post("/query", {"question": question})
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                result = None

        if result:
            st.success(result["answer"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows returned", result["row_count"])
            c2.metric("Latency", f"{result['latency_ms']} ms")
            c3.metric("SQL attempts", result["attempts"])

            if result.get("assumptions"):
                st.info(f"Assumptions: {result['assumptions']}")
            for w in result.get("warnings", []):
                st.warning(w)

            if result["rows"]:
                st.dataframe(pd.DataFrame(result["rows"]), use_container_width=True)
            elif result["sql"]:
                st.caption("No rows matched that query.")

            if result["sql"]:
                with st.expander("Generated SQL"):
                    st.code(result["sql"], language="sql")

# --------------------------------------------------------------------------
# Tab 2 — Anomalies
# --------------------------------------------------------------------------
with tab_anomaly:
    c1, c2 = st.columns([1, 3])
    severity = c1.selectbox("Severity", ["all", "high", "medium", "low"])
    limit = c2.slider("Max rows", 10, 300, 100, step=10)

    if st.button("Run detection", type="primary"):
        with st.spinner("Scanning tickets..."):
            try:
                params = {"limit": limit}
                if severity != "all":
                    params["severity"] = severity
                report = api_get("/anomalies", **params)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                report = None

        if report:
            st.session_state["anomaly_report"] = report

    report = st.session_state.get("anomaly_report")
    if report:
        if report.get("summary"):
            st.info(report["summary"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total flags", report["total"])
        c2.metric("High severity", report["by_severity"]["high"])
        c3.metric("Tickets flagged", report["unique_tickets_flagged"])
        c4.metric("Scanned", report["tickets_scanned"])

        st.caption(
            f"Reference date (dataset 'now'): {report['reference_date']} · "
            f"Thresholds: {report['thresholds']}"
        )

        if report["by_rule"]:
            st.bar_chart(pd.Series(report["by_rule"], name="count"))

        if report["anomalies"]:
            st.subheader("Flagged tickets")
            st.dataframe(pd.DataFrame(report["anomalies"]), use_container_width=True)
            if report.get("truncated"):
                st.caption("List truncated — raise the row cap to see more.")

        if report["agent_anomalies"]:
            st.subheader("Agent-level outliers")
            st.dataframe(
                pd.DataFrame(report["agent_anomalies"]), use_container_width=True
            )

# --------------------------------------------------------------------------
# Tab 3 — Raw data
# --------------------------------------------------------------------------
with tab_data:
    st.caption("Deterministic view of the ingested dataset — no LLM involved.")
    try:
        stats = api_get("/stats")
        c1, c2, c3 = st.columns(3)
        c1.write("**By status**")
        c1.bar_chart(pd.Series(stats["by_status"]))
        c2.write("**By priority**")
        c2.bar_chart(pd.Series(stats["by_priority"]))
        c3.write("**By category**")
        c3.bar_chart(pd.Series(stats["by_category"]))
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
