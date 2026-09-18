"""Anomaly detection over the ticket table.

Deliberately deterministic: these are SLA breaches and statistical outliers, and
an operations team needs the same input to produce the same flags every time. The
LLM is used only *after* detection, to write a short narrative summary — it never
decides what counts as an anomaly.

Two families of detector:
  * rule-based  — encode the SLA / business policy (stale high-priority tickets,
                  slow first response, unrated escalations, low ratings)
  * statistical — IQR outliers on resolution time, computed per priority band so
                  a slow Critical ticket is not judged against Low-priority norms,
                  plus agent-level rating outliers
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.config import (
    AGENT_MIN_TICKETS,
    IQR_MULTIPLIER,
    LOW_RATING_THRESHOLD,
    SLOW_FIRST_RESPONSE_HRS,
    STALE_HIGH_PRIORITY_HRS,
)
from app.db import load_all
from app.llm import LLMError, get_client

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

SUMMARY_SYSTEM_PROMPT = """You are a support operations analyst. You are given a \
JSON summary of anomalies detected in a ticket dataset by deterministic rules.

Write 2-4 sentences for a team lead: what the biggest problem is, roughly how
widespread it is, and what to look at first. Use only the numbers provided —
never invent counts, agents or ticket ids. No markdown headers, no bullet lists.
If there are no anomalies, say the dataset looks healthy against these rules."""


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df


def _record(
    df_row: pd.Series, rule: str, severity: str, detail: str
) -> dict[str, Any]:
    return {
        "ticket_id": df_row.get("ticket_id"),
        "rule": rule,
        "severity": severity,
        "detail": detail,
        "category": df_row.get("category"),
        "priority": df_row.get("priority"),
        "status": df_row.get("status"),
        "agent_id": df_row.get("agent_id"),
        "created_at": df_row.get("created_at"),
        "response_time_hrs": _clean(df_row.get("response_time_hrs")),
        "resolution_time_hrs": _clean(df_row.get("resolution_time_hrs")),
        "customer_rating": _clean(df_row.get("customer_rating")),
    }


def _clean(value: Any) -> Any:
    """NaN -> None so the value survives JSON serialisation."""
    return None if pd.isna(value) else value


# --------------------------------------------------------------------------
# Rule-based detectors
# --------------------------------------------------------------------------
def stale_high_priority(df: pd.DataFrame, reference: pd.Timestamp) -> list[dict]:
    """High/Critical tickets still unresolved longer than the SLA window."""
    mask = (
        df["priority"].isin(["High", "Critical"])
        & (df["status"] != "Resolved")
        & ((reference - df["created_at_dt"]).dt.total_seconds() / 3600 > STALE_HIGH_PRIORITY_HRS)
    )
    out = []
    for _, row in df[mask].iterrows():
        age = (reference - row["created_at_dt"]).total_seconds() / 3600
        out.append(
            _record(
                row,
                "stale_high_priority",
                "high",
                f"{row['priority']} ticket unresolved for {age:.1f}h "
                f"(SLA {STALE_HIGH_PRIORITY_HRS:.0f}h), status {row['status']}.",
            )
        )
    return out


def slow_first_response(df: pd.DataFrame) -> list[dict]:
    """First response slower than the SLA, weighted by priority."""
    thresholds = {
        "Critical": SLOW_FIRST_RESPONSE_HRS / 4,
        "High": SLOW_FIRST_RESPONSE_HRS / 2,
        "Medium": SLOW_FIRST_RESPONSE_HRS,
        "Low": SLOW_FIRST_RESPONSE_HRS * 2,
    }
    out = []
    for _, row in df.iterrows():
        limit = thresholds.get(row["priority"], SLOW_FIRST_RESPONSE_HRS)
        rt = row.get("response_time_hrs")
        if pd.notna(rt) and rt > limit:
            out.append(
                _record(
                    row,
                    "slow_first_response",
                    "high" if row["priority"] in ("Critical", "High") else "medium",
                    f"First response took {rt:.1f}h against a {limit:.1f}h "
                    f"target for {row['priority']} priority.",
                )
            )
    return out


def low_satisfaction(df: pd.DataFrame) -> list[dict]:
    """Resolved tickets that still left the customer unhappy."""
    mask = df["customer_rating"].notna() & (df["customer_rating"] <= LOW_RATING_THRESHOLD)
    return [
        _record(
            row,
            "low_customer_rating",
            "medium",
            f"Customer rated {int(row['customer_rating'])}/5 after "
            f"{row['resolution_time_hrs']:.1f}h to resolve."
            if pd.notna(row["resolution_time_hrs"])
            else f"Customer rated {int(row['customer_rating'])}/5.",
        )
        for _, row in df[mask].iterrows()
    ]


def inconsistent_timestamps(df: pd.DataFrame) -> list[dict]:
    """Data-integrity check: a ticket cannot be resolved before it was answered.

    Found 28 such rows in the supplied dataset. Flagging these matters because
    they silently corrupt any average over resolution time, and an anomaly system
    that only watches SLAs while trusting bad inputs is measuring noise.
    """
    mask = (
        df["resolution_time_hrs"].notna()
        & df["response_time_hrs"].notna()
        & (df["resolution_time_hrs"] < df["response_time_hrs"])
    )
    return [
        _record(
            row,
            "inconsistent_timestamps",
            "medium",
            f"Resolution time ({row['resolution_time_hrs']:.1f}h) is earlier than "
            f"first response ({row['response_time_hrs']:.1f}h) — likely a data "
            "quality issue, not a real SLA event.",
        )
        for _, row in df[mask].iterrows()
    ]


def aging_escalations(df: pd.DataFrame, reference: pd.Timestamp) -> list[dict]:
    """Escalated tickets with no resolution, ranked by how long they have sat."""
    mask = (df["status"] == "Escalated") & df["resolution_time_hrs"].isna()
    out = []
    for _, row in df[mask].iterrows():
        age_days = (reference - row["created_at_dt"]).total_seconds() / 86400
        if age_days < 7:
            continue
        out.append(
            _record(
                row,
                "aging_escalation",
                "high",
                f"Escalated {age_days:.0f} days ago with no resolution recorded.",
            )
        )
    return out


# --------------------------------------------------------------------------
# Statistical detectors
# --------------------------------------------------------------------------
def resolution_time_outliers(df: pd.DataFrame) -> list[dict]:
    """IQR outliers on resolution time, computed within each priority band."""
    out: list[dict] = []
    for priority, group in df.groupby("priority"):
        vals = group["resolution_time_hrs"].dropna()
        if len(vals) < 8:  # too few points for a meaningful quartile spread
            continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        upper = q3 + IQR_MULTIPLIER * iqr
        flagged = group[group["resolution_time_hrs"] > upper]
        for _, row in flagged.iterrows():
            out.append(
                _record(
                    row,
                    "resolution_time_outlier",
                    "high" if priority in ("Critical", "High") else "medium",
                    f"Resolution took {row['resolution_time_hrs']:.1f}h vs an "
                    f"upper-fence of {upper:.1f}h for {priority} tickets "
                    f"(median {vals.median():.1f}h).",
                )
            )
    return out


def agent_rating_outliers(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Agents whose mean rating sits well below the fleet average.

    Reported at agent level, not ticket level, so it is returned separately.
    """
    rated = df[df["customer_rating"].notna()]
    if rated.empty:
        return []
    stats = (
        rated.groupby("agent_id")["customer_rating"]
        .agg(["mean", "count"])
        .query(f"count >= {AGENT_MIN_TICKETS}")
    )
    if len(stats) < 3:
        return []
    overall, spread = stats["mean"].mean(), stats["mean"].std()
    if not spread or pd.isna(spread):
        return []
    out = []
    for agent_id, row in stats.iterrows():
        z = (row["mean"] - overall) / spread
        if z <= -1.5:
            out.append(
                {
                    "agent_id": agent_id,
                    "rule": "agent_rating_outlier",
                    "severity": "medium",
                    "mean_rating": round(float(row["mean"]), 2),
                    "tickets_rated": int(row["count"]),
                    "fleet_mean_rating": round(float(overall), 2),
                    "z_score": round(float(z), 2),
                    "detail": (
                        f"{agent_id} averages {row['mean']:.2f}/5 across "
                        f"{int(row['count'])} rated tickets vs a fleet average of "
                        f"{overall:.2f} (z={z:.2f})."
                    ),
                }
            )
    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def detect_anomalies(
    severity: str | None = None,
    limit: int = 100,
    include_summary: bool = True,
) -> dict[str, Any]:
    """Run every detector and return a structured report."""
    df = _prep(load_all())
    if df.empty:
        return {"total": 0, "anomalies": [], "agent_anomalies": [], "by_rule": {}}

    # The dataset is historical, so "now" is anchored to its newest ticket.
    reference = df["created_at_dt"].max()

    findings: list[dict] = []
    findings += stale_high_priority(df, reference)
    findings += slow_first_response(df)
    findings += low_satisfaction(df)
    findings += aging_escalations(df, reference)
    findings += inconsistent_timestamps(df)
    findings += resolution_time_outliers(df)

    # One ticket can trip several rules; keep them all but sort by urgency.
    findings.sort(key=lambda f: (SEVERITY_ORDER.get(f["severity"], 3), f["ticket_id"]))

    agent_findings = agent_rating_outliers(df)

    if severity:
        sev = severity.lower()
        findings = [f for f in findings if f["severity"] == sev]
        agent_findings = [f for f in agent_findings if f["severity"] == sev]

    by_rule: dict[str, int] = {}
    for f in findings + agent_findings:
        by_rule[f["rule"]] = by_rule.get(f["rule"], 0) + 1

    report: dict[str, Any] = {
        "reference_date": str(reference),
        "tickets_scanned": int(len(df)),
        "total": len(findings) + len(agent_findings),
        "by_rule": by_rule,
        "by_severity": {
            s: sum(1 for f in findings + agent_findings if f["severity"] == s)
            for s in ("high", "medium", "low")
        },
        "unique_tickets_flagged": len({f["ticket_id"] for f in findings}),
        "anomalies": findings[:limit],
        "agent_anomalies": agent_findings,
        "truncated": len(findings) > limit,
        "thresholds": {
            "stale_high_priority_hrs": STALE_HIGH_PRIORITY_HRS,
            "slow_first_response_hrs": SLOW_FIRST_RESPONSE_HRS,
            "low_rating_threshold": LOW_RATING_THRESHOLD,
            "iqr_multiplier": IQR_MULTIPLIER,
        },
    }

    if include_summary:
        report["summary"] = summarise(report)

    return report


def summarise(report: dict[str, Any]) -> str:
    """LLM narration of an already-computed report. Never changes the numbers."""
    facts = {
        "total_anomalies": report["total"],
        "by_rule": report["by_rule"],
        "by_severity": report["by_severity"],
        "tickets_scanned": report["tickets_scanned"],
        "unique_tickets_flagged": report["unique_tickets_flagged"],
        "agent_anomalies": report["agent_anomalies"][:5],
        "sample_findings": [
            {k: f[k] for k in ("ticket_id", "rule", "severity", "detail")}
            for f in report["anomalies"][:8]
        ],
    }
    try:
        return get_client().complete(SUMMARY_SYSTEM_PROMPT, str(facts)).strip()
    except LLMError as exc:
        logger.warning("Anomaly summary unavailable: %s", exc)
        return (
            f"{report['total']} anomalies across {report['tickets_scanned']} tickets. "
            "(Narrative summary unavailable — LLM not reachable.)"
        )
