"""FastAPI layer. Thin on purpose — all logic lives in the service modules."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import anomalies as anomaly_service
from app.config import DB_PATH
from app.db import load_all, schema_snapshot
from app.ingest import build_database, ensure_database
from app.llm import LLMError, get_client
from app.nl_query import answer_question

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_database()  # build SQLite from the CSV on first boot
    logger.info("Database ready at %s", DB_PATH)
    yield


app = FastAPI(
    title="Support Ticket Intelligence API",
    description=(
        "Natural-language querying and anomaly detection over a customer "
        "support ticket dataset. LLM writes SQL; SQLite computes the answer."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local evaluation only
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        json_schema_extra={"example": "How many Critical tickets are unresolved?"},
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    sql: str
    row_count: int
    rows: list[dict[str, Any]]
    assumptions: str = ""
    attempts: int = 1
    latency_ms: int = 0
    warnings: list[str] = []


class HealthResponse(BaseModel):
    status: str
    database: dict[str, Any]
    llm: dict[str, Any]


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness plus dependency checks. Returns 200 even when degraded, so the
    caller can see *which* dependency is down rather than just 'it failed'."""
    db_info: dict[str, Any] = {"path": str(DB_PATH), "reachable": False}
    try:
        snap = schema_snapshot()
        db_info.update(
            reachable=True,
            rows=snap["row_count"],
            date_range=[snap["min_created_at"], snap["max_created_at"]],
        )
    except Exception as exc:  # noqa: BLE001
        db_info["error"] = str(exc)[:200]

    try:
        llm_info = get_client().health()
    except LLMError as exc:
        llm_info = {"reachable": False, "error": str(exc)[:200]}

    ok = db_info.get("reachable") and llm_info.get("reachable")
    return HealthResponse(
        status="ok" if ok else "degraded", database=db_info, llm=llm_info
    )


@app.post("/query", response_model=QueryResponse, tags=["nl"])
def query(req: QueryRequest) -> QueryResponse:
    """Answer a natural-language question about the tickets."""
    try:
        return QueryResponse(**answer_question(req.question).to_dict())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in /query")
        raise HTTPException(status_code=500, detail=str(exc)[:300]) from exc


@app.get("/anomalies", tags=["anomalies"])
def anomalies(
    severity: Literal["high", "medium", "low"] | None = None,
    limit: int = Query(100, ge=1, le=500),
    summary: bool = True,
) -> dict[str, Any]:
    """Deterministic SLA and statistical anomaly detection, optionally narrated."""
    try:
        return anomaly_service.detect_anomalies(
            severity=severity, limit=limit, include_summary=summary
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in /anomalies")
        raise HTTPException(status_code=500, detail=str(exc)[:300]) from exc


@app.get("/stats", tags=["data"])
def stats() -> dict[str, Any]:
    """Deterministic descriptive stats — useful for the UI and for sanity-checking
    anything the LLM claims."""
    df = load_all()
    resolved = df[df["resolution_time_hrs"].notna()]
    return {
        "total_tickets": int(len(df)),
        "by_status": df["status"].value_counts().to_dict(),
        "by_priority": df["priority"].value_counts().to_dict(),
        "by_category": df["category"].value_counts().to_dict(),
        "unresolved": int((df["status"] != "Resolved").sum()),
        "avg_response_time_hrs": round(float(df["response_time_hrs"].mean()), 2),
        "avg_resolution_time_hrs": (
            round(float(resolved["resolution_time_hrs"].mean()), 2)
            if not resolved.empty
            else None
        ),
        "avg_customer_rating": (
            round(float(df["customer_rating"].dropna().mean()), 2)
            if df["customer_rating"].notna().any()
            else None
        ),
        "agents": int(df["agent_id"].nunique()),
    }


@app.post("/ingest", tags=["data"])
def ingest() -> dict[str, Any]:
    """Rebuild the SQLite database from the CSV (idempotent)."""
    try:
        return build_database()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)[:300]) from exc


@app.get("/", tags=["system"])
def root() -> dict[str, Any]:
    return {
        "service": "Support Ticket Intelligence API",
        "docs": "/docs",
        "endpoints": ["/health", "/query", "/anomalies", "/stats", "/ingest"],
    }
