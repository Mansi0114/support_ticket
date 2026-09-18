"""Central configuration. All values overridable via environment variables."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

CSV_PATH = Path(os.getenv("CSV_PATH", DATA_DIR / "support_tickets.csv"))
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "tickets.db"))
TABLE_NAME = "tickets"

# --- LLM -------------------------------------------------------------------
# Provider is pluggable: "groq" (free tier) or "ollama" (fully local).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "45"))

# --- Anomaly thresholds ----------------------------------------------------
# Kept as config, not magic numbers buried in code, so ops can tune them.
IQR_MULTIPLIER = float(os.getenv("IQR_MULTIPLIER", "1.5"))
STALE_HIGH_PRIORITY_HRS = float(os.getenv("STALE_HIGH_PRIORITY_HRS", "24"))
SLOW_FIRST_RESPONSE_HRS = float(os.getenv("SLOW_FIRST_RESPONSE_HRS", "4"))
LOW_RATING_THRESHOLD = int(os.getenv("LOW_RATING_THRESHOLD", "2"))
AGENT_MIN_TICKETS = int(os.getenv("AGENT_MIN_TICKETS", "10"))

# --- Query guardrails ------------------------------------------------------
MAX_ROWS_RETURNED = int(os.getenv("MAX_ROWS_RETURNED", "200"))
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://localhost:{API_PORT}")
