# Support Ticket Intelligence

An LLM-powered system over a 500-row customer support ticket dataset. Ask
questions in plain English, get exact answers; run deterministic anomaly
detection over SLA breaches and statistical outliers. Exposed as both a REST API
(FastAPI) and a UI (Streamlit).

Built for the DOTMappers AI Engineer assessment sprint.

---

## Quick start

```bash
git clone <this-repo> && cd dotmappers-support-ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Add a free Groq key from https://console.groq.com/keys  ->  GROQ_API_KEY=...

./run.sh
```

`run.sh` ingests the CSV, starts the API on **http://localhost:8000** (Swagger at
`/docs`) and the UI on **http://localhost:8501**.

**Docker alternative:**

```bash
cp .env.example .env    # add GROQ_API_KEY
docker compose up
```

**Fully offline alternative** — no API key at all:

```bash
ollama pull llama3.1:8b && ollama serve
# in .env:  LLM_PROVIDER=ollama
./run.sh
```

**Tests** (no key, no network needed — the LLM is mocked):

```bash
pytest -q        # 24 passing
```

---

## Architecture

```
                 ┌──────────────┐        ┌──────────────┐
  CSV (500 rows) │  ingest.py   │──────▶ │  SQLite      │
                 │ validate +   │        │  tickets     │
                 │ derive cols  │        │  (read-only) │
                 └──────────────┘        └──────┬───────┘
                                                │
   "How many Critical              ┌────────────┴────────────┐
    tickets are unresolved?"       │                         │
            │              ┌───────▼────────┐      ┌─────────▼─────────┐
            └─────────────▶│  nl_query.py   │      │   anomalies.py    │
                           │                │      │  rules + IQR      │
                           │ 1. LLM → SQL   │      │  (deterministic)  │
                           │ 2. validate    │      │        │          │
                           │ 3. execute     │      │  LLM narrates     │
                           │ 4. LLM narrates│      │  the result only  │
                           └───────┬────────┘      └─────────┬─────────┘
                                   │                         │
                           ┌───────▼─────────────────────────▼───────┐
                           │        api.py (FastAPI)                 │
                           │  /health /query /anomalies /stats       │
                           └───────────────────┬─────────────────────┘
                                               │ HTTP
                                     ┌─────────▼─────────┐
                                     │  ui/app.py        │
                                     │  (Streamlit)      │
                                     └───────────────────┘
```

### The central design decision

**The LLM writes SQL; SQLite computes the answer.**

The obvious alternative is to dump the 500 rows into the prompt and let the model
answer directly. It would have worked for this dataset and been faster to build.
I did not do that, for three reasons:

1. **Correctness.** Counting and averaging over hundreds of rows is the single
   thing language models are least reliable at. A database is exact, every time.
2. **Auditability.** Every response returns the generated SQL. A support manager
   who does not trust "142 tickets" can read the query that produced it. An
   opaque model answer offers no such recourse.
3. **Scale.** Prompt-stuffing has a hard ceiling around a few thousand rows. This
   architecture is unchanged at 5 million — swap SQLite for Postgres and the rest
   of the code is identical.

The LLM is used for the two things it is genuinely good at: turning fuzzy human
phrasing into precise query structure, and turning result rows into a readable
sentence.

### Component notes

| Component | Choice | Why |
|---|---|---|
| Store | SQLite | Zero-install, file-based, ships with Python. Read-only URI mode is a real security boundary, not a convention. Swappable for Postgres via connection string. |
| LLM | Groq free tier (`openai/gpt-oss-120b`), Ollama fallback | Brief requires zero cost. Groq is fast enough for interactive use; Ollama covers evaluators who would rather not sign up for anything. Provider is a one-line config change. |
| NL → data | Text-to-SQL, `temperature=0` | Deterministic, auditable, and SQL is far better represented in training data than any bespoke DSL I could invent. |
| Anomalies | Rules + IQR, no LLM in the decision path | SLA breaches must be reproducible. The same input must always produce the same flags — an LLM deciding what counts as an anomaly would be unauditable and non-deterministic. |
| API | FastAPI | Pydantic validation and free OpenAPI docs at `/docs`. |
| UI | Streamlit, talking to the API over HTTP | One code path. Anything the UI does, an API client can do identically — and the UI doubles as an integration test. |

---

## How natural language querying works

1. **Schema + vocabulary injection.** The prompt carries the real column types
   *and* the actual distinct values (`Open`, `Escalated`, `AGT-07`) plus the
   dataset's date range, pulled live from the database. This kills the most
   common text-to-SQL failure — filtering on a value that does not exist and
   silently returning zero rows.
2. **Derived columns at ingest.** `created_year_month`, `created_iso_week` and
   `is_open` are computed during ingestion, so the model never has to write
   fragile SQLite date arithmetic.
3. **Structured output.** The model returns `{"sql": ..., "assumptions": ...}` in
   JSON mode. The `assumptions` field surfaces interpretation back to the user —
   e.g. it will tell you that it read "this month" as March 2024.
4. **Validation before execution.** `validate_sql()` rejects anything that is not
   a single read-only `SELECT`/`WITH`, strips comments before keyword scanning
   (so `/* */ DROP` cannot smuggle through), blocks stacked statements, and
   requires the query to touch the `tickets` table. The connection is *separately*
   opened read-only. Two independent layers, because a prompt instruction is a
   request, not a guarantee.
5. **One self-correction pass.** If the SQL errors, the error text goes back to
   the model once for a repair attempt. Bounded at two attempts total — retry
   loops against a rate-limited free tier are a liability, not a feature.
6. **Grounded narration.** A second call turns the returned rows into prose, with
   explicit instructions to use only the numbers present and never estimate. If
   this call fails, the API still returns the correct rows with a warning — the
   data path degrades gracefully when the narration path does not.

### Handling "today" in a historical dataset

The data runs 2024-01-01 to 2024-03-30. A question like *"which agent resolved
the most tickets this month?"* would return nothing if "this month" meant the
real current month. The system anchors every relative date expression to the
dataset's newest ticket and reports that anchor in `reference_date` and in the
`assumptions` field. This is a genuine ambiguity in the brief, resolved
explicitly rather than silently.

---

## Anomaly detection

Seven detectors, in two families. All thresholds live in `config.py` / `.env` so
operations can tune them without touching code.

**Rule-based (SLA / policy):**

| Rule | Logic | Severity |
|---|---|---|
| `stale_high_priority` | High or Critical, unresolved, older than 24h | high |
| `aging_escalation` | Escalated, no resolution, sitting 7+ days | high |
| `slow_first_response` | First response beyond a priority-weighted target (Critical 1h → Low 8h) | high / medium |
| `low_customer_rating` | Resolved but rated ≤ 2/5 | medium |
| `inconsistent_timestamps` | Resolution time earlier than first response — impossible | medium |

**Statistical:**

| Rule | Logic | Severity |
|---|---|---|
| `resolution_time_outlier` | Above the IQR upper fence, **computed within each priority band** | high / medium |
| `agent_rating_outlier` | Agent mean rating ≥ 1.5σ below fleet mean, min 10 rated tickets | medium |

Two choices worth calling out:

- **Outliers are scoped per priority.** A 30-hour Critical ticket and a 30-hour
  Low-priority ticket are not the same event. Pooling them would drown real
  Critical breaches in Low-priority noise.
- **Data integrity is treated as an anomaly class.** 28 rows in this dataset
  record a resolution time earlier than their first response. Those rows quietly
  corrupt every average over resolution time. An anomaly system that polices SLAs
  while trusting its inputs is measuring noise.

On the supplied dataset: **389 flags across 234 unique tickets** — 267 high, 122
medium.

---

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + per-dependency status (DB, LLM). Returns 200 even when degraded, so the caller can see *which* dependency is down. |
| `POST` | `/query` | Natural-language question → answer, SQL, rows |
| `GET` | `/anomalies` | Anomaly report; `?severity=high&limit=50&summary=false` |
| `GET` | `/stats` | Deterministic descriptive statistics (no LLM) |
| `POST` | `/ingest` | Rebuild the database from the CSV |

```bash
curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many Critical tickets are unresolved?"}'
```

---

## Example queries and outputs

**1. "How many tickets are currently open?"**

```json
{
  "answer": "There are 111 tickets with status Open. A further 62 are Escalated, so 173 remain unresolved overall.",
  "sql": "SELECT COUNT(*) AS ticket_count FROM tickets WHERE status = 'Open'",
  "row_count": 1,
  "rows": [{"ticket_count": 111}],
  "assumptions": "Interpreted 'open' as status = 'Open' specifically, not all unresolved tickets."
}
```

**2. "Which agent resolved the most tickets this month?"**

```json
{
  "sql": "SELECT agent_id, COUNT(*) AS resolved_count FROM tickets WHERE status = 'Resolved' AND created_year_month = '2024-03' GROUP BY agent_id ORDER BY resolved_count DESC LIMIT 1",
  "assumptions": "The dataset ends 2024-03-30, so 'this month' was read as March 2024 rather than the real current month."
}
```

**3. "Show me all Critical tickets not resolved within 12 hours."**

```json
{
  "sql": "SELECT ticket_id, created_at, status, agent_id, resolution_time_hrs FROM tickets WHERE priority = 'Critical' AND (resolution_time_hrs > 12 OR resolution_time_hrs IS NULL) ORDER BY created_at",
  "assumptions": "Included unresolved Critical tickets (NULL resolution time) alongside those that took over 12 hours."
}
```

**4. "What is the average customer rating for Technical category tickets?"**

```json
{
  "sql": "SELECT ROUND(AVG(customer_rating), 2) AS avg_rating, COUNT(*) AS rated_tickets FROM tickets WHERE category = 'Technical' AND customer_rating IS NOT NULL"
}
```

Note the `IS NOT NULL` — the prompt explicitly forbids treating missing ratings
as zero, which is the failure mode that would silently drag every average down.

**5. "Delete all tickets"** — refused before execution:

```json
{
  "answer": "I could not build a valid query for that question. Last error: Disallowed SQL keyword(s): ['delete']",
  "row_count": 0
}
```

**Anomaly output** (`GET /anomalies?severity=high&limit=3`):

```json
{
  "reference_date": "2024-03-30 18:06:00",
  "total": 389,
  "by_rule": {
    "slow_first_response": 160, "stale_high_priority": 80,
    "aging_escalation": 54, "low_customer_rating": 47,
    "inconsistent_timestamps": 28, "resolution_time_outlier": 18,
    "agent_rating_outlier": 2
  },
  "anomalies": [{
    "ticket_id": "TKT-003",
    "rule": "slow_first_response",
    "severity": "high",
    "detail": "First response took 4.8h against a 2.0h target for High priority."
  }]
}
```

---

## Known limitations

Honest list, in rough order of how much they would bother me in production.

1. **Text-to-SQL is not perfect.** Complex multi-clause questions ("compare
   Q1 vs Q2 resolution times by agent for Critical tickets only") can produce
   valid-but-wrong SQL. The generated query is always returned so a user can
   check, but there is no automated correctness check. **Fix:** a golden set of
   50–100 question→SQL pairs run in CI, scored on result equivalence.
2. **No conversational memory.** Each question is independent; "and what about
   Billing?" will not resolve against the previous turn.
3. **Row cap of 200.** Large result sets are silently truncated (flagged in the
   response, but still truncated). Fine for a UI, wrong for bulk export.
4. **Anomaly thresholds are assumed, not derived.** The 24h and 4h SLAs are my
   guesses — the brief supplied no SLA policy. In production they would come from
   the actual contract, and ideally be per-customer-tier.
5. **Free-tier rate limits.** Groq will 429 under rapid repeated querying. The
   error surfaces clearly rather than being swallowed, but there is no request
   queue or caching layer.
6. **Single-node, in-process.** No auth, no rate limiting, CORS wide open. Fine
   for local evaluation, not for anything exposed.
7. **Full table re-read per anomaly scan.** At 500 rows this is instant; at
   millions it would need to move into SQL aggregates or a scheduled job writing
   to a results table.
8. **Narration is not verified against the rows.** The prompt forbids inventing
   numbers, but nothing programmatically checks that the prose matches the
   returned data. A numeric-consistency check would be the next thing I built.

## What I would do next, with more time

- The golden eval set from limitation 1 — it is the difference between "it worked
  in my demo" and "I know the accuracy rate".
- Cache identical questions on a hash of the normalised text; free-tier calls are
  the scarce resource.
- Per-agent and per-category trend detection over time, rather than point-in-time
  flags only.
- Move the anomaly scan to a scheduled job with a results table, so the endpoint
  reads precomputed flags instead of rescanning.

## Project layout

```
app/
  config.py      env-driven configuration, all thresholds
  ingest.py      CSV validation, derived columns, SQLite build
  db.py          read-only execution + SQL guardrails + schema snapshot
  llm.py         provider abstraction (Groq / Ollama), JSON parsing
  nl_query.py    question → SQL → rows → grounded answer
  anomalies.py   seven detectors + LLM narration of the result
  api.py         FastAPI routes
ui/app.py        Streamlit client (talks to the API over HTTP)
tests/           24 tests, LLM mocked — runs with no key and no network
data/            support_tickets.csv (tickets.db is generated)
```
