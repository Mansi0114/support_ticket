#!/usr/bin/env bash
# Single-command start: ingests the CSV, boots the API, then the UI.
set -euo pipefail

if [ ! -f .env ]; then
  echo "No .env found — copying .env.example. Add your GROQ_API_KEY to it."
  cp .env.example .env
fi

echo "==> Ingesting CSV into SQLite"
python -m app.ingest

echo "==> Starting API on http://localhost:8000 (docs at /docs)"
uvicorn app.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT

sleep 3
echo "==> Starting UI on http://localhost:8501"
streamlit run ui/app.py --server.port 8501 --server.headless true

wait $API_PID
