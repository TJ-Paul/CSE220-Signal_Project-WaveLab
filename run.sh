#!/usr/bin/env bash
# Signal Lab — start the DSP API and the web frontend together.
#
#   ./run.sh            start both, stream logs, Ctrl+C stops both
#
# API  → http://localhost:8000  (FastAPI over audio_toolkit)
# Web  → http://localhost:5173  (React + Vite, proxies /api to the API)
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "error: .venv not found. Create it and install requirements.txt first." >&2
  exit 1
fi

if [ ! -d web/node_modules ]; then
  echo "Installing web dependencies…"
  (cd web && npm install)
fi

# Free the ports if a previous run left something listening.
for port in 8000 5173; do
  pids=$(lsof -ti:"$port" -sTCP:LISTEN || true)
  [ -n "$pids" ] && echo "$pids" | xargs kill 2>/dev/null || true
done

cleanup() {
  echo
  echo "Stopping Signal Lab…"
  kill 0 2>/dev/null || true
}
trap cleanup EXIT INT TERM

source .venv/bin/activate
python -m uvicorn server.main:app --port 8000 --reload &

(cd web && npm run dev) &

echo
echo "  Signal Lab"
echo "  ──────────────────────────────────────"
echo "  Web:  http://localhost:5173"
echo "  API:  http://localhost:8000/api/health"
echo "  Ctrl+C stops both."
echo

wait
