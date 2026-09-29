#!/usr/bin/env bash
# Signal Lab — start the DSP API and the web frontend together.
#
#   ./run.sh            development: this computer only, live reload
#   ./run.sh --demo     LAN demo: phones on the same Wi-Fi can open the app
#
# Ctrl+C stops everything.
#
# API  → http://127.0.0.1:8000  (FastAPI over audio_toolkit; never exposed)
# Web  → http://localhost:5173  (React + Vite, proxies /api to the API)
#
# Demo mode serves a production build on every network interface, keeps the
# Mac awake, runs the models offline, and opens a join page with QR codes.
# Wi-Fi details for the join QR go in .env.demo:
#   WIFI_SSID=SignalLab
#   WIFI_PASSWORD=your-password
set -euo pipefail

cd "$(dirname "$0")"

MODE=dev
case "${1:-}" in
  "") ;;
  --demo | --lan) MODE=demo ;;
  -h | --help)
    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  *)
    echo "error: unknown option '$1' (try --demo or --help)" >&2
    exit 1
    ;;
esac

if [ ! -d .venv ]; then
  echo "error: .venv not found. Create it and install requirements.txt first." >&2
  exit 1
fi

if [ ! -d web/node_modules ]; then
  echo "Installing web dependencies…"
  (cd web && npm install)
fi

# Free the ports if a previous run left something listening. `kill` only
# asks a process to stop, so wait until the port is really free (forcing it
# after ~5 s) — otherwise the new server races the old one and fails with
# "Address already in use".
for port in 8000 5173; do
  pids=$(lsof -ti:"$port" -sTCP:LISTEN || true)
  [ -z "$pids" ] && continue
  echo "Stopping the previous server on port ${port}…"
  echo "$pids" | xargs kill 2>/dev/null || true
  for _ in $(seq 50); do
    lsof -ti:"$port" -sTCP:LISTEN >/dev/null 2>&1 || break
    sleep 0.1
  done
  pids=$(lsof -ti:"$port" -sTCP:LISTEN || true)
  [ -n "$pids" ] && echo "$pids" | xargs kill -9 2>/dev/null || true
done

cleanup() {
  echo
  echo "Stopping Signal Lab…"
  kill 0 2>/dev/null || true
}
trap cleanup EXIT INT TERM

source .venv/bin/activate

if [ "$MODE" = dev ]; then
  python -m uvicorn server.main:app --host 127.0.0.1 --port 8000 --reload &
  (cd web && npm run dev) &

  echo
  echo "  Signal Lab"
  echo "  ──────────────────────────────────────"
  echo "  Web:  http://localhost:5173"
  echo "  API:  http://127.0.0.1:8000/api/health"
  echo "  Ctrl+C stops both."
  echo
  wait
  exit
fi

# ── LAN demo ────────────────────────────────────────────────────────────────
if [ -f .env.demo ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env.demo
  set +a
fi

echo "Building the web app…"
(cd web && npm run build)

# The router has no internet: load cached model weights without asking the
# Hugging Face hub for updates, which would stall until the request times out.
export HF_HUB_OFFLINE=1

python -m uvicorn server.main:app --host 127.0.0.1 --port 8000 &
(cd web && npx vite preview --host 0.0.0.0 --port 5173 --strictPort) &

# Keep the Mac (and the projector) awake for as long as the demo runs.
caffeinate -dims -w $$ &

for _ in $(seq 120); do
  curl -sf -o /dev/null http://127.0.0.1:8000/api/health &&
    curl -sf -o /dev/null http://127.0.0.1:5173/ && break
  sleep 0.5
done

python scripts/lan_demo.py --port 5173 || true
echo "  Ctrl+C stops the demo."
echo
wait
