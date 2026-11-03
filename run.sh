#!/usr/bin/env bash
# Structural Instability Monitor — one-command local launcher (localhost only, read-only, free).
#   ./run.sh start     build the UI if needed, start the server, open the browser
#   ./run.sh stop      stop the server
#   ./run.sh restart   stop + start (use after a reboot or code change)
#   ./run.sh status    is it running? which URL?
#   ./run.sh open      open the app in the default browser
#   ./run.sh rebuild   force a fresh UI build, then restart
#   ./run.sh test      backend tests + frontend unit tests (+ browser tests if the server is running)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST=127.0.0.1
PORT="${PORT:-8000}"
URL="http://$HOST:$PORT/#/monitor"
PIDFILE="$ROOT/outputs/logs/app.pid"
LOG="$ROOT/outputs/logs/app.log"
PY="$ROOT/.venv/bin/python"

say() { printf '\033[1;35m▸\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

running() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
# PIDs of THIS app's server only (matched by its exact module path), e.g. an orphan left by a crash or sleep.
ours() { pgrep -f "uvicorn riskcausal_api.main:app --host $HOST --port $PORT" || true; }

check() {
  [[ -x "$PY" ]] || die "Python environment missing. Run: make install"
  command -v npm >/dev/null || die "Node.js/npm not found. Install Node 20+ (free), then retry."
  [[ -d "$ROOT/outputs/runs" ]] || die "No saved runs in outputs/runs — the app replays saved results only."
}

build() {
  say "Building the user interface (first time only, ~20 s)…"
  ( cd "$ROOT/frontend" && { [[ -d node_modules ]] || npm install --no-audit --no-fund; } && npm run build ) >>"$LOG" 2>&1 \
    || die "UI build failed — see $LOG"
}

start() {
  check
  mkdir -p "$(dirname "$LOG")"
  if running; then say "Already running at $URL"; return; fi
  if [[ -n "$(ours)" ]]; then say "Found a leftover server from an earlier session; stopping it."; stop; fi
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    die "Port $PORT is in use by another program. Stop it, or run: PORT=8010 ./run.sh start"
  fi
  [[ -f "$ROOT/frontend/dist/index.html" ]] || build
  say "Starting the local server on $HOST:$PORT (read-only)…"
  # only the server goes to the background, fully detached from this terminal (no inherited pipes)
  ( cd "$ROOT" || exit 1
    nohup "$PY" -m uvicorn riskcausal_api.main:app --host "$HOST" --port "$PORT" </dev/null >>"$LOG" 2>&1 &
    echo $! >"$PIDFILE" ) </dev/null >/dev/null 2>&1
  for _ in $(seq 1 60); do
    curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1 && break
    sleep 0.5
  done
  # macOS framework Python re-launches itself as a child: record the PID that actually holds the port
  local listener; listener="$(lsof -t -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -1)"
  [[ -n "$listener" ]] || die "Server did not start — see $LOG"
  echo "$listener" >"$PIDFILE"
  curl -fsS "http://$HOST:$PORT/replay/meta" >/dev/null 2>&1 || die "Server started but the replay data did not load — see $LOG"
  say "Ready: $URL"
  [[ "${NO_OPEN:-0}" == "1" ]] || open "$URL" 2>/dev/null || true
}

stop() {
  local pids; pids="$( { [[ -f "$PIDFILE" ]] && cat "$PIDFILE"; ours; } | sort -u | tr '\n' ' ')"
  rm -f "$PIDFILE"
  if [[ -z "${pids// /}" ]]; then say "Not running."; return; fi
  kill $pids 2>/dev/null || true
  for _ in $(seq 1 40); do                      # wait until the port is free
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 0.25
  done
  kill -9 $pids 2>/dev/null || true
  say "Stopped."
}

status() {
  if running; then say "Running (pid $(cat "$PIDFILE")) at $URL"
  elif [[ -n "$(ours)" ]]; then say "A server is running without a pid file (pid $(ours | tr '\n' ' ')). Use ./run.sh restart."
  else say "Not running. Start with: ./run.sh start"; fi
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  open) open "$URL" ;;
  rebuild) stop; build; start ;;
  test)
    check
    "$ROOT/.venv/bin/pytest" -q
    ( cd "$ROOT/frontend" && npm test --silent )
    if running; then ( cd "$ROOT/frontend" && npx playwright test ); else say "Server not running: skipping browser tests (./run.sh start first)."; fi ;;
  *) echo "usage: ./run.sh {start|stop|restart|status|open|rebuild|test}"; exit 1 ;;
esac
