#!/usr/bin/env bash
# Read-only setup check for a fresh clone: lists what is installed, what data and saved outputs are present, which
# features therefore work, and how to obtain anything missing. It downloads nothing, starts nothing, changes
# nothing, and never prints secret values.   Usage: ./scripts/check_setup.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
PY="$ROOT/.venv/bin/python"
ok=0; missing=0
pass() { printf '  \033[1;32m✓\033[0m %s\n' "$*"; ok=$((ok + 1)); }
miss() { printf '  \033[1;31m✗\033[0m %s\n      → %s\n' "$1" "$2"; missing=$((missing + 1)); }
note() { printf '  \033[1;33m•\033[0m %s\n' "$*"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# run ids the app is configured to show (configs/dashboard.yaml)
cfg_run() { sed -nE "s/^ *$1: *([^ #]+).*/\1/p" configs/dashboard.yaml | head -1; }
PRIMARY="$(cfg_run primary)"; TWIN="$(cfg_run primary_twin)"; REF="$(cfg_run reference)"; EXPL="$(cfg_run exploratory)"

head_ "1. Tools"
if command -v python3.12 >/dev/null; then pass "python3.12 ($(python3.12 --version 2>&1))"
else miss "python3.12 not found" "install Python 3.12 (macOS: brew install python@3.12); 3.11 also works, 3.13+ is not supported"; fi
if [[ -x "$PY" ]] && "$PY" -c "import fastapi, lingam, pymc, statsmodels, yfinance" 2>/dev/null; then
  pass "Python environment .venv with the core, API and agent dependencies"
else miss "Python environment .venv missing or incomplete" "make install   (or: make install-locked for the exact tested versions)"; fi
if command -v node >/dev/null && [[ "$(node -p 'process.versions.node.split(".")[0]')" -ge 20 ]]; then pass "Node.js $(node --version)"
else miss "Node.js 20+ not found" "install Node 20 or newer (https://nodejs.org or brew install node)"; fi
if [[ -d frontend/node_modules ]]; then pass "frontend dependencies (frontend/node_modules)"
else miss "frontend dependencies not installed" "cd frontend && npm ci   (./run.sh start also does this on first launch)"; fi
if [[ -d "/Applications/Google Chrome.app" ]] || command -v google-chrome >/dev/null; then
  pass "Google Chrome (the browser tests use the installed Chrome, frontend/playwright.config.ts)"
else note "Google Chrome not found (only needed for browser tests): install Chrome, or change 'channel' in frontend/playwright.config.ts"; fi

head_ "2. Credentials (values are never printed)"
if [[ -f .env ]] && grep -qE '^FRED_API_KEY=.+' .env; then pass ".env with a FRED_API_KEY"
else miss "no FRED API key in .env (needed only to download/regenerate data)" \
  "cp .env.example .env, then add a free key from https://fred.stlouisfed.org/docs/api/api_key.html"; fi

head_ "3. Local LLM (optional: live 'Ask the agents' and re-generating explanations)"
if command -v ollama >/dev/null; then pass "Ollama installed"
else miss "Ollama not installed" "install the free Ollama app or CLI from https://ollama.com (macOS: brew install ollama)"; fi
if curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  pass "Ollama server running on 127.0.0.1:11434"
  if curl -fsS http://127.0.0.1:11434/api/tags 2>/dev/null | grep -q '"qwen2.5:7b"'; then pass "model qwen2.5:7b pulled"
  else miss "model qwen2.5:7b not pulled" "ollama pull qwen2.5:7b   (about 4.4 GB, free, Apache-2.0)"; fi
else note "Ollama server not running (./run.sh start launches it if Ollama is installed)"; fi

head_ "4. Data (gitignored; regenerate with your own downloads — see README 'Data setup')"
if [[ -f data/raw/latest_snapshot.json ]]; then pass "raw snapshot (Yahoo Finance, FRED, ALFRED)"
else miss "no raw data snapshot" ".venv/bin/riskcausal verify-data --profile full   (needs the FRED key)"; fi
if ls -d data/processed/full_* >/dev/null 2>&1; then pass "processed full-profile panel (data/processed/full_*)"
else miss "no processed full-profile panel (needed by the app and Portfolio Lens)" ".venv/bin/riskcausal run --profile full --experiment m4_selected"; fi
if ls -d data/interim/full_graphs_* >/dev/null 2>&1; then pass "rolling VarLiNGAM graph cache (data/interim/full_graphs_*)"
else miss "no graph cache (needed by the network view)" "built by the same 'riskcausal run --profile full …' command"; fi

head_ "5. Saved research outputs (gitignored; configured in configs/dashboard.yaml)"
if [[ -n "$PRIMARY" && -f "outputs/runs/$PRIMARY/metrics.json" ]]; then pass "primary run $PRIMARY"
else
  miss "primary run '$PRIMARY' not found in outputs/runs (the app cannot start without it)" \
    "regenerate a run, then set runs.primary in configs/dashboard.yaml to its id (see README)"
  found="$(ls outputs/runs 2>/dev/null | grep _full_ | tail -3 | tr '\n' ' ')"
  [[ -n "$found" ]] && note "full-profile runs present here: $found"
fi
for pair in "primary_twin:$TWIN:reproducibility check (optional)" "reference:$REF:dev-profile reference run" "exploratory:$EXPL:KRE banking-stress run"; do
  role="${pair%%:*}"; rest="${pair#*:}"; rid="${rest%%:*}"; what="${rest#*:}"
  if [[ -n "$rid" && -d "outputs/runs/$rid" ]]; then pass "$role run ($what)"
  else note "$role run '$rid' not present ($what) — update configs/dashboard.yaml to a run you have"; fi
done
if [[ -f "data/raw/fnspid/$PRIMARY/news_subset.parquet" ]]; then pass "FNSPID news subset for the primary run (News agent)"
else miss "no FNSPID news subset for the primary run (News agent abstains everywhere)" \
  ".venv/bin/riskcausal news-subset outputs/runs/<primary run id>   (streams a 5.7 GB public file once; keeps ~80 MB)"; fi
n_v4="$(ls outputs/agents_v4/all/[0-9][0-9][0-9][0-9]-*.json 2>/dev/null | wc -l | tr -d ' ')"
if [[ "$n_v4" -gt 0 ]]; then pass "saved agent explanations ($n_v4 dates in outputs/agents_v4/all)"
else note "no saved agent explanations (shown when Ollama is off): .venv/bin/python -m riskcausal_agents.cli v4-eval outputs/runs/<primary run id> --which all"; fi

head_ "Summary"
echo "  $ok checks passed, $missing missing."
if [[ -f "outputs/runs/$PRIMARY/metrics.json" && -d data/processed ]]; then
  echo "  The app can start: ./run.sh start"
else
  echo "  The app cannot start yet. What works now: unit tests (make test; tests that need the locked run are"
  echo "  skipped), the frontend unit tests (cd frontend && npm test) and reading the code and documentation."
fi
exit 0
