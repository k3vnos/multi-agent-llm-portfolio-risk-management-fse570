# Multi-Agent LLM System for Real-Time Portfolio Risk Management with Causal Inference

FSE 570 capstone (Dabola, Puri, Tanikella, Anumula). Research prototype testing whether instability in
**estimated directed dependence** (rolling VarLiNGAM, with Granger as a comparison) gives earlier warning of
transitions into stressed market regimes than VIX, NFCI, rolling-correlation VaR and GARCH baselines.

> VarLiNGAM/Granger output is *estimated directed dependence under modelling assumptions*, not proven causality.

**Status:** see [`PROGRESS.md`](PROGRESS.md). Plan: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).
Data availability: [`DATASET_MANIFEST.md`](DATASET_MANIFEST.md). Zero-cost rules: [`COST_POLICY.md`](COST_POLICY.md).

## Setup (local, free)
```bash
make install            # python3.12 venv + free open-source deps
cp .env.example .env    # add a free FRED API key (needed only for ALFRED point-in-time vintages)
make test
```

## Commands
| Command | What it does |
|---|---|
| `make verify-data` | M1: downloads every candidate source, writes `data/raw/...` snapshot and regenerates `DATASET_MANIFEST.md` |
| `.venv/bin/riskcausal verify-data --render-only` | re-render the manifest from the latest saved report |
| `make dev` | M2 prototype end to end on the dev profile (6 ETFs + 4 indicators): PIT panel → rolling VarLiNGAM/Granger → instability features → fold-local Markov labels → walk-forward Bayesian logit / GBM + VIX, NFCI, rolling-corr VaR baselines → `outputs/runs/<run_id>/` (`metrics.md`, `metrics.json`, `predictions.parquet`, figures) |
| `.venv/bin/riskcausal fetch-releases --profile dev` | (existing files are kept; `--refresh` re-downloads) M3: ALFRED first-release dates for the profile's non-revised daily FRED series (free key); required before `process`/`run` for `availability: release` series |
| `.venv/bin/riskcausal run --profile dev --experiment regimes3` | run with an overlay from `configs/experiments/<name>.yaml` (run id gets the experiment tag) |
| `.venv/bin/riskcausal diagnose-graphs --profile dev` | bootstrap noise floor for window-to-window graph change (written next to the graph cache) |
| `.venv/bin/riskcausal m4-select --profile dev` | M4 pre-registered validation-only selection (labels, graph grid, target sensitivity) → `selection.md` + `configs/experiments/m4_selected.yaml` |
| `.venv/bin/riskcausal ci <run_dir>` | block-bootstrap and Jeffreys confidence intervals from saved predictions → `ci.md` |
| `.venv/bin/riskcausal credit-regimes --profile dev` | exploratory descriptive credit-spread regime check |
| `.venv/bin/riskcausal run --profile dev --experiment m4_selected` | the M4 locked configuration (overlays combine: `m4_selected,banking_kre`) |
| `.venv/bin/riskcausal report <run_dir>` | regenerate figures and `metrics.md` from a run's saved artifacts |

Every run also writes `audit.json`: real-data leakage checks (fold geometry, prediction placement, publication
timing of every FRED value behind each feature, no back-filled listings). See `LEAKAGE_AUDIT.md`.

More stages are added milestone by milestone (see `PROGRESS.md`).

## Optional explanation layer (local LLM, M6)
| Command | What it does |
|---|---|
| `.venv/bin/riskcausal news-subset <run_dir>` | stream FNSPID `All_external.csv` once and keep only articles in the run's evidence windows (coverage ends 2020-06-11) |
| `.venv/bin/python -m riskcausal_agents.cli explain-one <run_dir> <date> --backend ollama` | explain one saved alert with the local model (default backend is `disabled`) |
| `.venv/bin/python -m riskcausal_agents.cli eval-set <run_dir> --backend ollama [--seed 571 --name heldout --exclude-from …]` | pre-specified stratified evaluation set → `explanations/<name>/eval_set.csv` (rubric: `M6_EVAL_RUBRIC.md`) |

Requires `ollama serve` bound to 127.0.0.1 and `ollama pull qwen2.5:7b`. Explanations are illustrative and never
change predictions or metrics.

## The product: Structural Instability Monitor (M8)
```bash
./run.sh start        # → http://127.0.0.1:8000  (./run.sh stop | restart | status | test)
```
React + TypeScript UI over the read-only local API: historical replay of the locked M5 run (monitor, timeline,
relationship network, alerts, four-agent evidence view, research & validation) with a built-in guided demo.
Everything is point-in-time as of the replay date. See **DEMO.md** for the walkthrough and troubleshooting.

## Read-only dashboard (M7, internal research view, Streamlit)
```bash
.venv/bin/uvicorn riskcausal_api.main:app --host 127.0.0.1 --port 8000          # read-only API
RISKCAUSAL_API_URL=http://127.0.0.1:8000 .venv/bin/streamlit run dashboard/app.py   # http://127.0.0.1:8501
```
Without `RISKCAUSAL_API_URL` the dashboard uses the same read-only API in-process. Primary dataset: the M5 locked
run (`configs/dashboard.yaml`). Streamlit telemetry is off and both servers bind 127.0.0.1 (`.streamlit/config.toml`).
Pages: overview, stress timeline, model comparison (CIs), graph snapshot (estimated directed dependence), alert
explorer (point-in-time evidence with deterministic descriptions; LLM drafts collapsed, labelled unverified,
flagged or suppressed), exploratory KRE banking stress.

## Reproducibility
- Same commit + same raw snapshot → bit-identical predictions, labels, features and metrics (verified for the M5
  locked run with a cold and a warm PyTensor compile cache). The Bayesian logit compiles with PyTensor's C
  backend because the numba backend's draws depended on compile-cache state.
- Across OS upgrades, Apple Accelerate can change floating-point results in the last bit (seen once, ≤ 1.4e-17,
  ranks unchanged).

## Ground rules
- No random splits; chronological walk-forward only. Every feature at time *t* uses only data published by *t*.
- Every reported number comes from a saved run under `outputs/runs/<run_id>/`.
- The optional LLM layer runs only on a local Ollama model and never influences predictions.
