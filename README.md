# Multi-Agent LLM System for Real-Time Portfolio Risk Management with Causal Inference

**FSE 570 Data Science Capstone** · Team: Dabola, Puri, Tanikella, Anumula
**Application:** *Structural Instability Monitor*

> **What this repository actually is.** The project title describes the original research vision. The implemented
> system is a **historical** financial-risk research pipeline with a **replay** dashboard and **on-demand, local**
> multi-agent LLM explanations of historical evidence. It does **not** monitor markets in real time, does not
> trade, and does not give investment advice. Graph outputs are *estimated directed dependence under modelling
> assumptions*, not proven causal effects.

---

## Contents
1. [Overview and motivation](#1-overview-and-motivation)
2. [Research objective and hypothesis](#2-research-objective-and-hypothesis)
3. [System architecture](#3-system-architecture)
4. [Technology stack](#4-technology-stack)
5. [Data sources](#5-data-sources)
6. [Methodology](#6-methodology)
7. [Multi-agent LLM layer](#7-multi-agent-llm-layer)
8. [The application: Structural Instability Monitor](#8-the-application-structural-instability-monitor)
9. [Installation](#9-installation)
10. [Data setup and regeneration](#10-data-setup-and-regeneration)
11. [Running the application](#11-running-the-application)
12. [Testing](#12-testing)
13. [Reproducibility](#13-reproducibility)
14. [Results](#14-results)
15. [Limitations](#15-limitations)
16. [Future work](#16-future-work)
17. [Repository structure](#17-repository-structure)
18. [Project records and contributing](#18-project-records-and-contributing)

---

## 1. Overview and motivation
Value-at-Risk and similar measures rely on historical covariance, and those relationships tend to break exactly
during the regime shifts they are meant to warn about (e.g. February–March 2020, March 2023). This project asks
whether changes in the *structure* of dependence between assets and macro-financial variables — estimated with
rolling VARLiNGAM — carry earlier warning of stress than standard indicators.

The project has three parts:
- **Core research pipeline** (`src/riskcausal/`): point-in-time data, rolling directed-dependence graphs,
  instability features, regime labels, chronological walk-forward models, baselines and evaluation.
- **Explanation layer** (`src/riskcausal_agents/`): Graph, Macro, News and Synthesis agents running on a local
  open-weight model (Qwen 2.5 7B through Ollama), with every model statement checked against saved evidence.
- **Application** (`src/riskcausal_api/`, `frontend/`): a read-only local FastAPI service and a React UI that
  replays the saved results day by day as they were knowable at the time.

## 2. Research objective and hypothesis
**Research question.** Does instability in estimated directed dependence among assets and macro variables give
earlier warning of transitions into a stressed regime than correlation- and index-based measures (VIX, NFCI,
rolling-correlation VaR, DCC-GARCH)?

**Hypothesis (pre-registered).** Graph-instability features improve the detection of stress onsets over the VIX
baseline, alone or added to standard controls. A negative answer was declared a legitimate, reportable outcome
before the evaluation was run — and it is the outcome we found (see [Results](#14-results)).

## 3. System architecture
```
 Yahoo Finance ─┐                ┌──────────── core pipeline (src/riskcausal) ────────────┐
 FRED / ALFRED ─┼─► ingest ─► data/raw (snapshots + SHA256 manifest)                       │
 FNSPID news ───┘   │   point-in-time panel (NYSE calendar, release lags, ALFRED vintages)  │
                    │   rolling VARLiNGAM + Granger graphs ─► instability features (SII)    │
                    │   Markov-switching regimes ─► "stress onset within 10 days" target    │
                    │   walk-forward Bayesian logit / gradient boosting + baselines         │
                    │   evaluation: ROC/PR-AUC, event recall/precision, bootstrap CIs       │
                    └──────────────► outputs/runs/<run_id>/ (config, predictions, metrics, audit)
                                              │ read only
        ┌─────────────────────────────────────┼──────────────────────────────────────┐
        ▼                                     ▼                                      ▼
 src/riskcausal_agents               src/riskcausal_api (FastAPI,            frontend/ (React + TS)
 Graph · Macro · News agents          127.0.0.1 only; GET routes plus         Simple Mode / Advanced Mode,
 → validator → Synthesis agent        2 POST routes for agent jobs)           replay, Portfolio Lens, Agent Trace
 (local Ollama, qwen2.5:7b)
```
Design rules that hold throughout:
- **Point-in-time:** every value at decision date *t* uses only information published by the close of *t*.
- **Chronological walk-forward only:** no random splits; tuning and thresholds only on training/validation folds.
- **Separation:** the core pipeline cannot import the agents, API or UI (tested). LLM output never feeds a
  prediction, score or alert.
- **Every reported number comes from a saved run artifact**, never typed by hand.

## 4. Technology stack
| Area | Tools |
|---|---|
| Language / environment | Python 3.12 (3.11 supported), `venv` + `pip`; Node.js 20+ |
| Data | pandas, NumPy, PyArrow, `yfinance`, `fredapi`, `pandas_market_calendars`, `huggingface_hub`, DuckDB |
| Statistics / ML | `lingam` (VARLiNGAM), statsmodels (Granger, ADF/KPSS, Markov switching), `arch` (GARCH), PyMC + ArviZ (Bayesian logit), scikit-learn (gradient boosting) |
| LLM | Ollama (local server), `qwen2.5:7b` (Apache-2.0), JSON-schema-constrained decoding |
| API | FastAPI, Uvicorn |
| UI | React 18, TypeScript, Vite, Tailwind CSS, Recharts; Streamlit for the internal research dashboard |
| Testing | pytest, Ruff, Vitest + Testing Library, Playwright |

Everything is free and runs locally; no paid API or cloud service is used ([`COST_POLICY.md`](COST_POLICY.md)).

## 5. Data sources
| Source | What | Access | Redistribution |
|---|---|---|---|
| Yahoo Finance (via `yfinance`) | daily adjusted prices for 25 ETFs (equity, sector, bonds, credit, commodities, dollar) | free, no key | **not redistributed** (Yahoo terms) — download your own |
| FRED / ALFRED (St. Louis Fed) | VIX, credit and term spreads, NFCI, unemployment, etc.; ALFRED vintages and first-release dates for point-in-time values | free API key | not redistributed; download your own |
| FNSPID (`Zihan1004/FNSPID`, Hugging Face) | financial news headlines and summaries | free public dataset | no licence stated → **not redistributed**; coverage ends **2020-06-11** |

The full source list, coverage, gaps and substitutions are in [`DATASET_MANIFEST.md`](DATASET_MANIFEST.md),
generated from executed download tests.

## 6. Methodology
**Universe.** Full profile: 25 ETFs; the graph uses 13 daily nodes — SPY, IWM, EFA, EEM, XLF, TLT, LQD, HYG, GLD,
UUP and changes in VIX, BAA–10Y credit spread and 10Y–2Y term spread. The graph sample starts at HYG's listing
(2007-04). A 6-ETF dev profile is used for fast development.

**Point-in-time panel.** NYSE trading calendar; log returns and first differences; macro values usable only from
the business day after publication (ALFRED vintages for revised series, first-release dates otherwise); ETFs
enter only after listing. Stationarity is checked with ADF and KPSS.

**Directed dependence (VARLiNGAM).** Rolling windows (locked setting: W = 500 trading days, step 5, lag order by
BIC up to 5) estimate contemporaneous (B0) and lagged coefficient matrices; an edge exists when the standardised
coefficient exceeds 0.05. Rolling Granger graphs are estimated for comparison. VARLiNGAM assumes linearity,
non-Gaussian independent shocks and no hidden confounders; these assumptions are not verified here, so results
are described as *estimated directed dependence*.

**Structural Instability Index (SII).** A pre-specified composite of four graph-change measures between
consecutive windows — structural Hamming distance and Frobenius norm of change, each for B0 and the lagged
matrices — standardised with training-period statistics only.

**Target.** A two-regime Markov-switching model of 21-day SPY realised volatility labels calm/stress periods. The
prediction target is *a transition into stress within the next 10 trading days*. Training labels are fitted on
each fold's training data only; the ex-post labels used for scoring are never used as features or for tuning.

**Models and baselines.** Bayesian logistic regression (PyMC) and gradient boosting, each on three feature sets:
controls only (VIX, NFCI), instability only, and combined. Baselines: VIX level, NFCI (point-in-time), the SII
alone, rolling-correlation VaR, DCC-GARCH VaR and DCC mean correlation.

**Protocol.** Expanding-window walk-forward with annual test blocks, purging and an embargo; settings chosen once
on inner validation folds with a pre-registered grid and selection rule (M4), then a **single locked
evaluation** on the full profile (M5). Alert thresholds use the same rule for every method (≤ 4 false-alarm
episodes per year and ≤ 10% of calm validation days). Uncertainty: paired stationary block-bootstrap CIs and
Jeffreys intervals for event-level recall. Every run writes a leakage audit (`audit.json`); see
[`LEAKAGE_AUDIT.md`](LEAKAGE_AUDIT.md).

## 7. Multi-agent LLM layer
The agents **explain** saved alerts; they never produce or change a score or alert.

| Agent | Responsibility |
|---|---|
| **Graph agent** | reads changes in the estimated relationship network (edges appeared / disappeared / sign flipped / strengthened / weakened) |
| **Macro agent** | reads market and macro indicators (levels vs their point-in-time percentiles, 5-day VIX change) and the alert status; with a portfolio selected, also the portfolio's risk facts |
| **News agent** | reads headlines published before the 4 pm New York close on the decision date; abstains when there is no coverage |
| **Synthesis agent** | combines only facts the validator confirmed: chooses key facts, an overall class and which agents had no data |

**Local model.** `qwen2.5:7b` (Q4_K_M) through Ollama on `127.0.0.1:11434`, temperature 0, fixed seed. The
backend rejects non-local hosts and never falls back to another provider.

**Evidence grounding and validation (prompt version `m10-v4-r2`).**
1. Each agent receives numbered facts with **numbers only** (no words such as "high" or "low").
2. Its output is constrained by a JSON schema: a fact id and a label from that fact's fixed vocabulary, plus an
   optional short note.
3. A deterministic validator compares every label with the label computed from the saved numbers, and checks
   notes for direction errors, change-over-time words on level facts, correlation, causal, risk-level and forecast
   claims, advice, and numbers, names or dates absent from the evidence.
4. What users see is a **rule-rendered sentence** for each confirmed fact. A model note is shown only if it passed
   every check; synthesis notes are never shown as statements (see [Results](#llm-evaluation-m10)).
5. If the synthesis fails validation, a deterministic summary is shown instead.

The validator catches label errors against the saved numbers; it does **not** guarantee that every passed note is
faithful (see the hand-check results below).

**Live "Ask the agents".** A button in the UI runs all four agents on the local model for the selected replay
date (one job at a time, real progress per agent, cancellation, 300 s limit). It processes **historical** data
on demand; it is not real-time monitoring. Without Ollama the UI shows saved explanations generated offline with
the same pipeline.

**Agent Trace.** For each run: the inputs each agent saw, confirmed statements, rejected statements with the
reason, abstentions, and the synthesis outcome. Simple Mode shows a compact version.

**Earlier version.** The M6 free-text agents (`agents.py`, `validator.py`) are kept for the locked M6 evaluation
and are superseded in the UI.

## 8. The application: Structural Instability Monitor
A historical replay of the locked M5 run: choose a past date (or a preset such as the COVID-19 crash) and every
page shows only what was knowable at that day's close.
- **Simple Mode** (default, beginner-friendly): Briefing, Relationships, My portfolio, Ask the AI agents, What we found.
- **Advanced Mode:** Monitor, Timeline, Network, Alerts, Portfolio Lens, Agents, Research & validation.
- **Portfolio Lens (exploratory).** Build a portfolio from the project's ETFs (weights must total 100%, or start
  from a labelled example such as 60/40). It shows point-in-time volatility (63/252 days), historical and
  normal-model VaR/CVaR (95%), 252-day maximum drawdown, Euler risk contributions, and network changes touching
  the holdings, and can pass portfolio facts to the agents. It assumes daily rebalancing, maps ETFs outside the
  graph to the nearest graph node, and is **separate from the locked research evaluation**.
- **Guided demo:** a 7-step tour in each mode, including a portfolio and a live agent run.

An internal Streamlit research dashboard (`dashboard/`) presents the same saved runs for analysis.
Walkthrough and troubleshooting: [`DEMO.md`](DEMO.md).

## 9. Installation
Prerequisites: macOS or Linux, Python 3.12 (or 3.11), Node.js 20+, Git. Optional: Ollama (for the agents) and
Google Chrome (for the browser tests).

```bash
git clone https://github.com/k3vnos/multi-agent-llm-portfolio-risk-management-fse570.git
cd multi-agent-llm-portfolio-risk-management-fse570

make install            # .venv + all extras (core, news, agents, api, dashboard, dev)
# or: make install-locked   # the exact versions that were tested (requirements-lock.txt)

cd frontend && npm ci && cd ..
cp .env.example .env    # then add your own free FRED API key (see below); .env is gitignored
make check              # read-only report of what is installed / missing, and how to get it
```

**FRED API key (free).** Create an account at <https://fred.stlouisfed.org>, request a key at
<https://fred.stlouisfed.org/docs/api/api_key.html>, and put it in `.env` as `FRED_API_KEY=...`. Never commit
`.env`.

**Ollama and Qwen (optional, free, about 4.4 GB of model weights, stored by Ollama outside the repository).**
```bash
brew install ollama          # or the installer from https://ollama.com
ollama serve                 # listens on 127.0.0.1:11434 (./run.sh start also launches it)
ollama pull qwen2.5:7b
```

## 10. Data setup and regeneration
`data/` and `outputs/` are **not in the repository** (licensing and size). A fresh clone therefore cannot start
the application or reproduce the results until the data are downloaded and the pipeline is run. Run the steps
in order:

| Step | Command | Produces | Notes |
|---|---|---|---|
| 1. Download and verify sources | `.venv/bin/riskcausal verify-data --profile full` | `data/raw/<snapshot>/` + manifest; regenerates `DATASET_MANIFEST.md` | needs the FRED key; also probes FNSPID with small byte-range reads |
| 2. First-release dates | `.venv/bin/riskcausal fetch-releases --profile full` | `data/raw/fred_releases/` | needed for point-in-time timing of non-revised daily FRED series |
| 3. Locked configuration, full profile | `.venv/bin/riskcausal run --profile full --experiment m4_selected` | `data/processed/`, `data/interim/full_graphs_*`, `outputs/runs/<run_id>/` | about 10 minutes on an Apple M2 (graph build ~4 min) |
| 4. News subset (optional) | `.venv/bin/riskcausal news-subset outputs/runs/<run_id>` | `data/raw/fnspid/<run_id>/` (~80 MB) | streams the public 5.7 GB `All_external.csv` once and keeps only evidence windows; nothing is stored whole |
| 5. Saved agent explanations (optional) | `.venv/bin/python -m riskcausal_agents.cli v4-eval outputs/runs/<run_id> --which all` | `outputs/agents_v4/all/` | needs Ollama; about 1 minute per alert date |
| 6. Point the app at your runs | edit `configs/dashboard.yaml` → `runs.primary: <run_id>` | — | see below |

Other runs shown by the app (all optional):
- `primary_twin`: repeat step 3 to obtain a re-run for the reproducibility comparison.
- `reference`: dev-profile run `fetch-releases --profile dev` then `run --profile dev --experiment m4_selected`.
- `exploratory`: KRE banking-stress target `run --profile full --experiment m4_selected,banking_kre`.

Set each role in `configs/dashboard.yaml` to the run ids you generated (run ids contain a timestamp, so they
will differ from the originals). Pages that depend on a role you did not generate may not work.

**Expected layout after setup**
```
data/raw/{yfinance,fred,alfred,fred_releases,fnspid}/ ...   data/raw/latest_snapshot.json
data/processed/full_<snapshot>/                              data/interim/full_graphs_<hash>/, markov_cache/
outputs/runs/<run_id>/   (metrics.json, predictions.parquet, audit.json, ...)
outputs/agents_v4/all/<date>.json
```

**Important:** Yahoo Finance data downloaded today can differ from the original snapshot (2026-10-18), e.g.
through dividend and split re-adjustments or vendor corrections, and FRED series can be revised. Regenerated
results are therefore **not guaranteed to match** the reported numbers; exact reproduction of the locked
evaluation needs the original raw snapshot and saved run artifacts, which are kept by the project team and
are not in this repository.

**Without data, a fresh clone supports:** the Python unit tests (tests needing the locked run skip), the frontend
unit tests, linting, and code review. **It does not support:** the app, the browser tests, the Portfolio Lens,
saved or live agent explanations, or the research results.

## 11. Running the application
```bash
./run.sh start      # builds the UI on first use, starts Ollama if installed, serves http://127.0.0.1:8000
./run.sh status     # app and local-model status
./run.sh stop       # stops the app (and Ollama only if run.sh started it)
./run.sh restart | open | rebuild | test
NO_OLLAMA=1 ./run.sh start   # run without the local model (saved explanations only)
```
Everything binds to `127.0.0.1`. Demo journey: Simple Mode → **My portfolio** → *Classic 60/40* → *COVID-19
crash* → date 16 Mar 2020 → **Ask the AI agents** → **▶ Ask the agents**, or press **▶ Guided demo**.

Internal research dashboard:
```bash
.venv/bin/uvicorn riskcausal_api.main:app --host 127.0.0.1 --port 8000
RISKCAUSAL_API_URL=http://127.0.0.1:8000 .venv/bin/streamlit run dashboard/app.py   # http://127.0.0.1:8501
```

## 12. Testing
```bash
make test                         # pytest (backend, pipeline, leakage, agents, API, cost policy)
make lint                         # ruff
cd frontend && npm test           # Vitest unit tests
cd frontend && npx playwright test   # browser tests; needs ./run.sh start, the data and Chrome
./run.sh test                     # all of the above (browser tests only if the app is running)
```
Results on the project machine with full data: pytest 178 passed, Vitest 21 passed, Ruff clean (2026-11-07);
Playwright 23 passed (2026-11-04, including a live Qwen run and the COVID-19 portfolio-to-agents journey). In a fresh clone
without data, the tests that need the locked M5 run are skipped. The live-model browser tests skip when Ollama
is not running.

## 13. Reproducibility
- Same commit + same raw snapshot → bit-identical predictions, labels, features and metrics (verified for the M5
  locked run with a cold and a warm PyTensor compile cache; the Bayesian logit uses PyTensor's C backend for
  this reason).
- After a macOS upgrade, one baseline (rolling-correlation VaR) differed from earlier saved values by ≤ 1.4e-17
  (ranks unchanged), attributed to floating-point changes in the system maths library.
- Re-downloaded data are not guaranteed to reproduce the reported numbers (see §10).
- Research commands: `riskcausal m4-select` (validation-only selection), `riskcausal ci <run_dir>` (bootstrap
  CIs), `riskcausal report <run_dir>` (regenerate tables and figures),
  `python -m riskcausal_agents.cli v4-report --which heldout` (score the LLM evaluation set).

## 14. Results
### Early-warning evaluation (locked M5 run)
Run `20261030T225627Z_full_c1cccc3aa2_m4_selected`; test period 2014-01-02 to 2026-10-16; 21 stress onsets;
95% paired block-bootstrap CIs.

| Method | ROC-AUC [95% CI] | PR-AUC | Onsets detected |
|---|---|---|---|
| VIX level (baseline) | **0.721** [0.624, 0.807] | 0.293 | 14/21 |
| Bayesian logit — controls (VIX, NFCI) | 0.721 [0.628, 0.808] | 0.279 | 14/21 |
| DCC-GARCH VaR | 0.692 [0.598, 0.768] | 0.240 | 14/21 |
| Bayesian logit — combined (controls + instability) | **0.671** [0.569, 0.769] | 0.224 | 14/21 |
| Structural Instability Index (alone) | 0.601 [0.467, 0.722] | 0.157 | 3/21 |
| Bayesian logit — instability only | 0.438 [0.334, 0.548] | 0.109 | 4/21 |

**Finding (negative).** The structural-instability features did **not** improve on the VIX baseline.
Instability-only models were significantly worse than VIX (Bayes −0.283 [−0.437, −0.109]); adding instability
to the controls did not help (Bayes −0.050 [−0.115, 0.013]); the SII alone was not distinguishable from VIX and
detected 3 of 21 onsets. An exploratory banking-stress target (KRE, including SVB in March 2023) gave the same
picture. Full tables: [`PROGRESS.md`](PROGRESS.md) (M5) and the decision memo
[`DECISION_MEMO_M4.md`](DECISION_MEMO_M4.md).

### LLM evaluation (M10)
Fresh held-out set of 35 dates (23 alert episodes never used in development + 12 random no-alert days), scored
once with frozen prompts. Details: [`M10_AGENT_EVAL.md`](M10_AGENT_EVAL.md).

| Measure | Result |
|---|---|
| Raw label accuracy (before validation) | **80.5%** (239/297) — below the 90% target |
| — alert dates / no-alert days | 86.5% (179/207) / 66.7% (60/90) |
| Incorrect or duplicate labels rejected by the validator | all **58** (none displayed) |
| Synthesis confirmed by the validator | **33/35** dates |
| Model notes that passed automatic checks, hand-checked | 97/106 faithful — specialists 96/96, **synthesis 1/10** |

Because synthesis notes were unreliable, they are not displayed. The hand check is a first pass by the
development assistant and still needs team review. The earlier M6 free-text agents reached 24/33 (73%) faithful
claims on their held-out set; the two figures measure different things and are not directly comparable.

## 15. Limitations
- **Not real-time.** The application replays saved historical results; agents run on demand on historical data.
- **Not causal proof.** VARLiNGAM results depend on untested assumptions (linearity, non-Gaussian independent
  shocks, no hidden confounders); bootstrap refits of one window disagree on many edges.
- **Small event sample.** 21 test onsets (2014–2026); event-level intervals are wide. Labels depend on the SPY
  volatility regime model.
- **LLM reliability.** A 7B model with raw label accuracy of 80.5%; the validator rejects label errors but cannot
  guarantee every displayed note is faithful. News themes are checked by keywords; the causal-wording check
  misses some variants (e.g. "effects").
- **News coverage** ends 2020-06-11; the News agent abstains after that.
- **Portfolio Lens** is exploratory: daily rebalancing, ETFs mapped to graph-node proxies, not part of the
  locked evaluation, and its portfolio facts were not in the held-out LLM evaluation. Not investment advice.
- **Reproducibility from a fresh clone** requires re-downloading data that may differ from the original snapshot.

## 16. Future work
- Larger event samples (other markets, longer history) and alternative regime definitions.
- Nonlinear or latent-confounder-robust structure learning and graph-stability filtering.
- A larger or fine-tuned local model, team review of the hand checks, and a second independent rater.
- News sources with coverage after 2020 and semantic (non-keyword) theme validation.
- A genuine streaming data pipeline would be required before any real-time use.

## 17. Repository structure
```
src/riskcausal/            core pipeline: data/ (ingest, PIT, calendar), causal/ (VARLiNGAM, Granger),
                           features/, regimes/, models/, baselines/, validation/, evaluation/, cli.py
src/riskcausal_agents/     evidence bundles, facts.py + structured.py (v4 agents), validator, Ollama backend, cli.py
src/riskcausal_api/        FastAPI app: replay, portfolio, live agent jobs, run store
frontend/                  React + TypeScript UI (src/), Playwright browser tests (e2e/)
dashboard/                 Streamlit research dashboard
configs/                   base.yaml, profiles/ (dev, full), experiments/ (m4_selected, banking_kre, …), dashboard.yaml
tests/                     pytest suite (pipeline, leakage, agents, API, dashboard, cost policy)
scripts/check_setup.sh     read-only setup check (make check)
run.sh, Makefile           launcher and shortcuts
requirements-lock.txt      exact tested Python versions
*.md                       plan, progress log, manifests, audits, evaluation write-ups (see below)
data/, outputs/            local only, gitignored
```

## 18. Project records and contributing
| Document | Content |
|---|---|
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | original plan and design decisions |
| [`PROGRESS.md`](PROGRESS.md) | milestone log M0–M10 with actual results |
| [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) | every configuration tried and the data split used |
| [`LEAKAGE_AUDIT.md`](LEAKAGE_AUDIT.md) | leakage risks, guards and enforcing tests |
| [`DATASET_MANIFEST.md`](DATASET_MANIFEST.md) | verified data availability |
| [`DECISION_MEMO_M4.md`](DECISION_MEMO_M4.md), [`M6_EVAL_RUBRIC.md`](M6_EVAL_RUBRIC.md), [`M10_AGENT_EVAL.md`](M10_AGENT_EVAL.md) | decision gate, LLM rubric and evaluation |
| [`DEMO.md`](DEMO.md) | demo walkthrough and troubleshooting |
| [`COST_POLICY.md`](COST_POLICY.md) | zero-cost, local-only rules (one approved private GitHub repository) |

**Contributing.** Create a feature branch (`git switch -c feature/<topic>`), keep commits focused, run
`make test` and `make lint` (and the frontend tests for UI changes), then open a pull request against `main`.
Do not commit `.env`, `data/`, `outputs/` or model files, and do not change locked results or evaluation records.
