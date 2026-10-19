# IMPLEMENTATION_PLAN — Multi-Agent LLM System for Real-Time Portfolio Risk Management with Causal Inference

FSE 570 Capstone · Dabola, Puri, Tanikella, Anumula

> **Status:** v3 — architecture approved; v2 added the seven review requirements (§F); v3 adds the
> **zero-cost / local-only restrictions (§G), which override anything else in this plan.**
> **Approved 2026-10-18.**
>
> **Tooling deviation (recorded at M0):** `uv` is not installed on this Mac, while Homebrew Python 3.12 is.
> To avoid installing extra tooling, the project uses a standard `python3.12 -m venv .venv` + `pip`
> (both free and already present). References to `uv` below should be read as "the project venv".

**Living records maintained throughout:**
- `COST_POLICY.md` — zero-cost, local-only rules (§G); written in M0, before anything is installed.
- `DATASET_MANIFEST.md` — verified data access/coverage (written in M1, before any modelling).
- `PROGRESS.md` — updated at the end of every milestone (template in §F6).
- `LEAKAGE_AUDIT.md` — checklist of every leakage risk, the guard, and the test that enforces it (§F3).
- `EXPERIMENT_LOG.md` — every configuration tried, which data split it was evaluated on, and why (§F4).

---

## 0. Context

**Why this project exists.** VaR/CVaR are built on historical covariance, and those relationships break during
the regime shifts they are meant to warn about (Feb–Mar 2020, Mar 2023 SVB). The proposal asks a single,
falsifiable research question:

> *Does instability in estimated **directed dependence** among assets/macro variables (rolling VarLiNGAM) give
> earlier warning of transitions into a stressed regime than correlation-based and index-based measures
> (VIX, NFCI, rolling-correlation VaR, GARCH/DCC)?*

A "no" is a legitimate, reportable result (proposal §5).

**Sources of truth.** `FSE570_Proposal.pdf` is authoritative. The slide deck
(`multi_agent_llm_portfolio_risk_capstone_rubric_aligned (1).pptx`) is secondary; where they diverge it is listed in
§E (item E18) and the PDF wins unless you decide otherwise.

**Starting state.** The project directory contains only the two documents. It is not a git repo. Nothing is
reused; this is greenfield.

**Non-negotiable constraints (from your brief + PDF)**
- No invented results, metrics, datasets or causal claims. Every number in the report traces to a saved artifact under `outputs/runs/<run_id>/`.
- VarLiNGAM/Granger output is called **"estimated directed dependence"**, never "causality", in code, logs, API and dashboard.
- Point-in-time discipline: every feature at decision time *t* uses only information published by end of day *t*.
- Chronological walk-forward only; no random splits; tuning only inside training folds.
- LLM agents are explanation-only. They never feed predictions, and their historical knowledge is never evidence of early warning.
- Core pipeline (`src/riskcausal/`) has zero imports from optional `agents/`, `api/`, `dashboard/`.
- Secrets only via env vars; `.env.example` committed, `.env` gitignored.
- Start with a small "dev" profile, scale to the full universe only after it works end-to-end.

---

## A. Architecture

```
                         ┌──────────────── CORE RESEARCH PIPELINE (required) ─────────────────┐
 yfinance ─┐             │                                                                      │
 FRED  ────┼─► L1 Ingest ─► data/raw/ (immutable snapshots + manifest.json w/ SHA256, pull time) │
 ALFRED ───┤             │        │                                                             │
 FNSPID ───┘             │        ▼                                                             │
                         │  L2 Process: trading calendar, as-of joins (ALFRED vintages + release │
                         │     lag), returns, transforms, ADF/KPSS stationarity report          │
                         │        │  → data/processed/panel_<profile>.parquet                  │
                         │        ▼                                                             │
                         │  L3 Directed dependence: rolling VarLiNGAM (+ rolling Granger)       │
                         │        │  → per-window B0, B1..Bk, adjacency, causal order          │
                         │        ▼                                                             │
                         │  L4 Features: graph-instability (edge churn, weight change) +        │
                         │     control features (VIX, NFCI, rolling corr) — all as-of t         │
                         │        ▼                                                             │
                         │  L5 Target: Markov-switching (Hamilton 1989) on realized vol →       │
                         │     calm/stress labels → "transition into stress within h days"      │
                         │        ▼                                                             │
                         │  L6 Models: Bayesian logistic (PyMC) + gradient boosting             │
                         │  L7 Walk-forward engine (expanding train, purged/embargoed test)     │
                         │  L8 Baselines: VIX, NFCI, rolling-corr VaR, DCC-GARCH               │
                         │  L9 Evaluation: precision/recall, false alarms, lead time, AUC,      │
                         │     Brier, block-bootstrap CIs; 2020 & 2023 case studies            │
                         │        → outputs/runs/<run_id>/ (config snapshot, preds, metrics)   │
                         └──────────────────────────────────────────────────────────────────────┘
                                         │ read-only (saved artifacts only)
              ┌──────────────────────────┼─────────────────────────────┐
              ▼                          ▼                             ▼
  OPTIONAL agents/ (grounded)     OPTIONAL api/ (FastAPI)       OPTIONAL dashboard/
  news · macro · graph agents     GET /alerts /graph /episodes   reads API; timeline,
  → coordinator, cites doc IDs    /explanations                  graph view, alert detail
```

**Key design decisions**
1. **Artifact-driven layers.** Each layer reads the previous layer's Parquet/JSON artifacts and writes its own; each is a CLI subcommand, cached by config hash. Re-running with the same config + raw snapshot reproduces outputs bit-for-bit (seeded).
2. **Point-in-time module** (`pit.py`) is the single place that answers "what value of series X was known at end of day t". Macro: latest ALFRED vintage with `realtime_start ≤ t − release_lag`. Non-revised daily series (VIXCLS, Treasury yields, BAA10Y) come from FRED with a 1-business-day lag rule. Unit tests assert no feature row ever references a vintage/date > t.
3. **Timestamp convention.** Row `t` = information through close of trading day `t`. Prediction issued at close of `t` for window `(t, t+h]`. Graph for window ending `t` is a feature at `t`.
4. **Leakage guards in the walk-forward engine.** Purge the last `h` training rows (their labels look into the test block), embargo gap between train/test, regime model and scalers refit on train only, thresholds chosen on train (inner time-split) only.
5. **Run registry.** `outputs/runs/<timestamp>_<config_hash>/` holds `config.yaml`, `git_sha`, `data_manifest.json`, predictions, metrics, figures, logs. Reports/README tables are generated from these files, never typed by hand.
6. **Grounding contract for agents.** Inputs = (alert record, graph JSON for that window, as-of macro values, news docs with `date ≤ alert date`). Output schema requires `doc_id`/`edge_id` citations for every claim; a validator rejects outputs with uncited or out-of-window citations. Explanations are stored separately and never touch metrics.

---

## B. Directory structure

```
portfolio-risk-capstone/
├── IMPLEMENTATION_PLAN.md
├── README.md                     # setup, commands, reproducibility, caveats on "directed dependence"
├── pyproject.toml                # deps + optional extras [agents], [api], [dashboard], [dev]
├── uv.lock
├── Makefile                      # thin aliases over the CLI
├── COST_POLICY.md
├── .env.example                  # FRED_API_KEY= (free key), OLLAMA_HOST=http://localhost:11434 — no paid-API keys
├── .gitignore                    # .env, data/, outputs/ (except small fixtures)
├── configs/
│   ├── base.yaml                 # shared defaults
│   ├── universe.yaml             # ETF list, FRED series, which nodes enter the graph
│   ├── profiles/dev.yaml         # 6 ETFs + 3 macro, fast settings
│   ├── profiles/full.yaml        # ~24 ETFs, full sample
│   └── experiments/*.yaml        # ablations / sensitivity (window length, h, etc.)
├── data/                         # gitignored
│   ├── raw/{yfinance,fred,alfred,fnspid,gdelt}/<snapshot_date>/ + manifest.json
│   ├── interim/
│   └── processed/
├── outputs/runs/<run_id>/        # gitignored; all reported numbers come from here
├── src/riskcausal/               # CORE — no imports from agents/api/dashboard
│   ├── config.py                 # pydantic models loading YAML + env
│   ├── logging_utils.py
│   ├── cli.py                    # typer: ingest, process, graph, features, label, train, evaluate, report
│   ├── registry.py               # run ids, config hashing, manifests
│   ├── data/ingest_yf.py · ingest_fred.py · ingest_alfred.py · ingest_fnspid.py · ingest_gdelt.py
│   ├── data/calendar.py · pit.py · transforms.py · stationarity.py · panel.py
│   ├── causal/varlingam_rolling.py · granger_rolling.py · graph_io.py
│   ├── features/instability.py · controls.py
│   ├── regimes/markov.py · targets.py
│   ├── models/bayes_logit.py · gbm.py
│   ├── baselines/vix.py · nfci.py · corr_var.py · dcc_garch.py
│   ├── validation/walkforward.py · thresholds.py
│   └── evaluation/metrics.py · lead_time.py · episodes.py · bootstrap.py · plots.py · report.py
├── agents/                       # OPTIONAL
│   ├── retrieval.py              # date-bounded news retrieval (≤ alert date)
│   ├── news_agent.py · macro_agent.py · graph_agent.py · coordinator.py
│   ├── schemas.py · validator.py # citation enforcement
│   └── prompts/
├── api/                          # OPTIONAL FastAPI, read-only over outputs/
│   └── main.py · routes/
├── dashboard/                    # OPTIONAL (Streamlit or static HTML+Plotly via API)
├── notebooks/                    # exploratory only; never the source of reported numbers
└── tests/
    ├── fixtures/                 # tiny synthetic + frozen sample data
    ├── test_pit.py · test_calendar.py · test_transforms.py · test_stationarity.py
    ├── test_varlingam_synthetic.py   # recovers known DAG on simulated LiNGAM data
    ├── test_instability.py · test_markov.py · test_targets.py
    ├── test_walkforward_no_leakage.py
    ├── test_metrics_lead_time.py · test_baselines.py
    └── agents/test_validator.py · api/test_routes.py
```

---

## C. Dependencies

Python **3.11 or 3.12** managed with `uv` (machine has 3.14; PyMC/lingam/arch wheels lag new Pythons).

| Purpose | Package |
|---|---|
| Core data | `pandas`, `numpy`, `pyarrow`, `scipy` |
| Market data | `yfinance` |
| Macro / ALFRED | `fredapi` (supports vintages/`get_series_all_releases`); `requests` fallback to raw ALFRED API |
| News | `datasets` / `huggingface_hub` (FNSPID), `duckdb` for filtering large CSVs; GDELT via `requests` (optional) |
| Calendar | `pandas_market_calendars` (NYSE) |
| Stationarity / Granger / Markov | `statsmodels` (ADF, KPSS, `grangercausalitytests`/VAR, `MarkovRegression`) |
| Directed dependence | `lingam` (VARLiNGAM) |
| GARCH | `arch` (univariate GARCH); DCC step implemented in-house (Engle 2002, two-step QML) |
| Bayesian logistic | `pymc`, `arviz` |
| Gradient boosting | `scikit-learn` (`HistGradientBoostingClassifier`) — `lightgbm` only if needed |
| Config / CLI / logging | `pydantic`, `pydantic-settings`, `pyyaml`, `typer`, stdlib `logging` (JSON formatter) |
| Plots | `matplotlib`, `plotly` |
| Dev | `pytest`, `pytest-cov`, `hypothesis`, `ruff`, `mypy` (light) |
| [agents] extra | `ollama` Python client (MIT) talking to a **local** Ollama server; `jsonschema`. No paid-LLM SDKs (no `anthropic`, `openai`, etc.) |
| [api] extra | `fastapi`, `uvicorn`, `httpx` (tests) |
| [dashboard] extra | `streamlit` (or none, if static HTML served by FastAPI) |

---

## D. Milestones (mapped to the PDF's 14-week schedule)

Each milestone ends with: tests green, a CLI command that runs it, artifacts in `data/` or `outputs/runs/`,
and a `PROGRESS.md` entry. Order is: **verify data → thin end-to-end prototype → harden & audit → scale → optional layers.**

**M0 — Scaffold (week 1)**
`COST_POLICY.md` first (§G), then local `git init` (main branch, no remote), pyproject + uv, config loader, logging, CLI skeleton, `.env.example`, `.gitignore`, pytest + ruff, README stub, empty `PROGRESS.md` / `LEAKAGE_AUDIT.md` / `EXPERIMENT_LOG.md`.
*Done when:* `uv run riskcausal --help` and `uv run pytest` pass.

**M1 — Dataset verification → `DATASET_MANIFEST.md` (week 1–2) — gate before any modelling**
`riskcausal verify-data --profile full` actually calls each source and records, per item:
- **ETFs** (every candidate ticker): first/last available date, # trading days, missing-day rate vs NYSE calendar, split/dividend adjustment confirmed, download status (ok / failed + error).
- **FRED series IDs** (VIXCLS, BAA10Y, AAA10Y, T10Y2Y, T10Y3M, DGS10, UNRATE, NFCI, BAMLH0A0HYM2, …): frequency, first/last date, missing rate, whether ALFRED vintages exist and how far back, download status.
- **FNSPID**: whether the Hugging Face dataset is reachable, file list + sizes, schema, date range, # records/ticker coverage from a sampled read, coverage of Feb–Mar 2020 and Mar 2023, license.
- **GDELT** (optional): reachability of the endpoint for one test date.
The manifest is generated from a JSON verification report (`data/raw/verification_<date>.json`), not written by hand. Anything that fails is marked FAILED and the plan adapts (e.g., E3 credit-spread substitute) — nothing is assumed to exist.
*Done when:* manifest committed; user reviews any substitutions before M2.

**M2 — Thin end-to-end prototype, dev profile (weeks 2–5)**
Goal: every core stage runs once, end to end, on real data, with the simplest defensible choices.
- Dev profile: **6 ETFs** (SPY, TLT, LQD, GLD, XLF, IEF — all pre-2005, final list confirmed by M1) + **3–4 indicators** (VIXCLS, BAA10Y, T10Y2Y; NFCI via ALFRED).
- Ingest + raw snapshot/manifest → NYSE calendar, log returns, diffs, PIT as-of joins, ADF+KPSS report.
- Rolling VarLiNGAM (initial hypothesis settings, E6) + rolling Granger.
- Instability components + composite.
- Markov-switching labels (fold-local, §F3) → transition target.
- Walk-forward with Bayesian logistic + gradient boosting.
- **Simple baselines only:** VIX, NFCI, rolling-correlation VaR. (DCC-GARCH deferred to M4.)
- Metrics + 2020/2023 case-study plots → `outputs/runs/<id>/`.
*Done when:* `make dev` runs ingest→report with no manual steps; outputs are real (no mocks); PROGRESS.md records actual metrics and runtime, whatever they are.

**M3 — Leakage audit & hardening (weeks 5–6)**
Work through every row of `LEAKAGE_AUDIT.md` (§F3), adding the enforcing test for each. Add synthetic-DAG recovery test for VarLiNGAM, toy-graph tests for instability, simulated two-regime test for the Markov model, hand-computed lead-time tests. Overlapping-window stability diagnostics (proposal Risk 2). Early held-out check (proposal Risk 1, week 5) on the dev profile.
*Done when:* all audit rows have a passing test; findings recorded in PROGRESS.md.

**M4 — Robustness on validation data + remaining baseline + Week-8 decision gate (weeks 6–8)**
Run the pre-registered robustness grid (§F4) using **only inner validation folds**. Add DCC-GARCH baseline (Engle 2002) now that the core works. Then a single locked evaluation on the outer test blocks; block-bootstrap CIs.
*Decision gate:* memo generated from saved metrics — continue / lengthen windows or average methods / pivot to explaining a negative result.

**M5 — Scale to full profile (week 8–9)**
~24 ETFs and the full indicator set from DATASET_MANIFEST.md, ~10 graph nodes (E1). Same locked protocol; any new config choices are again made on validation folds only and logged.

**M6 — Optional: grounded multi-agent explanations (weeks 9–11, only if core is complete and on schedule)**
FNSPID ingest (filtered by date windows around alerts + ticker→sector map), date-bounded retrieval, news/macro/graph agents, coordinator, citation validator, hand-checked sample with a rubric. Runs only on saved alerts; outputs stored under the run dir.
**LLM backend = local Ollama only** (§G2). Agent interfaces are written against an `LLMBackend` protocol with two implementations: `OllamaBackend` (localhost only) and `DisabledBackend` (default until Ollama is installed and approved — raises a clear "agents disabled" error, never falls back to anything else). Model candidates sized for this Mac (Apple M2, 24 GB RAM): `qwen2.5:7b` or `llama3.1:8b` (~5 GB, q4); 14B q4 (~9 GB) only if quality requires it.

**M7 — Optional: FastAPI + dashboard (week 11)**
Read-only endpoints over a chosen run: `/alerts`, `/graph/{date}`, `/episodes`, `/explanations/{alert_id}`, `/metrics`. Simple dashboard: stress timeline with alerts, graph view for a date, alert detail with cited evidence. First thing dropped if time is short.

**M8 — Write-up (weeks 12–14)**
`riskcausal report --run <id>` generates all tables/figures; README reproduction section; limitations (assumptions of LiNGAM, LLM hindsight caveat, two-event caveat).

---

## E. Research ambiguities needing decisions (with proposed defaults)

| # | Question | Proposed default |
|---|---|---|
| E1 | ETF universe, and **which subset becomes graph nodes**? VarLiNGAM on 24+ vars with ~250-obs windows is unreliable. | ~24 ETFs ingested; graph uses ~10 nodes: SPY, IWM, EFA, EEM, XLF, TLT, LQD, HYG, GLD, UUP (+ KRE sensitivity for 2023) plus daily macro nodes. |
| E2 | Ragged panel: HYG (2007-04), UUP (2007-02), DBC/USO (2006), XLRE/XLC (2015/2018). | Graph sample starts when all graph nodes exist (~mid-2007); late-listed ETFs used only in non-graph features/portfolio. |
| E3 | Credit spreads: ICE BofA OAS series on FRED now have truncated history. | Use BAA10Y (and AAA10Y) for long history; verify in M1 and report what was actually available. |
| E4 | Mixed frequencies in the graph. | Graph nodes = daily series only (ETF returns, ΔVIX, Δcredit spread, Δterm spread). Weekly/monthly (NFCI, UNRATE) are PIT control features/baselines, not graph nodes. |
| E5 | Stationary transforms. | Log returns for ETFs; first differences for VIX (or log-diff) and spreads; confirm via ADF+KPSS per sub-period. |
| E6 | VarLiNGAM hyperparameters: window W, step s, lag p, pruning/edge threshold, bootstrap. | W=250, s=5, p∈{1..5} by BIC, `prune=True`, edge exists if |b|>0.05 (standardized); bootstrap only for case-study windows (cost). Sensitivity W∈{126,250,500}. |
| E7 | Exact instability metric. | Report components separately (SHD, ‖ΔB0‖_F, ‖ΔB_lag‖_F) + z-scored composite using train-only stats. |
| E8 | Granger role: comparison only, or also a feature? | Both: report graph agreement and run Granger-instability as an ablation model. |
| E9 | Regime model specifics: which realized-vol series, # regimes, label truth for evaluation. | 21-day realized vol of SPY (sensitivity: equal-weight portfolio); 2 regimes, switching mean+variance. Training labels from MS fit on train data only; evaluation "truth" = ex-post smoothed labels (targets only, never features), with filtered-probability labels as a sensitivity check. **Needs your sign-off.** |
| E10 | Stress onset & horizon. | Onset = first stress day after ≥10 calm days; stress run must last ≥5 days; h ∈ {5, 10, 20}, primary h=10. |
| E11 | Portfolio for VaR baselines. | Equal-weight across graph-node ETFs (sensitivity: 60/40 SPY/IEF). |
| E12 | GARCH baseline form. | DCC-GARCH(1,1) (Engle 2002) implemented in-house on `arch` univariate fits; score = forecast portfolio vol and mean dynamic correlation. |
| E13 | Walk-forward scheme. | Expanding train, min 5 years, annual test blocks, refit yearly, purge h + 5-day embargo. |
| E14 | Alert threshold rule & false-alarm definition. | Threshold chosen on inner train split to cap false alarms at ≤ N alert-episodes/yr (N=4) — same rule for all models/baselines. False alarm = alert episode with no onset within h. |
| E15 | ALFRED: which series & release-time lag. | Vintages for NFCI, UNRATE (and any revised series); market-based series via FRED; +1 business-day lag on all macro values. |
| E16 | FNSPID scope: stock-level tickers, size (tens of GB), coverage end (~2023). | Subset to windows around alerts; map tickers→GICS sector→sector ETF; verify March 2023 coverage in M8; GDELT only if gaps. |
| E17 | LLM model choice (local only, §G2). | Local Ollama, 7–8B instruct model first; run only on saved alerts. Ollama is not currently installed — installing it needs your approval. |
| E18 | **PPTX vs PDF divergences**: deck lists Redis, PostgreSQL, AWS; historical VaR/CVaR baseline; "lead time vs VaR breach"; drawdown capture; 15-week timeline. PDF says "simple service and dashboard", baselines VIX/NFCI/rolling-corr VaR/GARCH, 14 weeks. | Follow PDF. Add historical VaR/CVaR and drawdown capture as cheap extra metrics. **No AWS or any cloud deployment (§G1)**; no Redis/Postgres — local Parquet files + FastAPI on localhost. |
| E19 | Bayesian logit details. | PyMC NUTS, Normal(0,1) priors on standardized features, feature sets: instability-only / controls-only / combined (ablation). |
| E20 | Significance of comparisons. | Stationary block bootstrap CIs on metric differences; no claim of superiority without CIs. |
| E21 | Python/tooling. | Python 3.12 via uv (free, local); local `git init` with `main` only — no remote, no push (§G3). Feature branches as sibling worktrees per your global rule. |

---

## F. Review requirements (v2 additions)

### F1. Dataset verification first
Milestone M1 above. No statistical model code is written until `DATASET_MANIFEST.md` exists and was produced by executed download tests. FNSPID is verified for reachability, schema and coverage early, even though it is only used in M6.

### F2. Working prototype before scale
Milestone M2 above: 6 ETFs + 3–4 indicators, every stage once, end to end. Scale (M5) and complex baselines (DCC-GARCH, M4) only after the prototype runs.

### F3. Leakage audit — two label families, kept apart

| Label / object | Built from | Used for | Never used for |
|---|---|---|---|
| `label_train[k]` | MS model fit on fold-k training data only (smoothed probs within train) | fitting fold-k models | anything in fold-k test |
| `label_eval` (retrospective) | MS fit on full sample, ex-post smoothed — "what we know now" | scoring test predictions | features, scaling, thresholds, hyperparameter choice |
| `regime_filtered_asof` (optional feature) | MS params from fold-k train, filtered prob using data ≤ t | feature at t | — |

Both label families are saved; results are reported against `label_eval` (primary) and against fold-local filtered labels (sensitivity), so the reader can see how much the retrospective-truth choice matters.

`LEAKAGE_AUDIT.md` rows, each with an enforcing test:
1. **PIT macro** — value at t uses latest vintage with `realtime_start ≤ t − lag`; test with synthetic revised vintages.
2. **Release lag** — +1 business day on every macro series; test that a Friday release appears Monday.
3. **Rolling windows** — graph/feature at t uses rows ≤ t only; test by perturbing rows > t and asserting features ≤ t unchanged.
4. **Transforms** — no centered rolling ops, no `bfill`, no full-sample means; lint-style test greps + property test.
5. **Scaling** — scalers/z-scores (incl. instability composite) fit on train only; test fails if fit sees test indices.
6. **Markov labels** — fold-local fit for training labels; `label_eval` imported only by `evaluation/`; import-boundary test.
7. **Target horizon purge** — last h training rows dropped + embargo; assert `train_max + h + embargo < test_min`.
8. **Thresholds / hyperparameters** — chosen on inner validation split only; assert outer test indices never reach the tuner.
9. **Ragged panel / survivorship** — ETF enters only after its first trade date; no back-filling of pre-listing data.
10. **yfinance adjustment** — adjusted prices use later splits/dividends; documented as a known, return-neutral adjustment (affects levels, not log returns beyond dividend inclusion).
11. **LLM hindsight** — agents never produce or influence predictions; news retrieval bounded to `≤ alert date`.

### F4. Settings are hypotheses — robustness without touching the test set
- Initial values in E6/E9/E10/E13/E14 are labelled **H0 settings** in config.
- A **pre-registered grid** (e.g., W∈{126,250,500}, edge threshold∈{0.03,0.05,0.1}, h∈{5,10,20}, regimes∈{2,3}, stress min-duration∈{3,5,10}) is written to `configs/experiments/robustness.yaml` before running.
- Evaluated only on **inner validation folds** inside each outer training period. Selection rule fixed in advance (e.g., median validation PR-AUC across folds).
- Outer test blocks are scored **once** with the selected config; H0 settings are also reported on test for transparency.
- Every run (incl. discarded ones) is appended to `EXPERIMENT_LOG.md` with run id, config hash, split used, and outcome.

### F5. Priorities
Core statistical pipeline (M1–M5) is the project. DCC-GARCH waits until M4. Agents (M6), API/dashboard (M7) start only once M2–M4 are complete and documented; they are the first to be cut.

### F6. `PROGRESS.md` — updated at the end of every milestone
```
## M<n> — <name> — <date>
Completed: …
Tests: <command> → <actual pass/fail counts pasted from output>
Artifacts: <paths, run ids>
Actual results: <numbers copied from outputs/runs/<id>/metrics.json, or "none yet">
Unresolved problems: …
Decisions needed: …
Next steps: …
```

### F7. No fabricated results
- Every metric in README/PROGRESS/report is read from a saved run artifact; `report.py` is the only producer of tables.
- Synthetic/mock data exists only in `tests/fixtures/` and for unit tests. Any artifact produced from synthetic data carries `synthetic: true` in its metadata, figures get a "SYNTHETIC DATA" watermark, and the report/API refuse to present a synthetic run as a result.
- If a stage fails or a source is unavailable, that is recorded as a failure — not filled with placeholder numbers.

---

## G. Zero-cost, local-only restrictions (v3 — override everything above)

### G1. Zero additional spending
- Claude Code runs only through your existing Claude Pro subscription. Pro does **not** imply Anthropic API access, and the project never assumes it does.
- The project code makes **no** requests to the Anthropic API or any other paid LLM API, and no such SDK is a dependency.
- No access to or changes in Anthropic billing, API keys, payment methods or subscriptions.
- No cloud resources, no deployments (AWS/Azure/GCP/etc.), no paid services, trials or purchases.
- Only free, open-source libraries, publicly accessible datasets, and software running on this Mac.
- **If any component turns out to need payment, work stops and I ask you.** Nothing proceeds automatically.

### G2. LLM agents must be free and local
- Agents use a local model through **Ollama** (free, MIT-licensed, runs on-device). This Mac (Apple M2, 24 GB RAM, ~460 GB free disk) can run 7–8B models comfortably.
- Ollama is **not installed** right now. I will ask before installing it or pulling any model (multi-GB downloads).
- Until then, agents use `DisabledBackend`: interfaces, prompts, retrieval, validator and tests are built, but execution is off (`agents.enabled: false` default).
- The Ollama client is hard-restricted to `localhost`/`127.0.0.1`; a non-local `OLLAMA_HOST` is rejected. There is no fallback to any other provider — if the local model is unavailable, the agent step fails loudly.

### G3. Everything stays local
- Development and all runs happen on this Mac.
- Local git only: no GitHub repo creation, no `git remote add`, no push, no `gh` commands. You upload to GitHub yourself later.
- FastAPI and the dashboard bind to `127.0.0.1` only.

### G4. Cost verification
Before installing or connecting anything, I confirm it is free for this use and note it in `COST_POLICY.md`. Current assessment, to be re-checked at install time:

| Component | Cost for this use | Note |
|---|---|---|
| Python libs in §C (pandas, statsmodels, lingam, pymc, arch, scikit-learn, yfinance, fredapi, fastapi, streamlit, ollama client…) | Free, open source (PyPI) | installed locally via uv |
| uv, Python 3.12 | Free, open source | local |
| Yahoo Finance via yfinance | Free, no key | public endpoints; rate limits only. Personal/research use |
| FRED / ALFRED API | Free | needs a free API key you create at fred.stlouisfed.org; no billing |
| FNSPID (Hugging Face) | Free public dataset | public download; no HF account needed if the repo is public (verified in M1); large — disk only |
| GDELT | Free public data | optional |
| Ollama + open-weight model | Free, local | needs your approval to install; model license noted (e.g. Llama/Qwen community licences) |
| Streamlit (local) | Free | local only — **not** Streamlit Community Cloud |

### G5. Defaults enforce the policy
- `configs/base.yaml`: `agents.enabled: false`, `agents.backend: disabled`, `api.host: 127.0.0.1`, `deploy: none`.
- `.env.example` contains only `FRED_API_KEY` and `OLLAMA_HOST=http://localhost:11434`; no paid-API key fields.
- A test (`tests/test_cost_policy.py`) fails if a paid-LLM SDK appears in `pyproject.toml`/lockfile, if any source file imports one, or if config defaults point to a non-local host.
- `COST_POLICY.md` (written in M0) holds §G1–G5 plus the verification table, and is linked from the README.

---

## Verification (how the whole thing is checked)

- `uv run pytest` — unit tests incl. synthetic-DAG recovery, PIT no-lookahead, walk-forward leakage, lead-time math on hand-built sequences.
- `riskcausal verify-data --profile full` → regenerates `DATASET_MANIFEST.md` from live download tests.
- `make dev` = `riskcausal ingest/process/graph/features/label/train/evaluate --profile dev` end-to-end on the small profile; produces `outputs/runs/<id>/metrics.json` and figures.
- Reproducibility check: run `make dev` twice from the same raw snapshot → identical metrics hash.
- `riskcausal report --run <id>` regenerates every table/figure; README numbers are copied from that output only.
- Optional layers: `pytest tests/agents tests/api`; `uvicorn api.main:app` then hit `/alerts` and the dashboard locally.
