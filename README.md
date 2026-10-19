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

More stages are added milestone by milestone (see `PROGRESS.md`).

## Ground rules
- No random splits; chronological walk-forward only. Every feature at time *t* uses only data published by *t*.
- Every reported number comes from a saved run under `outputs/runs/<run_id>/`.
- The optional LLM layer runs only on a local Ollama model and never influences predictions.
