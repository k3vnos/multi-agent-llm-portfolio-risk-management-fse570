"""Command-line entry point: `riskcausal <stage> --profile dev`."""

from __future__ import annotations

import logging

import typer

from riskcausal.config import load_config
from riskcausal.logging_utils import setup_logging

app = typer.Typer(add_completion=False, help="Directed-dependence instability early-warning pipeline (local only).")
log = logging.getLogger("riskcausal")


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    setup_logging("DEBUG" if verbose else "INFO")


@app.command("verify-data")
def verify_data(profile: str = "full", render_only: bool = typer.Option(
        False, help="Re-render DATASET_MANIFEST.md from the latest JSON report without downloading.")) -> None:
    """M1: download/verify every candidate source and regenerate DATASET_MANIFEST.md."""
    from riskcausal.data.verify import render_latest, run_verification

    cfg = load_config(profile)
    path = render_latest(cfg) if render_only else run_verification(cfg)
    log.info("verification report: %s", path)


@app.command()
def process(profile: str = "dev") -> None:
    """Build the processed PIT panel for a profile from the latest raw snapshot."""
    from riskcausal.data.panel import build_panel

    build_panel(load_config(profile))


@app.command("fetch-releases")
def fetch_releases(profile: str = "dev", refresh: bool = typer.Option(
        False, help="Re-download series whose release file already exists (default: keep raw data immutable).")) -> None:
    """Download ALFRED first-release dates for the profile's non-revised daily FRED series (free API key)."""
    import json

    import pandas as pd

    from riskcausal.config import fred_api_key
    from riskcausal.data.calendar import nyse_sessions
    from riskcausal.data.ingest_fred import download_first_releases
    from riskcausal.data.panel import releases_dir
    from riskcausal.registry import write_json

    cfg = load_config(profile)
    key = fred_api_key()
    if not key:
        raise typer.BadParameter("FRED_API_KEY missing from .env")
    out = releases_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    sess = nyse_sessions("1990-01-01", today)
    spath = out / "summary.json"
    summary = json.loads(spath.read_text()) if spath.exists() else {}
    for sid in cfg.fred_series:
        if cfg.universe["fred"][sid]["revised"]:
            continue
        if (out / f"{sid}.parquet").exists() and not refresh:
            log.info("%s: release file exists, kept unchanged (use --refresh to re-download)", sid)
            continue
        first = download_first_releases(sid, key, int(cfg.start_date[:4]), today)
        first.to_frame().to_parquet(out / f"{sid}.parquet")
        lag = pd.Series(sess.searchsorted(first.values) - sess.searchsorted(first.index), index=first.index)
        summary[sid] = {"fetched": today, "tracked_from_obs": str(first.index.min().date()), "n_tracked": len(first),
                        "release_lag_sessions_counts": {str(k): int(v) for k, v in
                                                        lag.clip(upper=5).value_counts().sort_index().items()},
                        "share_released_more_than_1_session_late": float((lag > 1).mean()),
                        "max_release_lag_sessions": int(lag.max())}
        log.info("%s: first releases for %d observations (tracked from %s)", sid, len(first),
                 summary[sid]["tracked_from_obs"])
    write_json(spath, summary)


@app.command()
def run(profile: str = "dev", experiment: str = typer.Option(None, help="overlay configs/experiments/<name>.yaml"),
        rebuild_panel: bool = True) -> None:
    """Full core pipeline: panel -> rolling graphs -> features -> walk-forward -> evaluation -> report."""
    from riskcausal.pipeline import run_pipeline

    run_dir = run_pipeline(load_config(profile, experiment=experiment), rebuild_panel=rebuild_panel)
    typer.echo(str(run_dir))


@app.command("m4-select")
def m4_select(profile: str = "dev") -> None:
    """M4: pre-registered validation-only selection (labels, graph grid, target sensitivity). No test data."""
    from riskcausal.validation.select import run_selection

    run_selection(profile)


@app.command()
def ci(run_dir: str, n_rep: int = 2000, mean_block: float = 63) -> None:
    """Block-bootstrap and Jeffreys confidence intervals from a run's saved predictions (writes ci.json/ci.md)."""
    from riskcausal.evaluation.bootstrap import run_cis

    run_cis(run_dir, n_rep=n_rep, mean_block=mean_block)
    typer.echo(f"{run_dir}/ci.md")


@app.command("credit-regimes")
def credit_regimes_cmd(profile: str = "dev", series: str = "BAA10Y") -> None:
    """Exploratory B2: descriptive 2-regime Markov on a point-in-time credit spread (no models)."""
    import json

    from riskcausal.regimes.credit import credit_regimes

    typer.echo(json.dumps(credit_regimes(load_config(profile), series), indent=2))


@app.command("news-subset")
def news_subset(run_dir: str, path: str = "Stock_news/All_external.csv", lookback_days: int = 7) -> None:
    """Stream one FNSPID file once and keep only articles in evidence windows of a run (free public data)."""
    import time
    from pathlib import Path

    from riskcausal.config import PROJECT_ROOT
    from riskcausal.data.ingest_fnspid import evidence_dates, news_windows, stream_subset
    from riskcausal.registry import write_json

    dates = evidence_dates(run_dir)
    windows = news_windows(dates, lookback_days)
    out = PROJECT_ROOT / "data" / "raw" / "fnspid" / Path(run_dir).name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = stream_subset(path, windows)
    rows = res.pop("rows")
    rows.to_parquet(out / "news_subset.parquet")
    write_json(out / "manifest.json", {
        "source": f"huggingface.co/datasets/Zihan1004/FNSPID/{path}", "run_dir": str(run_dir),
        "n_evidence_dates": len(dates), "lookback_days": lookback_days,
        "evidence_dates_span": [str(dates.min().date()), str(dates.max().date())],
        "n_rows_kept": len(rows), "stream_secs": round(time.time() - t0, 1), **res,
        "columns_kept": list(rows.columns), "point_in_time_rule": "enforced at retrieval: published_utc <= "
        "16:00 America/New_York on the decision date"})
    log.info("kept %d of %d rows (%s bytes streamed) -> %s", len(rows), res["n_rows_total"], res["bytes"], out)


@app.command("diagnose-graphs")
def diagnose_graphs(profile: str = "dev", n_windows: int = 100, n_boot: int = 3, block: int = 10) -> None:
    """Noise floor for graph instability: change between bootstrap refits of the *same* window vs consecutive
    windows (proposal Risk 2). Writes into the graph cache directory."""
    from riskcausal.causal.diagnostics import noise_floor
    from riskcausal.pipeline import build_graphs

    cfg = load_config(profile)
    g, _, gdir = build_graphs(cfg)
    typer.echo(str(noise_floor(cfg, g, gdir, n_windows, n_boot, block)))


@app.command()
def report(run_dir: str) -> None:
    """Regenerate figures and metrics.md for an existing run from its saved artifacts."""
    from riskcausal.evaluation.plots import make_figures
    from riskcausal.evaluation.report import write_report

    make_figures(run_dir)
    typer.echo(str(write_report(run_dir)))


if __name__ == "__main__":
    app()
