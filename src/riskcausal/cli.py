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
def fetch_releases(profile: str = "dev") -> None:
    """Download ALFRED first-release dates for the profile's non-revised daily FRED series (free API key)."""
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
    summary = {"fetched": today}
    for sid in cfg.fred_series:
        if cfg.universe["fred"][sid]["revised"]:
            continue
        first = download_first_releases(sid, key, int(cfg.start_date[:4]), today)
        first.to_frame().to_parquet(out / f"{sid}.parquet")
        lag = pd.Series(sess.searchsorted(first.values) - sess.searchsorted(first.index), index=first.index)
        summary[sid] = {"tracked_from_obs": str(first.index.min().date()), "n_tracked": len(first),
                        "release_lag_sessions_counts": {str(k): int(v) for k, v in
                                                        lag.clip(upper=5).value_counts().sort_index().items()},
                        "share_released_more_than_1_session_late": float((lag > 1).mean()),
                        "max_release_lag_sessions": int(lag.max())}
        log.info("%s: first releases for %d observations (tracked from %s)", sid, len(first),
                 summary[sid]["tracked_from_obs"])
    write_json(out / "summary.json", summary)


@app.command()
def run(profile: str = "dev", experiment: str = typer.Option(None, help="overlay configs/experiments/<name>.yaml"),
        rebuild_panel: bool = True) -> None:
    """Full core pipeline: panel -> rolling graphs -> features -> walk-forward -> evaluation -> report."""
    from riskcausal.pipeline import run_pipeline

    run_dir = run_pipeline(load_config(profile, experiment=experiment), rebuild_panel=rebuild_panel)
    typer.echo(str(run_dir))


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
