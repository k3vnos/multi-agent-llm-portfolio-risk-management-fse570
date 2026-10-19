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


if __name__ == "__main__":
    app()
