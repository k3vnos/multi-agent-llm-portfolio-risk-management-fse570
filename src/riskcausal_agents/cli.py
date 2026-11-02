"""`python -m riskcausal_agents.cli <command>` — optional explanation layer (local Ollama only)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False, help="Grounded alert explanations (local LLM; never affects predictions).")
log = logging.getLogger("riskcausal_agents")
EXPLAINED = "bayes_logit:combined"
N_PER_STRATUM = 3


def _evidence(run_dir: Path):
    from riskcausal.config import PROJECT_ROOT
    from riskcausal_agents.evidence import RunEvidence

    news_dir = PROJECT_ROOT / "data" / "raw" / "fnspid" / run_dir.name
    news, end = None, None
    if (news_dir / "news_subset.parquet").exists():
        news = pd.read_parquet(news_dir / "news_subset.parquet")
        end = pd.Timestamp(json.loads((news_dir / "manifest.json").read_text())["max_utc"])
    return RunEvidence(run_dir, news, end)


def _backend(backend: str, model: str):
    from riskcausal_agents.backends import DisabledBackend, OllamaBackend

    if backend == "ollama":
        b = OllamaBackend(model=model)
        return b, b.model_digest()
    return DisabledBackend(), None


def select_eval_set(run_dir: Path, news_end: pd.Timestamp | None, seed: int = 570) -> pd.DataFrame:
    """Pre-specified stratified sample of alert episodes of the explained method: news coverage (yes/no) x
    retrospective outcome (true alert / false alarm). Labels are used ONLY for stratification and scoring."""
    from riskcausal.evaluation.metrics import alert_episodes
    from riskcausal.pipeline import MERGE_GAP

    P = pd.read_parquet(run_dir / "predictions.parquet")
    L = pd.read_parquet(run_dir / "labels_eval.parquet")
    g = P[P["method"] == EXPLAINED].sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(g["date"])
    tr = L.reindex(dates)
    on = np.flatnonzero(tr["onset"].fillna(0).to_numpy())
    st = tr["stress"].fillna(0).to_numpy()
    import yaml

    h = int(yaml.safe_load((run_dir / "config.yaml").read_text())["target"]["horizon"])
    rows = []
    for s, e in alert_episodes(g["alert"].to_numpy(bool), MERGE_GAP):
        if st[s] == 1:
            continue                                       # in-stress episodes are neither hits nor false alarms
        outcome = "true_alert" if ((on > s) & (on <= e + h)).any() else "false_alarm"
        cutoff = dates[s].tz_localize("America/New_York").replace(hour=16).tz_convert("UTC")
        covered = news_end is not None and cutoff <= news_end
        rows.append({"date": dates[s], "outcome": outcome, "news_covered": covered})
    ep = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    picks = []
    for (cov, out), grp in ep.groupby(["news_covered", "outcome"]):
        k = min(N_PER_STRATUM, len(grp))
        idx = np.sort(rng.choice(len(grp), size=k, replace=False))
        picks.append(grp.iloc[idx].assign(stratum=f"news={'yes' if cov else 'no'}|{out}"))
    return pd.concat(picks).sort_values("date").reset_index(drop=True)


@app.command()
def explain_one(run_dir: str, date: str, method: str = EXPLAINED, backend: str = "disabled",
                model: str = "qwen2.5:7b") -> None:
    """Explain one saved alert. Default backend is disabled (COST_POLICY); pass --backend ollama to run locally."""
    from riskcausal_agents.agents import explain, save

    rd = Path(run_dir)
    b, digest = _backend(backend, model)
    rec = explain(_evidence(rd).bundle(method, date), b, digest)
    typer.echo(str(save(rec, rd / "explanations")))


@app.command()
def eval_set(run_dir: str, backend: str = "disabled", model: str = "qwen2.5:7b") -> None:
    """Select the pre-specified evaluation set, explain each alert, write explanations/eval/eval_set.csv."""
    from riskcausal_agents.agents import explain, save

    rd = Path(run_dir)
    ev = _evidence(rd)
    sel = select_eval_set(rd, ev.news_end)
    b, digest = _backend(backend, model)
    rows = []
    for _, r in sel.iterrows():
        rec = explain(ev.bundle(EXPLAINED, r["date"]), b, digest)
        p = save(rec, rd / "explanations")
        n_news = sum(1 for e in rec["bundle"]["items"].values() if e["kind"] == "news")
        rows.append({"alert_id": rec["alert_id"], "date": str(r["date"].date()), "stratum": r["stratum"],
                     "file": p.name, "n_news_items": n_news,
                     "n_claims": len(rec["explanation"]["claims"]),
                     "n_rejected_total": sum(len(s.get("rejected", [])) for s in rec["specialists"])
                     + len(rec["synthesis"].get("rejected", [])),
                     "abstain": rec["explanation"]["abstain"],
                     # manual rubric columns (M6_EVAL_RUBRIC.md), filled by a human checker
                     "R2_faithful_claims": "", "R3_hindsight_free": "", "R5_uncertainty_adequate": "",
                     "R6_abstention_appropriate": "", "checker_notes": "", "checked_by": ""})
        log.info("explained %s (%s): %d claims, abstain=%s", rec["alert_id"], r["stratum"], rows[-1]["n_claims"],
                 rows[-1]["abstain"])
    out = rd / "explanations" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "eval_set.csv", index=False)
    (out / "selection.json").write_text(json.dumps({
        "method": EXPLAINED, "n_per_stratum": N_PER_STRATUM, "seed": 570, "backend": b.name,
        "model_digest": digest, "news_coverage_end": str(ev.news_end),
        "strata_counts": sel["stratum"].value_counts().to_dict()}, indent=2))
    typer.echo(str(out / "eval_set.csv"))


if __name__ == "__main__":
    from riskcausal.logging_utils import setup_logging

    setup_logging("INFO")
    app()
