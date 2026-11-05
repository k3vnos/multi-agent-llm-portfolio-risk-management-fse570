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


def select_eval_set(run_dir: Path, news_end: pd.Timestamp | None, seed: int = 570,
                    exclude: set[str] | None = None) -> pd.DataFrame:
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
    if exclude:
        ep = ep[~ep["date"].dt.strftime("%Y-%m-%d").isin(exclude)]
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
def eval_set(run_dir: str, backend: str = "disabled", model: str = "qwen2.5:7b", seed: int = 570,
             name: str = "eval", exclude_from: str = typer.Option(
                 None, help="eval_set.csv whose alerts are excluded (held-out set)")) -> None:
    """Select a pre-specified evaluation set, explain each alert, write explanations/<name>/eval_set.csv."""
    from riskcausal_agents.agents import explain, save

    rd = Path(run_dir)
    ev = _evidence(rd)
    excl = set(pd.read_csv(exclude_from)["date"]) if exclude_from else None
    sel = select_eval_set(rd, ev.news_end, seed=seed, exclude=excl)
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
    out = rd / "explanations" / name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "eval_set.csv", index=False)
    (out / "selection.json").write_text(json.dumps({
        "method": EXPLAINED, "n_per_stratum": N_PER_STRATUM, "seed": seed, "excluded_from": exclude_from,
        "prompt_version": __import__("riskcausal_agents.agents", fromlist=["x"]).PROMPT_VERSION, "backend": b.name,
        "model_digest": digest, "news_coverage_end": str(ev.news_end),
        "strata_counts": sel["stratum"].value_counts().to_dict()}, indent=2))
    typer.echo(str(out / "eval_set.csv"))


V4_DIR = "outputs/agents_v4"


def episode_starts(run_dir: Path) -> list[pd.Timestamp]:
    """First day of every alert episode of the explained method (no outcome labels involved)."""
    from riskcausal.evaluation.metrics import alert_episodes
    from riskcausal.pipeline import MERGE_GAP

    P = pd.read_parquet(run_dir / "predictions.parquet")
    g = P[P["method"] == EXPLAINED].sort_values("date").reset_index(drop=True)
    return [pd.Timestamp(g["date"].iloc[s]) for s, _ in alert_episodes(g["alert"].to_numpy(bool), MERGE_GAP)]


def v4_dates(run_dir: Path, which: str, seed: int = 572, n_quiet: int = 12) -> pd.DataFrame:
    """tune  = the M6 tuning alerts (seed 570), the only dates used while developing the v4 prompts;
    heldout = every alert episode never used by M6 or v4 development, plus `n_quiet` random non-alert days (seed 572)
              so the "below alert line" path is also tested — scored once, after the prompts were frozen;
    all     = every alert episode (for the saved explanations shown in the app)."""
    used = {n: set(pd.read_csv(run_dir / "explanations" / n / "eval_set.csv")["date"]) for n in ("eval", "heldout")}
    eps = [d.strftime("%Y-%m-%d") for d in episode_starts(run_dir)]
    if which == "tune":
        return pd.DataFrame({"date": sorted(used["eval"]), "group": "m6_tuning_alert"})
    if which == "all":
        return pd.DataFrame({"date": eps, "group": ["m6_tuning_alert" if d in used["eval"] else "m6_heldout_alert"
                                                    if d in used["heldout"] else "fresh_alert" for d in eps]})
    fresh = [d for d in eps if d not in used["eval"] | used["heldout"]]
    P = pd.read_parquet(run_dir / "predictions.parquet")
    g = P[(P["method"] == EXPLAINED) & ~P["alert"]].sort_values("date")
    quiet = sorted(np.random.default_rng(seed).choice(g["date"].dt.strftime("%Y-%m-%d").to_numpy(), n_quiet,
                                                      replace=False))
    return pd.DataFrame({"date": fresh + list(quiet), "group": ["fresh_alert"] * len(fresh) + ["fresh_no_alert"] * len(quiet)})


@app.command()
def v4_eval(run_dir: str, which: str = "tune", model: str = "qwen2.5:7b", limit: int = 0) -> None:
    """Run the v4 structured agents (local Ollama) on a date set; write outputs/agents_v4/<which>/ (the locked
    run directory is only read)."""
    from riskcausal.config import PROJECT_ROOT
    from riskcausal_agents.backends import OllamaBackend
    from riskcausal_agents.structured import PROMPT_VERSION, explain, summarise

    rd = Path(run_dir)
    ev = _evidence(rd)
    thr = float(ev.cfg.graph.edge_threshold)
    sel = v4_dates(rd, which)
    if limit:
        sel = sel.head(limit)
    b = OllamaBackend(model=model)
    digest = b.model_digest()
    out = PROJECT_ROOT / V4_DIR / which
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in sel.iterrows():
        # identical inputs + frozen prompt + fixed seed give an identical record: reuse one made for another set
        prev = [p for p in (PROJECT_ROOT / V4_DIR).glob(f"*/{r['date']}.json") if p.parent.name != which]
        rec = next((x for x in (json.loads(p.read_text()) for p in prev)
                    if x["prompt_version"] == PROMPT_VERSION and x["model_digest"] == digest), None)
        rec = rec or explain(ev.bundle(EXPLAINED, r["date"]), b, thr, model_digest=digest)
        rec["group"] = r["group"]
        (out / f"{r['date']}.json").write_text(json.dumps(rec, indent=2, default=str))
        rows.append({"date": r["date"], "group": r["group"], **{k: (";".join(v) if isinstance(v, list) else v)
                                                                 for k, v in summarise(rec).items()}, "secs": rec["secs"]})
        log.info("v4 %s %s: %s", which, r["date"], rows[-1])
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    (out / "selection.json").write_text(json.dumps({
        "which": which, "run_id": rd.name, "method": EXPLAINED, "prompt_version": PROMPT_VERSION, "model": model,
        "model_digest": digest, "n": len(rows), "groups": sel["group"].value_counts().to_dict(),
        "doc": v4_dates.__doc__}, indent=2))
    typer.echo(str(out / "summary.csv"))


def _reason_class(r: str) -> str:
    return r.split(":")[0].replace("note ", "note: ").strip()


def v4_score(recs: list[dict]) -> dict:
    """Automatic scores of v4 records (no hindsight labels involved)."""
    from collections import Counter

    asserted = correct = 0
    by_kind, rej, note_rej = Counter(), Counter(), Counter()
    ok_kind = Counter()
    notes_kept = notes_total = syn_called = syn_valid = calls = 0
    abst, model_abst, empty = Counter(), Counter(), 0
    for r in recs:
        facts = {f["id"]: f for f in r["facts"]}
        for s in r["specialists"]:
            calls += s["called_model"]
            if s["abstain"]:
                abst[s["role"]] += 1
                model_abst[s["role"]] += bool(s.get("model_abstained"))
            for c in s["claims"]:
                k = facts[c["fact_id"]]["kind"]
                asserted += 1; correct += 1; by_kind[k] += 1; ok_kind[k] += 1
                if c.get("note") or c.get("note_rejected"):
                    notes_total += 1
                    notes_kept += bool(c.get("note"))
                    for x in (c.get("note_rejected") or {}).get("reasons", []):
                        note_rej[_reason_class(x)] += 1
            for c in s["rejected"]:
                k = facts.get(c["fact_id"], {}).get("kind", "unknown id")
                asserted += 1; by_kind[k] += 1
                for x in c["reasons"]:
                    rej[_reason_class(x)] += 1
        syn = r["synthesis"]
        calls += syn["called_model"]
        syn_called += syn["called_model"]; syn_valid += bool(syn.get("validated"))
        empty += not any(s["claims"] for s in r["specialists"])
    return {"n_dates": len(recs), "model_calls": calls, "claims_asserted": asserted, "labels_correct": correct,
            "label_accuracy_before_validation": round(correct / asserted, 4) if asserted else None,
            "label_accuracy_by_kind": {k: f"{ok_kind[k]}/{by_kind[k]}" for k in sorted(by_kind)},
            "rejected_claims_by_reason": dict(rej), "notes_written": notes_total, "notes_passed_checks": notes_kept,
            "notes_rejected_by_reason": dict(note_rej), "synthesis_called": syn_called, "synthesis_validated": syn_valid,
            "abstentions_by_agent": dict(abst), "model_initiated_abstentions": dict(model_abst),
            "dates_with_no_confirmed_fact": empty,
            "shown_statement_faithfulness": "by construction: shown statements are rendered from facts whose label the "
                                            "validator confirmed; model notes are scored separately by hand"}


@app.command()
def v4_report(which: str = "heldout") -> None:
    """Score a v4 set: writes outputs/agents_v4/<which>/scores.json and notes_handcheck.csv (one row per model note
    that passed the automatic checks, for the hand faithfulness check)."""
    from riskcausal.config import PROJECT_ROOT

    d = PROJECT_ROOT / V4_DIR / which
    recs = [json.loads(p.read_text()) for p in sorted(d.glob("20*.json"))]
    out = {"set": which, "overall": v4_score(recs)}
    for g in sorted({r.get("group") for r in recs}):
        out[f"group:{g}"] = v4_score([r for r in recs if r.get("group") == g])
    (d / "scores.json").write_text(json.dumps(out, indent=2))
    rows = []
    for r in recs:
        facts = {f["id"]: f for f in r["facts"]}
        for s in r["specialists"]:
            rows += [{"date": r["decision_date"], "agent": s["role"], "fact_id": c["fact_id"], "label": c["label"],
                      "fact_text": facts[c["fact_id"]]["text"][:300], "note": c["note"], "faithful": "", "why": ""}
                     for c in s["claims"] if c.get("note")]
        if r["synthesis"].get("note"):
            rows.append({"date": r["decision_date"], "agent": "synthesis", "fact_id": ",".join(r["synthesis"]["key_facts"]),
                         "label": r["synthesis"]["overall"], "fact_text": (r["synthesis"]["text"] or "")[:300],
                         "note": r["synthesis"]["note"], "faithful": "", "why": ""})
    hc = d / "notes_handcheck.csv"
    if not hc.exists():                                  # never overwrite a completed hand check
        pd.DataFrame(rows).to_csv(hc, index=False)
    typer.echo(json.dumps(out["overall"], indent=2))


if __name__ == "__main__":
    from riskcausal.logging_utils import setup_logging

    setup_logging("INFO")
    app()
