"""metrics.md generated from metrics.json — the only producer of reported tables."""

from __future__ import annotations

from pathlib import Path

from riskcausal.registry import read_json


def _f(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _table(block: dict) -> list[str]:
    L = [("| method | onsets | detected | recall | precision | false alarms/yr | median lead (sessions) | "
         "ROC-AUC | PR-AUC | base rate | score coverage |"), "|" + "---|" * 11]
    for m, r in sorted(block.items()):
        e, d = r["event"], r["day"]
        L.append(f"| {m} | {e['n_onsets']} | {e['n_detected']} | {_f(e['recall'])} | {_f(e['precision'])} | "
                 f"{_f(e['false_alarms_per_year'], 2)} | {_f(e['median_lead'], 1)} | {_f(d.get('roc_auc'))} | "
                 f"{_f(d.get('pr_auc'))} | {_f(d.get('base_rate'))} | {_f(r['score_coverage'], 2)} |")
    return L


def write_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    M = read_json(run_dir / "metrics.json")
    prov = read_json(run_dir / "provenance.json")
    L = [f"# Run report — `{prov['run_id']}`", "",
         (f"_Generated from `metrics.json` (git {prov['git_sha']}, config {prov['config_hash']}, "
         f"synthetic={prov['synthetic']}). Do not edit by hand._"), "",
         f"- Out-of-sample span: {M['test_span'][0]} → {M['test_span'][1]} (annual walk-forward folds)",
         f"- Truth: {M['truth']}",
         (f"- Retrospective regimes: annualised vol {[round(v, 3) for v in M['label_eval_summary']['regime_annualised_vol']]}, "
         f"stress share {M['label_eval_summary']['stress_share']:.3f}"), "",
         "## Full out-of-sample span", ""] + _table(M["full_span"])
    if M.get("common_span"):
        L += ["", f"## Common span (all methods scored): {M['common_span'][0]} → {M['common_span'][1]}", ""]
        L += _table(M["common_span_metrics"])
    L += ["", "## Sensitivity: truth = filtered (real-time) Markov labels with training-period parameters", ""]
    L += _table(M["sensitivity_filtered_labels"])
    L += ["", "## Bayesian logit: posterior ranges", ""]
    for m, r in sorted(M["bayes_lead_intervals"].items()):
        q = r["recall_q05_q50_q95"]
        fa = r["fa_per_year_q05_q50_q95"]
        L.append(f"- **{m}** ({r['n_draws']} draws): recall 5/50/95% = {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}; "
                 f"false alarms/yr = {fa[0]:.2f}/{fa[1]:.2f}/{fa[2]:.2f}")
    L += ["", "Figures: `figures/case_2020.png`, `figures/case_2023.png`, `figures/timeline_full_test.png`.",
          "", "_Directed-dependence graphs are estimates under LiNGAM assumptions, not proven causal structure._"]
    out = run_dir / "metrics.md"
    out.write_text("\n".join(L))
    return out
