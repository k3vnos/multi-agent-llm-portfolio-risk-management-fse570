"""metrics.md generated from metrics.json — the only producer of reported tables."""

from __future__ import annotations

from pathlib import Path

from riskcausal.registry import read_json


def _f(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _table(block: dict) -> list[str]:
    L = [("| method | onsets | detected | recall | precision | false alarms/yr | calm-day alert share | "
          "median lead (sessions) | ROC-AUC | PR-AUC | base rate | score coverage |"), "|" + "---|" * 12]
    for m, r in sorted(block.items()):
        e, d = r["event"], r["day"]
        L.append(f"| {m} | {e['n_onsets']} | {e['n_detected']} | {_f(e['recall'])} | {_f(e['precision'])} | "
                 f"{_f(e['false_alarms_per_year'], 2)} | {_f(e.get('calm_alert_share'), 2)} | "
                 f"{_f(e['median_lead'], 1)} | {_f(d.get('roc_auc'))} | "
                 f"{_f(d.get('pr_auc'))} | {_f(d.get('base_rate'))} | {_f(r['score_coverage'], 2)} |")
    return L


def _rule_name(M: dict, rule: str) -> str:
    r = M["alert_rules"][rule]
    budget = "" if r["max_calm_alert_share"] is None else f" + alerts on <= {r['max_calm_alert_share']:.0%} of validation calm days"
    tag = " (primary)" if rule == M["primary_rule"] else " (original H0 rule, kept for comparison)"
    return f"`{rule}`: <= {r['max_fa_per_year']:g} false-alarm episodes/yr{budget}{tag}"


def write_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    M = read_json(run_dir / "metrics.json")
    prov = read_json(run_dir / "provenance.json")
    au = M.get("audit", {})
    L = [f"# Run report — `{prov['run_id']}`", "",
         (f"_Generated from `metrics.json` (git {prov['git_sha']}, config {prov['config_hash']}, "
          f"synthetic={prov['synthetic']}). Do not edit by hand._"), "",
         f"- Experiment: {M.get('experiment') or 'none (H0 settings)'}; Markov regimes: {M['n_regimes']}",
         f"- Out-of-sample span: {M['test_span'][0]} → {M['test_span'][1]} (annual walk-forward folds)",
         f"- Truth: {M['truth']}",
         (f"- Retrospective regimes: annualised vol {[round(v, 3) for v in M['label_eval_summary']['regime_annualised_vol']]}, "
          f"stress share {M['label_eval_summary']['stress_share']:.3f}, "
          f"{len(M['label_eval_summary']['onsets'])} onsets"),
         (f"- Leakage audit: **{'PASSED' if au.get('passed') else 'FAILED'}** ({len(au.get('checks', []))} checks, "
          "`audit.json`)"), ""]
    L += [("_ROC-AUC, PR-AUC and base rate are threshold-free and identical under every alert rule; only the "
           "event-level columns change between the rule sections below._")]
    for rule in M["by_rule"]:
        R = M["by_rule"][rule]
        L += ["", f"## Alert rule {_rule_name(M, rule)}", ""]
        if M.get("common_span"):
            L += [f"### Common span (all methods scored): {M['common_span'][0]} → {M['common_span'][1]}", ""]
            L += _table(R["common_span_metrics"])
        L += ["", "### Full out-of-sample span", ""] + _table(R["full_span"])
        L += ["", "### Sensitivity: truth = filtered (real-time) Markov labels with training-period parameters", ""]
        L += _table(R["sensitivity_filtered_labels"])
        L += ["", "### Bayesian logit: posterior ranges", ""]
        for m, r in sorted(R["bayes_lead_intervals"].items()):
            q = r["recall_q05_q50_q95"]
            fa = r["fa_per_year_q05_q50_q95"]
            L.append(f"- **{m}** ({r['n_draws']} draws): recall 5/50/95% = {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}; "
                     f"false alarms/yr = {fa[0]:.2f}/{fa[1]:.2f}/{fa[2]:.2f}")
    L += ["", "## Inner-validation spans (median over folds; truth = labels fit on training data)", "",
          "| method | folds | median ROC-AUC | median PR-AUC | median base rate |", "|---|---|---|---|---|"]
    for m, r in sorted(M["validation_median_over_folds"].items()):
        L.append(f"| {m} | {r['n_folds']} | {_f(r['median_roc_auc'])} | {_f(r['median_pr_auc'])} | "
                 f"{_f(r['median_base_rate'])} |")
    L += ["", "## Leakage audit checks", "", "| check | passed | violations | checked |", "|---|---|---|---|"]
    for c in au.get("checks", []):
        L.append(f"| {c['check']} | {c['passed']} | {c['n_violations']} | {c['n_checked']} |")
    L += ["", ("Figures: `figures/case_2020.png`, `figures/case_2023.png`, `figures/timeline_full_test.png` "
               "(alerts shown under the primary rule)."),
          "", "_Directed-dependence graphs are estimates under LiNGAM assumptions, not proven causal structure._"]
    out = run_dir / "metrics.md"
    out.write_text("\n".join(L))
    return out
