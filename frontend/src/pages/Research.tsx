import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type MethodRow, type Research as R } from "../api";
import { Card, ErrorState, Info, Loading, Pill } from "../components/ui";
import { fmt, methodName } from "../format";
import { G } from "../glossary";
import { useAsync } from "../hooks";

export default function Research() {
  const r = useAsync(() => api.research(), []);
  if (r.error) return <ErrorState message={r.error} />;
  if (!r.data) return <Loading />;
  const d = r.data;
  return (
    <div className="space-y-4">
      <section className="rounded-xl border border-crit/40 bg-crit/10 p-5" data-testid="conclusion">
        <div className="text-[10.5px] font-medium uppercase tracking-[0.14em] text-[#ff9c9c]">Main research finding</div>
        <p className="mt-1.5 text-[15px] leading-relaxed text-ink">{d.conclusion}</p>
        <p className="mt-1 text-xs text-muted">Locked full-profile evaluation · settings chosen on validation data only · one scoring of the test period ·
          {" "}{d.primary.data.methods[0] ? `${d.primary.data.methods.find((m) => m.method === "vix")?.n_onsets} stress onsets` : ""}</p>
      </section>
      <TrackRecord rows={d.primary.data.methods} span={d.primary.data.span} />
      <div className="grid gap-4 lg:grid-cols-2"><CiChart rows={d.primary.data.methods} /><Differences d={d} /></div>
      <div className="grid gap-4 lg:grid-cols-2"><LeadTimes d={d} /><Exploratory d={d} /></div>
      <Integrity d={d} />
    </div>
  );
}

const ORDER = ["vix", "bayes_logit:controls", "dcc_var", "bayes_logit:combined", "inst_composite", "gbm:combined", "gbm:controls",
               "bayes_logit:granger", "gbm:instability", "gbm:granger", "corr_var", "nfci", "dcc_corr", "bayes_logit:instability"];
const ci = (c: [number, number] | null, d = 3) => (c ? `[${c[0].toFixed(d)}, ${c[1].toFixed(d)}]` : "—");
const isGraph = (m: string) => m.includes("instability") || m === "inst_composite" || m.includes("granger") || m.includes("combined");

function TrackRecord({ rows, span }: { rows: MethodRow[]; span: [string, string] }) {
  const sorted = [...rows].sort((a, b) => (b.roc_auc ?? 0) - (a.roc_auc ?? 0));
  return (
    <Card title="Track record against baselines" subtitle={`out-of-sample ${span[0]} → ${span[1]} · alert rule: ≤4 false alarms/yr and ≤10% of calm days on validation data`}
      info="Every number is read from the saved metrics and confidence-interval files of the locked M5 run. Highlighted rows use the graph-instability measures.">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[900px] text-left text-xs" data-testid="track-record">
          <thead><tr className="text-[10.5px] uppercase tracking-wider text-dim">
            <th className="py-2 font-medium">Signal</th><th className="font-medium">ROC-AUC <Info text={G.rocauc} /></th><th className="font-medium">95% CI <Info text={G.ci} /></th>
            <th className="font-medium">PR-AUC <Info text={G.prauc} /></th><th className="font-medium">Onsets detected</th><th className="font-medium">Precision</th>
            <th className="font-medium">False alarms / yr</th><th className="font-medium">Median lead (days)</th></tr></thead>
          <tbody>
            {sorted.map((m) => (
              <tr key={m.method} className={`border-t border-line ${isGraph(m.method) ? "bg-accent/[0.06]" : ""}`} data-method={m.method}>
                <td className="py-2 pr-3 text-ink">{methodName(m.method)}{m.method === "vix" && <> <Pill tone="good">benchmark</Pill></>}</td>
                <td className="font-mono text-ink">{fmt(m.roc_auc, 3)}</td><td className="font-mono text-muted">{ci(m.roc_ci95)}</td>
                <td className="font-mono text-muted">{fmt(m.pr_auc, 3)} <span className="text-dim">/ base {fmt(m.base_rate, 3)}</span></td>
                <td className="font-mono text-muted">{m.n_detected}/{m.n_onsets} <span className="text-dim">{ci(m.recall_ci95, 2)}</span></td>
                <td className="font-mono text-muted">{fmt(m.precision, 2)}</td><td className="font-mono text-muted">{fmt(m.false_alarms_per_year, 2)}</td>
                <td className="font-mono text-muted">{fmt(m.median_lead, 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function CiChart({ rows }: { rows: MethodRow[] }) {
  const sorted = ORDER.map((m) => rows.find((r) => r.method === m)).filter(Boolean) as MethodRow[];
  const x = (v: number) => ((v - 0.3) / (0.9 - 0.3)) * 100;
  const vix = rows.find((r) => r.method === "vix")?.roc_auc ?? 0;
  return (
    <Card title="ROC-AUC with 95% confidence intervals" info={G.ci} subtitle="dashed: VIX and chance (0.5)">
      <div className="space-y-1.5" data-testid="ci-chart">
        {sorted.map((m) => (
          <div key={m.method} className="grid grid-cols-[170px_1fr_44px] items-center gap-2 text-[11px]">
            <span className="truncate text-muted" title={methodName(m.method)}>{methodName(m.method).replace(" (baseline)", "")}</span>
            <div className="relative h-4">
              <div className="absolute inset-y-0 border-l border-dashed border-dim" style={{ left: `${x(0.5)}%` }} />
              <div className="absolute inset-y-0 border-l border-dashed border-vix" style={{ left: `${x(vix)}%` }} />
              {m.roc_ci95 && <div className="absolute top-1/2 h-[2px] -translate-y-1/2 bg-muted" style={{ left: `${x(m.roc_ci95[0])}%`, width: `${x(m.roc_ci95[1]) - x(m.roc_ci95[0])}%` }} />}
              <div className="absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full" style={{ left: `${x(m.roc_auc ?? 0)}%`, background: isGraph(m.method) ? "#9085e9" : "#d95926" }} />
            </div>
            <span className="text-right font-mono text-ink">{fmt(m.roc_auc, 3)}</span>
          </div>
        ))}
        <div className="grid grid-cols-[170px_1fr_44px] gap-2 text-[10px] text-dim"><span /><div className="flex justify-between"><span>0.30</span><span>0.50</span><span>0.70</span><span>0.90</span></div><span /></div>
        <div className="flex gap-4 pt-1 text-[11px] text-muted"><span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-sii" />uses graph instability</span><span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-vix" />baseline / controls only</span></div>
      </div>
    </Card>
  );
}

function Differences({ d }: { d: R }) {
  const keys = ["bayes_logit:instability - vix", "inst_composite - vix", "dcc_var - vix", "bayes_logit:controls - vix",
                "bayes_logit:combined - bayes_logit:controls", "gbm:combined - gbm:controls"];
  const rows = keys.map((k) => d.primary.data.differences.find((x) => x.comparison === k)).filter(Boolean) as R["primary"]["data"]["differences"];
  const name = (c: string) => c.split(" - ").map((m) => methodName(m).replace(" (baseline)", "")).join("  minus  ");
  return (
    <Card title="Paired comparisons" subtitle="difference in ROC-AUC, 95% CI" info="A difference is called significant only if its 95% interval excludes zero.">
      <ul className="space-y-2 text-xs" data-testid="differences">
        {rows.map((x) => (
          <li key={x.comparison} className="rounded-md border border-line bg-panel2 px-3 py-2">
            <div className="text-muted">{name(x.comparison)}</div>
            <div className="mt-0.5 flex items-center justify-between"><span className="font-mono text-ink">{x.delta_roc >= 0 ? "+" : ""}{x.delta_roc.toFixed(3)} {ci(x.roc_ci95)}</span>
              {x.excludes_zero ? <Pill tone={x.delta_roc < 0 ? "crit" : "good"}>{x.delta_roc < 0 ? "significantly worse" : "significantly better"}</Pill> : <Pill>not distinguishable from 0</Pill>}</div>
          </li>
        ))}
      </ul>
    </Card>
  );
}

function LeadTimes({ d }: { d: R }) {
  const [m, setM] = useState("vix");
  const data = useMemo(() => {
    const xs = d.lead_times[m] || [];
    return Array.from({ length: 10 }, (_, i) => ({ lead: i + 1, n: xs.filter((v) => v === i + 1).length }));
  }, [d, m]);
  const row = d.primary.data.methods.find((r) => r.method === m);
  return (
    <Card title="Lead-time distribution" subtitle="trading days between the first alert and a detected stress onset"
      right={<select value={m} onChange={(e) => setM(e.target.value)} aria-label="Lead-time method" className="rounded-md border border-line bg-panel2 px-2 py-1 text-xs text-ink [color-scheme:dark]">
        {ORDER.map((k) => <option key={k} value={k}>{methodName(k)}</option>)}</select>}>
      <div style={{ height: 200 }} data-testid="lead-chart">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -20 }}>
            <CartesianGrid stroke="#1f2733" vertical={false} />
            <XAxis dataKey="lead" stroke="#5d6673" tick={{ fill: "#8a93a0", fontSize: 11 }} tickLine={false} />
            <YAxis allowDecimals={false} stroke="#5d6673" tick={{ fill: "#8a93a0", fontSize: 11 }} tickLine={false} axisLine={false} />
            <Tooltip contentStyle={{ background: "#161c25", border: "1px solid #1f2733", fontSize: 12 }} labelFormatter={(l) => `${l} trading days early`} formatter={(v: number) => [v, "onsets"]} />
            <Bar dataKey="n" fill={isGraph(m) ? "#9085e9" : "#d95926"} radius={[4, 4, 0, 0]} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-2 text-xs text-muted">{row ? `${row.n_detected} of ${row.n_onsets} onsets detected; ` : ""}a lead of 10 means the alert was already on 10 days before the onset (the maximum the evaluation counts).</p>
    </Card>
  );
}

function Exploratory({ d }: { d: R }) {
  const rows = [...d.exploratory.data.methods].sort((a, b) => (b.roc_auc ?? 0) - (a.roc_auc ?? 0)).slice(0, 8);
  return (
    <Card title="Exploratory: regional-banking stress (KRE volatility)" right={<Pill tone="accent">exploratory</Pill>}
      info="A separately defined target, never used for model selection. Under it, 9 March 2023 (the SVB week) is a stress onset; under the primary SPY target it is not.">
      <table className="w-full text-left text-xs" data-testid="exploratory">
        <thead><tr className="text-[10.5px] uppercase tracking-wider text-dim"><th className="py-1.5 font-medium">Signal</th><th className="font-medium">ROC-AUC</th><th className="font-medium">95% CI</th><th className="font-medium">Detected</th></tr></thead>
        <tbody>{rows.map((m) => <tr key={m.method} className="border-t border-line"><td className="py-1.5 text-ink">{methodName(m.method)}</td><td className="font-mono">{fmt(m.roc_auc, 3)}</td><td className="font-mono text-muted">{ci(m.roc_ci95)}</td><td className="font-mono text-muted">{m.n_detected}/{m.n_onsets}</td></tr>)}</tbody>
      </table>
      <p className="mt-2 text-xs text-muted">Same conclusion: instability measures do not beat VIX or the control model. Before the SVB onset only the DCC correlation baseline was alerting — a single event, anecdotal.</p>
    </Card>
  );
}

function Integrity({ d }: { d: R }) {
  const audit = d.primary.data.audit;
  return (
    <Card title="Leakage audit, reproducibility and provenance" info="These checks recompute timing properties from raw data for the locked run.">
      <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
        <ul className="space-y-1.5 text-xs" data-testid="audit">
          {audit.map((c) => (
            <li key={c.check} className="flex items-start justify-between gap-3 border-b border-line pb-1.5"><span className="text-muted">{c.check}</span>
              <span className={`whitespace-nowrap font-mono ${c.passed ? "text-[#5bd65b]" : "text-[#ff8a8a]"}`}>{c.passed ? "✓ pass" : "✗ fail"} · {c.n_violations}/{c.n_checked.toLocaleString()}</span></li>
          ))}
        </ul>
        <div className="space-y-2 text-xs text-muted">
          <div>Reproducibility: <span className={d.reproducibility.status === "bit-identical" ? "text-[#5bd65b]" : "text-[#ff8a8a]"}>{d.reproducibility.status}</span> versus an independent re-run.</div>
          <div>Locked run: <span className="font-mono text-ink">{d.primary.meta.run_id}</span></div>
          <div>Code version: <span className="font-mono text-ink">{d.primary.meta.git_sha.slice(0, 10)}</span></div>
          <div>Exploratory run: <span className="font-mono text-ink">{d.exploratory.meta.run_id}</span></div>
          <div>Truth labels: {d.primary.data.truth}</div>
        </div>
      </div>
    </Card>
  );
}
