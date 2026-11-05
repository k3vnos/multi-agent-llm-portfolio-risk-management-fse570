import { useEffect, useMemo, useState } from "react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { live, type PortfolioAnalysis, type Universe } from "../api";
import { AskAgents } from "../components/AskAgents";
import { C, monthTicks } from "../components/charts";
import { ReplayBar } from "../components/Shell";
import { Details, SimpleReplay, ToneChip } from "../components/simple";
import { ErrorState } from "../components/ui";
import { longDate, signed } from "../format";
import { useAsync } from "../hooks";
import { short, tone, toneWord } from "../plain";
import { formatSpec, normalise, parseSpec, problems, riskSentences, total, type Row } from "../portfolio";
import { useRoute } from "../state";

/** Portfolio Lens (exploratory). Simple Mode = "My portfolio"; Advanced Mode = "Portfolio Lens" with every number. */
export default function PortfolioLens({ simple = false }: { simple?: boolean }) {
  const { date, pf } = useRoute();
  const uni = useAsync(() => live.universe(), []);
  const a = useAsync(() => (pf ? live.analyze(pf, date) : Promise.resolve(null)), [pf, date]);
  const t = simple ? "text-[16px]" : "text-[13px]";
  return (
    <div className="space-y-5">
      {simple ? <SimpleReplay /> : <ReplayBar />}
      <div className={`rounded-xl border border-warn/50 bg-warn/10 px-4 py-3 ${t} text-warn`} data-testid="exploratory-banner">
        <b>Exploratory feature.</b> Portfolio Lens is separate from the project's locked research evaluation. Numbers are
        historical — computed only from daily prices up to {longDate(date)} — and are not forecasts or investment advice.
      </div>
      {uni.error ? <ErrorState message={uni.error} /> : uni.data && <Editor uni={uni.data} simple={simple} />}
      {pf && (a.error ? <ErrorState message={a.error} /> : !a.data ? <div className={`${t} text-muted`} role="status">Calculating…</div> : (
        <>
          <Metrics a={a.data} simple={simple} />
          <div className="grid gap-5 xl:grid-cols-2">
            <Contributions a={a.data} simple={simple} />
            <NetworkLinks a={a.data} simple={simple} />
          </div>
          <AskAgents compact={simple} portfolio={pf} />
          <Details label="How these numbers are calculated" open={!simple}>
            <pre className="whitespace-pre-wrap font-sans text-[13px] text-muted">{a.data.definitions}</pre>
          </Details>
        </>
      ))}
    </div>
  );
}

function Editor({ uni, simple }: { uni: Universe; simple: boolean }) {
  const { pf, setPf } = useRoute();
  const [rows, setRows] = useState<Row[]>(() => parseSpec(pf));
  useEffect(() => { setRows(parseSpec(pf)); }, [pf]);
  const tickers = uni.etfs.map((e) => e.ticker);
  const issues = problems(rows, tickers);
  const tot = total(rows);
  const dirty = formatSpec(rows) !== (pf || "");
  const set = (i: number, r: Partial<Row>) => setRows((rs) => rs.map((x, k) => (k === i ? { ...x, ...r } : x)));
  const t = simple ? "text-[16px]" : "text-[13px]";
  const input = "rounded-lg border border-line bg-panel2 px-2 py-1.5 text-ink [color-scheme:dark]";
  return (
    <section className={`rounded-2xl border border-line bg-panel ${simple ? "p-6" : "p-4"}`} data-testid="portfolio-editor">
      <h2 className={`${simple ? "text-[22px]" : "text-[16px]"} font-semibold text-ink`}>Build a portfolio from the project's 25 funds</h2>
      <div className={`mt-3 flex flex-wrap items-center gap-2 ${t}`}>
        <span className="text-muted">Start from an example:</span>
        {uni.examples.map((ex) => (
          <button key={ex.id} type="button" title={ex.about} onClick={() => setPf(ex.spec)}
            className={`rounded-lg border px-3 py-1.5 ${pf === ex.spec ? "border-accent bg-accent/20 text-ink" : "border-line bg-panel2 text-ink hover:border-accent"}`}>
            {ex.name} <span className="text-[12px] text-muted">(example)</span></button>
        ))}
      </div>
      <table className={`mt-4 w-full max-w-2xl ${t}`}>
        <thead><tr className="text-left text-[12px] uppercase tracking-wider text-muted"><th className="pb-1 font-medium">Fund</th><th className="pb-1 font-medium">Weight %</th><th /></tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} data-testid="holding-row">
              <td className="py-1 pr-2">
                <select aria-label={`Fund ${i + 1}`} value={r.ticker} onChange={(e) => set(i, { ticker: e.target.value })} className={`${input} w-full`}>
                  <option value="">Choose…</option>
                  {uni.etfs.map((e) => <option key={e.ticker} value={e.ticker}>{e.ticker} — {e.name}</option>)}
                </select>
              </td>
              <td className="py-1 pr-2"><input aria-label={`Weight ${i + 1}`} type="number" min={0} max={100} step={1} value={Number.isFinite(r.weight) ? r.weight : ""}
                onChange={(e) => set(i, { weight: e.target.value === "" ? NaN : Number(e.target.value) })} className={`${input} w-24 font-mono`} /></td>
              <td className="py-1"><button type="button" aria-label={`Remove ${r.ticker || "row"}`} onClick={() => setRows((rs) => rs.filter((_, k) => k !== i))} className="px-2 text-muted hover:text-crit">✕</button></td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className={`mt-2 flex flex-wrap items-center gap-3 ${t}`}>
        <button type="button" onClick={() => setRows((rs) => [...rs, { ticker: "", weight: 0 }])} disabled={rows.length >= 10} className="rounded-lg border border-line bg-panel2 px-3 py-1.5 text-ink hover:border-accent disabled:opacity-40">+ Add fund</button>
        <span data-testid="weight-total" className={`font-mono ${Math.abs(tot - 100) < 0.01 ? "text-[#6ee06e]" : "text-warn"}`}>Total {tot}%</span>
        {Math.abs(tot - 100) > 0.01 && tot > 0 && <button type="button" onClick={() => setRows(normalise(rows))} className="text-accent hover:underline">Scale to 100%</button>}
        <button type="button" onClick={() => setPf(formatSpec(rows))} disabled={issues.length > 0 || !dirty}
          className="rounded-lg bg-accent px-4 py-1.5 font-semibold text-[#0b0f14] hover:bg-[#a59cf0] disabled:cursor-not-allowed disabled:opacity-40">Analyse this portfolio</button>
        {pf && <button type="button" onClick={() => setPf(null)} className="text-muted hover:text-ink">Clear</button>}
      </div>
      {issues.length > 0 && rows.length > 0 && <ul className={`mt-2 ${simple ? "text-[14px]" : "text-[12px]"} text-warn`} data-testid="editor-problems">{issues.map((x) => <li key={x}>• {x}</li>)}</ul>}
    </section>
  );
}

function Tile({ label, value, sub, testid, chip }: { label: string; value: string; sub?: string; testid?: string; chip?: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-line bg-panel2 p-3" data-testid={testid}>
      <div className="text-[12px] font-medium uppercase tracking-wider text-muted">{label}</div>
      <div className="mt-0.5 flex items-center gap-2"><span className="font-mono text-[22px] text-ink">{value}</span>{chip}</div>
      {sub && <div className="text-[12.5px] text-muted">{sub}</div>}
    </div>
  );
}

function Metrics({ a, simple }: { a: PortfolioAnalysis; simple: boolean }) {
  const m = a.metrics;
  const ticks = useMemo(() => monthTicks(a.series.dates), [a.series.dates]);
  const data = a.series.dates.map((d, i) => ({ d, v: (a.series.value[i] - 1) * 100, dd: a.series.drawdown[i] * 100 }));
  return (
    <section className={`rounded-2xl border border-line bg-panel ${simple ? "p-6" : "p-4"}`} data-testid="portfolio-metrics">
      <h2 className={`${simple ? "text-[22px]" : "text-[16px]"} font-semibold text-ink`}>How this portfolio behaved up to {longDate(a.date)}</h2>
      <p className={`${simple ? "text-[15px]" : "text-[12px]"} text-muted`}>{a.spec.replace(/,/g, " · ").replace(/:/g, " ")}% · daily rebalanced · data through {a.data_through}</p>
      <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Tile testid="m-vol" label="Volatility (3 months)" value={`${m.vol_63d.toFixed(1)}%`} chip={<ToneChip tone={tone(m.vol_63d_pct)} />}
          sub={`yearly-rate swings; ${toneWord(m.vol_63d_pct)} for this portfolio's past 5 years`} />
        <Tile testid="m-ret" label="Last month" value={`${signed(m.return_21d, 1)}%`} sub={`past year ${signed(m.return_252d, 1)}%`} />
        <Tile testid="m-var" label="Bad-day loss (VaR 95%)" value={`${m.var95_hist.toFixed(2)}%`} sub={`on 1 day in 20 over the past year it lost at least this much${simple ? "" : `; CVaR ${m.cvar95_hist.toFixed(2)}%, normal-model VaR ${m.var95_param.toFixed(2)}%`}`} />
        <Tile testid="m-dd" label="Worst fall (1 year)" value={`${m.max_drawdown_252d.toFixed(1)}%`} sub="largest peak-to-trough drop in the past year" />
      </div>
      {!simple && m.avg_comovement_63d !== null && (
        <p className="mt-2 text-[12.5px] text-muted">Average pairwise co-movement of the holdings (Pearson, 63 sessions): {m.avg_comovement_63d.toFixed(2)} now vs {m.avg_comovement_prev_63d?.toFixed(2)} in the previous 63 sessions.</p>
      )}
      <div className="mt-4 h-[220px]" role="img" aria-label="Portfolio value and drawdown over the past year">
        <ResponsiveContainer>
          <AreaChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="d" ticks={ticks} tickFormatter={(d) => new Date(`${d}T12:00:00Z`).toLocaleDateString("en-US", { month: "short", year: "2-digit", timeZone: "UTC" })} stroke={C.axis} fontSize={12} />
            <YAxis stroke={C.axis} fontSize={12} tickFormatter={(v) => `${v.toFixed(0)}%`} width={44} />
            <Tooltip contentStyle={{ background: "#11161d", border: "1px solid #2a3340", fontSize: 12 }} labelFormatter={(d) => longDate(String(d))}
              formatter={(v: number, k) => [`${v.toFixed(2)}%`, k === "v" ? "growth since a year earlier" : "below previous peak"]} />
            <Area type="monotone" dataKey="v" stroke="#9085e9" fill="#9085e9" fillOpacity={0.12} strokeWidth={2} isAnimationActive={false} />
            <Area type="monotone" dataKey="dd" stroke="#e66767" fill="#e66767" fillOpacity={0.15} strokeWidth={1.5} isAnimationActive={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-1 flex gap-5 text-[13px] text-muted"><span className="flex items-center gap-2"><span className="h-1 w-6 rounded bg-[#9085e9]" />growth over the past year</span><span className="flex items-center gap-2"><span className="h-1 w-6 rounded bg-[#e66767]" />distance below previous peak</span></div>
    </section>
  );
}

function Contributions({ a, simple }: { a: PortfolioAnalysis; simple: boolean }) {
  const max = Math.max(100, ...a.holdings.map((h) => Math.abs(h.risk_share)), ...a.holdings.map((h) => h.weight));
  const bar = (v: number, color: string) => (
    <div className="relative h-2.5 rounded bg-panel2"><div className="absolute top-0 h-2.5 rounded" style={{ width: `${(Math.abs(v) / max) * 100}%`, background: color, opacity: v < 0 ? 0.5 : 1 }} /></div>
  );
  return (
    <section className={`rounded-2xl border border-line bg-panel ${simple ? "p-6" : "p-4"}`} data-testid="risk-contributions">
      <h2 className={`${simple ? "text-[20px]" : "text-[15px]"} font-semibold text-ink`}>Which holdings carry the risk?</h2>
      <ul className={`mt-2 space-y-1 ${simple ? "text-[16px]" : "text-[13px]"}`} data-testid="risk-sentences">{riskSentences(a).map((s) => <li key={s}>{s}</li>)}</ul>
      <table className={`mt-3 w-full ${simple ? "text-[15px]" : "text-[12.5px]"}`}>
        <thead><tr className="text-left text-[11.5px] uppercase tracking-wider text-muted"><th className="font-medium">Fund</th><th className="font-medium">Money vs share of risk</th><th className="text-right font-medium">Weight</th><th className="text-right font-medium">Risk</th></tr></thead>
        <tbody>{a.holdings.map((h) => (
          <tr key={h.ticker} className="border-t border-line/60">
            <td className="py-1.5 pr-2"><b className="text-ink">{h.ticker}</b> <span className="text-muted">{simple ? "" : h.name}</span></td>
            <td className="w-1/2 py-1.5 pr-2"><div className="space-y-1">{bar(h.weight, "#5d6673")}{bar(h.risk_share, h.risk_share < 0 ? "#3987e5" : "#d95926")}</div></td>
            <td className="py-1.5 text-right font-mono">{h.weight.toFixed(0)}%</td><td className="py-1.5 text-right font-mono">{h.risk_share.toFixed(1)}%</td>
          </tr>))}
        </tbody>
      </table>
      <p className="mt-2 text-[12.5px] text-muted">Grey = share of the money; orange = share of the portfolio's 3-month risk (Euler contribution, sums to 100%); blue = negative share, i.e. the holding offset the others.</p>
    </section>
  );
}

function NetworkLinks({ a, simple }: { a: PortfolioAnalysis; simple: boolean }) {
  const { go } = useRoute();
  const n = a.network;
  const proxies = a.holdings.filter((h) => h.node_is_proxy);
  return (
    <section className={`rounded-2xl border border-line bg-panel ${simple ? "p-6" : "p-4"}`} data-testid="portfolio-network">
      <h2 className={`${simple ? "text-[20px]" : "text-[15px]"} font-semibold text-ink`}>How your holdings sit in the changing relationship network</h2>
      <p className={`mt-1 ${simple ? "text-[16px]" : "text-[13px]"}`} data-testid="network-exposure">
        {n.n_links === 0 ? "None of the estimated links touching your holdings changed materially this week."
          : `${n.n_links} estimated link${n.n_links === 1 ? "" : "s"} touching your holdings changed versus the previous week; they involve funds holding ${n.weight_on_changed_nodes}% of the money.`}</p>
      <ul className={`mt-2 space-y-1.5 ${simple ? "text-[15px]" : "text-[12.5px]"}`} data-testid="portfolio-links">
        {n.links.slice(0, simple ? 5 : 12).map((e) => (
          <li key={e.id} className="flex flex-wrap items-baseline gap-x-2"><span className="font-medium text-ink">{short(e.source)} → {short(e.target)}</span>
            <span className="text-muted">{e.change}{simple ? "" : ` (${signed(e.before)} → ${signed(e.after)})`}</span>
            <span className="text-[12px] text-dim">your {e.holdings.join(", ")}</span></li>
        ))}
      </ul>
      <p className="mt-2 text-[12.5px] text-muted">
        Links are estimated directed dependence (VarLiNGAM, window ending {n.window_end} vs {n.previous_window_end}), not causal effects, and single changes are noisy.
        {proxies.length > 0 && ` Funds that are not network nodes are placed on the closest node: ${proxies.map((h) => `${h.ticker}→${h.node}`).join(", ")}.`}
        {n.not_in_graph.length > 0 && ` Not represented in the network: ${n.not_in_graph.join(", ")}.`}</p>
      <button type="button" onClick={() => go(simple ? "relations" : "network")} className="mt-2 text-[14px] font-medium text-accent hover:underline">Open the relationship network →</button>
    </section>
  );
}
