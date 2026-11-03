import { useMemo } from "react";
import {
  Area, AreaChart, CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Scatter,
  Tooltip, XAxis, YAxis,
} from "recharts";
import type { Series } from "../api";
import { fmt, longDate, ordinal } from "../format";

export const C = { sii: "#9085e9", vix: "#d95926", nfci: "#199e70", grid: "#1f2733", axis: "#5d6673", ink: "#e6e8eb",
                   muted: "#8a93a0", stress: "rgba(208,59,59,0.13)", hindsight: "rgba(250,178,25,0.08)" };

export type Range = "6M" | "1Y" | "3Y" | "5Y" | "Max";
const DAYS: Record<Range, number> = { "6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260, Max: 1e9 };

interface Row {
  date: string; sii_pct: number | null; vix_pct: number | null; nfci_pct: number | null; sii: number | null;
  vix: number | null; nfci: number | null; score: number | null; thr: number | null; p: number | null;
  alert: number | null; future: boolean; retro: number | null;
}

/** Rows for [date - range, date] (point-in-time); with `hindsight`, also the 63 sessions after the replay date. */
export function window(s: Series, date: string, range: Range, hindsight = false): Row[] {
  let end = s.dates.findIndex((d) => d > date);
  end = end === -1 ? s.dates.length - 1 : end - 1;
  const start = Math.max(0, end - DAYS[range] + 1);
  const last = hindsight ? Math.min(s.dates.length - 1, end + 63) : end;
  const out: Row[] = [];
  for (let i = start; i <= last; i++) {
    out.push({ date: s.dates[i], sii_pct: s.sii_pct[i], vix_pct: s.vix_pct[i], nfci_pct: s.nfci_pct[i], sii: s.sii[i],
               vix: s.vix[i], nfci: s.nfci[i], score: s.model_score[i], thr: s.model_threshold[i],
               p: s.regime_p_stress[i], alert: s.model_alert[i] ? 103 : null, future: i > end,
               retro: s.retrospective_stress[i] });
  }
  return out;
}

/** Contiguous runs where `pred` holds, as [startDate, endDate] pairs. */
export function runs(rows: Row[], pred: (r: Row) => boolean): [string, string][] {
  const out: [string, string][] = [];
  let s: string | null = null, prev = "";
  for (const r of rows) {
    if (pred(r)) { if (!s) s = r.date; prev = r.date; } else if (s) { out.push([s, prev]); s = null; }
  }
  if (s) out.push([s, prev]);
  return out;
}

/** Month-start ticks (first trading day of each month), thinned to at most ~12 labels. */
export function monthTicks(dates: string[]): string[] {
  const firsts = dates.filter((d, i) => i === 0 || d.slice(0, 7) !== dates[i - 1].slice(0, 7)).slice(1);
  const step = [1, 2, 3, 6, 12, 24].find((k) => firsts.length / k <= 12) ?? 24;
  return firsts.filter((d) => (Number(d.slice(5, 7)) - 1) % step === 0);
}

const tick = (d: string) => new Date(`${d}T12:00:00Z`).toLocaleDateString("en-US", { month: "short", year: "2-digit", timeZone: "UTC" });

function StressTooltip({ active, payload }: { active?: boolean; payload?: { payload: Row }[] }) {
  if (!active || !payload?.length) return null;
  const r = payload[0].payload;
  return (
    <div className="rounded-lg border border-line bg-panel2/95 p-3 text-xs shadow-xl">
      <div className="mb-1.5 font-medium text-ink">{longDate(r.date)}{r.future && <span className="ml-2 text-warn">hindsight (after replay date)</span>}</div>
      <TipRow color={C.sii} label="SII" value={`${fmt(r.sii, 2)} · ${ordinal(r.sii_pct)} pct`} />
      <TipRow color={C.vix} label="VIX" value={`${fmt(r.vix, 1)} · ${ordinal(r.vix_pct)} pct`} />
      <TipRow color={C.nfci} label="NFCI" value={`${fmt(r.nfci, 2)} · ${ordinal(r.nfci_pct)} pct`} dashed />
      <div className="mt-1.5 border-t border-line pt-1.5 text-muted">
        Model score {fmt(r.score, 3)} / threshold {fmt(r.thr, 3)}{r.alert ? <span className="ml-1 text-[#ff8a8a]">▲ alert</span> : null}
        <br />Stressed-regime probability (as known then): {r.p === null ? "—" : `${Math.round(r.p * 100)}%`}
      </div>
    </div>
  );
}

function TipRow({ color, label, value, dashed = false }: { color: string; label: string; value: string; dashed?: boolean }) {
  return (
    <div className="flex items-center justify-between gap-6 py-0.5">
      <span className="flex items-center gap-1.5 text-muted">
        <svg width="14" height="4" aria-hidden><line x1="0" y1="2" x2="14" y2="2" stroke={color} strokeWidth="2" strokeDasharray={dashed ? "3 2" : undefined} /></svg>
        {label}
      </span>
      <span className="font-mono text-ink">{value}</span>
    </div>
  );
}

export function StressChart({ series, date, range, hindsight = false, height = 300, onPick }: {
  series: Series; date: string; range: Range; hindsight?: boolean; height?: number; onPick?: (d: string) => void;
}) {
  const rows = useMemo(() => window(series, date, range, hindsight), [series, date, range, hindsight]);
  const regime = useMemo(() => runs(rows, (r) => !r.future && (r.p ?? 0) > 0.5), [rows]);
  const retro = useMemo(() => (hindsight ? runs(rows, (r) => r.retro === 1) : []), [rows, hindsight]);
  if (!rows.length) return null;
  const replayEnd = rows.filter((r) => !r.future).at(-1)?.date;
  return (
    <div style={{ height }} data-testid="stress-chart">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 12, right: 12, bottom: 0, left: -12 }}
          onClick={(e) => { if (onPick && e?.activeLabel) onPick(String(e.activeLabel)); }}>
          <CartesianGrid stroke={C.grid} vertical={false} />
          {regime.map(([a, b]) => <ReferenceArea key={`r${a}`} x1={a} x2={b} y1={0} y2={106} fill={C.stress} ifOverflow="hidden" />)}
          {retro.map(([a, b]) => <ReferenceArea key={`h${a}`} x1={a} x2={b} y1={0} y2={106} fill={C.hindsight} stroke="rgba(250,178,25,0.35)" strokeDasharray="3 3" ifOverflow="hidden" />)}
          <XAxis dataKey="date" ticks={monthTicks(rows.map((r) => r.date))} tickFormatter={tick} interval={0} stroke={C.axis} tick={{ fill: C.muted, fontSize: 11 }} tickLine={false} />
          <YAxis domain={[0, 106]} ticks={[0, 20, 50, 80, 100]} stroke={C.axis} tick={{ fill: C.muted, fontSize: 11 }} tickLine={false} axisLine={false} />
          <ReferenceLine y={80} stroke={C.axis} strokeDasharray="2 4" />
          <Tooltip content={<StressTooltip />} cursor={{ stroke: C.muted, strokeWidth: 1 }} />
          <Line type="monotone" dataKey="nfci_pct" stroke={C.nfci} strokeWidth={1.8} strokeDasharray="5 3" dot={false} isAnimationActive={false} connectNulls />
          <Line type="monotone" dataKey="vix_pct" stroke={C.vix} strokeWidth={1.8} dot={false} isAnimationActive={false} connectNulls />
          <Line type="monotone" dataKey="sii_pct" stroke={C.sii} strokeWidth={2.4} dot={false} isAnimationActive={false} connectNulls />
          <Scatter dataKey="alert" isAnimationActive={false}
            shape={(p: { cx?: number; cy?: number }) => <path d={`M${p.cx},${(p.cy ?? 0) - 4} l3.5,6 h-7 z`} fill="#ff6b6b" />} />
          {replayEnd && <ReferenceLine x={replayEnd} stroke={C.ink} strokeWidth={1.2} label={{ value: "replay date", position: "insideTopRight", fill: C.muted, fontSize: 10 }} />}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export function Legend({ hindsight = false }: { hindsight?: boolean }) {
  const item = (el: JSX.Element, text: string) => <span className="flex items-center gap-1.5">{el}{text}</span>;
  const line = (c: string, dash?: string) => <svg width="18" height="6" aria-hidden><line x1="0" y1="3" x2="18" y2="3" stroke={c} strokeWidth="2.4" strokeDasharray={dash} /></svg>;
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted">
      {item(line(C.sii), "SII (structural instability)")}
      {item(line(C.vix), "VIX")}
      {item(line(C.nfci, "4 3"), "Chicago Fed NFCI")}
      {item(<span className="h-3 w-4 rounded-sm" style={{ background: "rgba(208,59,59,0.35)" }} />, "Stressed regime (as estimated then)")}
      {item(<span className="text-[#ff6b6b]">▲</span>, "Model alert")}
      {hindsight && item(<span className="h-3 w-4 rounded-sm border border-dashed border-warn/60" />, "Stress period known only in hindsight")}
    </div>
  );
}

export function Spark({ values, color = C.sii, height = 40 }: { values: (number | null)[]; color?: string; height?: number }) {
  const data = values.map((v, i) => ({ i, v }));
  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 4, right: 0, bottom: 0, left: 0 }}>
          <defs><linearGradient id="spk" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor={color} stopOpacity={0.35} /><stop offset="1" stopColor={color} stopOpacity={0} /></linearGradient></defs>
          <YAxis hide domain={["dataMin", "dataMax"]} />
          <Area type="monotone" dataKey="v" stroke={color} strokeWidth={1.6} fill="url(#spk)" isAnimationActive={false} connectNulls />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Small-multiple raw-value panel (one measure, one unit, one axis). */
export function RawPanel({ series, date, range, field, color, label, unit, thresholdField, hindsight = false }: {
  series: Series; date: string; range: Range; field: "sii" | "vix" | "nfci" | "score"; color: string; label: string;
  unit: string; thresholdField?: boolean; hindsight?: boolean;
}) {
  const rows = useMemo(() => window(series, date, range, hindsight), [series, date, range, hindsight]);
  const replayEnd = rows.filter((r) => !r.future).at(-1)?.date;
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-[11px] text-muted"><span className="text-ink">{label}</span><span>{unit}</span></div>
      <div style={{ height: 130 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={rows} margin={{ top: 6, right: 12, bottom: 0, left: -8 }}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="date" ticks={monthTicks(rows.map((r) => r.date)).filter((_, i, a) => a.length <= 6 || i % 2 === 0)} tickFormatter={tick} interval={0} stroke={C.axis} tick={{ fill: C.muted, fontSize: 10 }} tickLine={false} />
            <YAxis stroke={C.axis} tick={{ fill: C.muted, fontSize: 10 }} tickLine={false} axisLine={false} width={44} domain={["auto", "auto"]} />
            <Tooltip contentStyle={{ background: "#161c25", border: "1px solid #1f2733", fontSize: 12 }} labelStyle={{ color: C.ink }}
              formatter={(v: number) => fmt(v, field === "score" ? 3 : 2)} labelFormatter={(d: string) => longDate(d)} />
            <Line type="monotone" dataKey={field} name={label} stroke={color} strokeWidth={1.8} dot={false} isAnimationActive={false} connectNulls />
            {thresholdField && <Line type="stepAfter" dataKey="thr" name="threshold" stroke={C.muted} strokeDasharray="4 3" strokeWidth={1} dot={false} isAnimationActive={false} />}
            {replayEnd && <ReferenceLine x={replayEnd} stroke={C.ink} strokeWidth={1} />}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
