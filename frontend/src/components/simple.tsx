import { useEffect, useRef, useState, type ReactNode } from "react";
import { longDate } from "../format";
import type { Tone } from "../plain";
import { useRoute } from "../state";
import { useData } from "./Shell";
import { Info } from "./ui";

/** Large-type card for Simple Mode (projector-readable). */
export function SCard({ title, kicker, info, children, className = "", right }: {
  title: ReactNode; kicker?: string; info?: string; children: ReactNode; className?: string; right?: ReactNode;
}) {
  return (
    <section className={`rounded-2xl border border-line bg-panel p-6 ${className}`}>
      {kicker && <div className="mb-1 text-[13px] font-semibold uppercase tracking-[0.14em] text-accent">{kicker}</div>}
      <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
        <h2 className="flex items-center gap-2 text-[22px] font-semibold leading-snug text-ink">{title}{info && <Info text={info} />}</h2>
        {right}
      </div>
      <div className="text-[16px] leading-relaxed text-ink/90">{children}</div>
    </section>
  );
}

export function ToneChip({ tone }: { tone: Tone }) {
  const t = {
    high: ["▲ unusually high", "border-warn/60 bg-warn/10 text-warn"],
    low: ["▼ unusually low", "border-neg/60 bg-neg/10 text-[#8fbaf2]"],
    normal: ["● normal", "border-good/60 bg-good/10 text-[#6ee06e]"],
    unknown: ["– n/a", "border-line text-muted"],
  }[tone];
  return <span className={`inline-flex shrink-0 items-center rounded-md border px-2 py-0.5 text-[13px] font-medium ${t[1]}`}>{t[0]}</span>;
}

/** Expandable "details" block: technical terms and full evidence live here, not on the surface. */
export function Details({ label, children, open: initial = false, testid }: { label: string; children: ReactNode; open?: boolean; testid?: string }) {
  const [open, setOpen] = useState(initial);
  return (
    <div className="mt-4 rounded-xl border border-line bg-panel2" data-testid={testid}>
      <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open}
        className="flex w-full items-center gap-2 px-4 py-3 text-left text-[15px] font-medium text-ink hover:text-accent">
        <span className="text-muted">{open ? "▾" : "▸"}</span>{label}
      </button>
      {open && <div className="border-t border-line px-4 py-4 text-[15px]">{children}</div>}
    </div>
  );
}

/** Horizontal gauge: where today's score sits relative to the alert line (both on a 0–1 scale). */
export function ScoreGauge({ score, threshold, alert }: { score: number; threshold: number; alert: boolean }) {
  const pct = (v: number) => `${Math.max(0, Math.min(1, v)) * 100}%`;
  return (
    <div className="mt-2" data-testid="score-gauge">
      <div className="relative mb-2 h-6 text-[13px]">
        <span className="absolute -translate-x-1/2 whitespace-nowrap font-medium text-ink" style={{ left: pct(threshold) }}>▼ alert line {threshold.toFixed(2)}</span>
      </div>
      <div className="relative h-4 rounded-full bg-gradient-to-r from-[#1c3a2b] via-[#3a3420] to-[#4a1f1f]">
        <div className="absolute inset-y-[-6px] w-[3px] rounded bg-ink" style={{ left: pct(threshold) }} />
        <div className={`absolute top-1/2 h-6 w-6 -translate-x-1/2 -translate-y-1/2 rounded-full border-[3px] border-bg ${alert ? "bg-crit" : "bg-good"}`}
          style={{ left: pct(score) }} />
      </div>
      <div className="mt-2 flex justify-between text-[13px] text-muted">
        <span>0 · low chance</span>
        <span>1 · high chance</span>
      </div>
    </div>
  );
}

/** Simple replay controls: big buttons, plain labels, and an always-visible "historical replay" notice. */
export function SimpleReplay() {
  const { meta, series } = useData();
  const { date, setDate } = useRoute();
  const [playing, setPlaying] = useState(false);
  const idx = Math.max(0, series.dates.findIndex((d) => d >= date));
  const timer = useRef<number | null>(null);
  useEffect(() => {
    if (!playing) return;
    timer.current = window.setInterval(() => {
      const i = series.dates.findIndex((d) => d >= date);
      const next = Math.min(series.dates.length - 1, (i < 0 ? 0 : i) + 5);
      setDate(series.dates[next]);
      if (next >= series.dates.length - 1) setPlaying(false);
    }, 900);
    return () => { if (timer.current) window.clearInterval(timer.current); };
  }, [playing, date, series.dates, setDate]);
  const step = (n: number) => setDate(series.dates[Math.min(series.dates.length - 1, Math.max(0, idx + n))]);
  const big = "rounded-lg border border-line bg-panel2 px-4 py-2 text-[15px] text-ink hover:border-accent";
  const covid = meta.events.find((e) => e.label.includes("COVID"));
  const bank = meta.events.find((e) => e.target === "exploratory");
  return (
    <section className="rounded-2xl border border-warn/40 bg-panel p-5" data-testid="simple-replay">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-[13px] font-semibold uppercase tracking-[0.14em] text-warn">Historical replay · past data, not live</div>
          <div className="mt-1 text-[26px] font-semibold text-ink" data-testid="replay-date">{longDate(series.dates[idx])}</div>
          <div className="text-[15px] text-muted">Everything below is what was known at the market close on this date.</div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button type="button" className={big} onClick={() => step(-5)}>◀ Back a week</button>
          <button type="button" className={`${big} ${playing ? "border-accent bg-accent/20" : ""}`} onClick={() => setPlaying((p) => !p)}>{playing ? "❚❚ Pause" : "▶ Play forward"}</button>
          <button type="button" className={big} onClick={() => step(5)}>Forward a week ▶</button>
          <input type="date" aria-label="Replay date" value={series.dates[idx]} min={meta.start} max={meta.end}
            onChange={(e) => e.target.value && setDate(e.target.value)}
            className="rounded-lg border border-line bg-panel2 px-3 py-2 text-[15px] text-ink [color-scheme:dark]" />
        </div>
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-2 text-[15px]">
        <span className="text-muted">Jump to:</span>
        {covid && <button type="button" className={big} onClick={() => { setPlaying(false); setDate(covid.focus!); }}>COVID-19 crash · Feb 2020</button>}
        {bank && <button type="button" className={big} onClick={() => { setPlaying(false); setDate(bank.focus!); }} title={bank.note}>Regional-bank stress · Mar 2023</button>}
        <button type="button" className={big} onClick={() => { setPlaying(false); setDate(meta.end); }}>Latest data · {longDate(meta.end)}</button>
      </div>
    </section>
  );
}

export function SimpleLoading() {
  return <div className="flex min-h-[200px] items-center justify-center gap-3 text-[16px] text-muted" role="status"><span className="h-5 w-5 animate-spin rounded-full border-2 border-dim border-t-accent" />Loading…</div>;
}
