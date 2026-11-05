import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import type { Meta, Series } from "../api";
import { longDate } from "../format";
import { G } from "../glossary";
import { ADVANCED_PAGES, COUNTERPART, modeOf, SIMPLE_PAGES, type Page, useRoute } from "../state";
import { Button, Info } from "./ui";

interface DataCtx { meta: Meta; series: Series }
const Data = createContext<DataCtx | null>(null);
export const DataProvider = ({ value, children }: { value: DataCtx; children: ReactNode }) => <Data.Provider value={value}>{children}</Data.Provider>;
export function useData(): DataCtx {
  const d = useContext(Data);
  if (!d) throw new Error("useData outside DataProvider");
  return d;
}

const TITLES: Record<Page, string> = {
  home: "Briefing", relations: "Relationships", portfolio: "My portfolio", explain: "Ask the AI agents", findings: "What we found",
  monitor: "Monitor", timeline: "Timeline", network: "Relationship network", alerts: "Alerts", lens: "Portfolio Lens", agents: "AI agents",
  research: "Research & validation",
};

function ModeToggle() {
  const { page, go } = useRoute();
  const mode = modeOf(page);
  const btn = (m: "simple" | "advanced", label: string) => (
    <button type="button" aria-pressed={mode === m} onClick={() => mode !== m && go(COUNTERPART[page])}
      className={`rounded-md px-3 py-1 text-[13px] font-medium transition-colors ${mode === m ? "bg-ink text-bg" : "text-muted hover:text-ink"}`}>
      {label}
    </button>
  );
  return (
    <div className="flex items-center gap-0.5 rounded-lg border border-line bg-panel2 p-0.5" role="group" aria-label="Display mode">
      {btn("simple", "Simple")}{btn("advanced", "Advanced")}
    </div>
  );
}

export function Logo() {
  return (
    <svg width="26" height="26" viewBox="0 0 26 26" aria-hidden>
      <line x1="6" y1="7" x2="19" y2="9" stroke="#3a4352" strokeWidth="1.4" /><line x1="6" y1="7" x2="11" y2="19" stroke="#3a4352" strokeWidth="1.4" />
      <line x1="19" y1="9" x2="11" y2="19" stroke="#9085e9" strokeWidth="1.6" />
      <circle cx="6" cy="7" r="3" fill="#9085e9" /><circle cx="19" cy="9" r="3" fill="#d95926" /><circle cx="11" cy="19" r="3" fill="#199e70" />
    </svg>
  );
}

export function Header({ onTour }: { onTour: () => void }) {
  const { page, go, date } = useRoute();
  return (
    <header className="sticky top-0 z-40 border-b border-line bg-bg/95 backdrop-blur">
      <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-x-6 gap-y-2 px-5 py-2.5">
        <button type="button" onClick={() => go(modeOf(page) === "simple" ? "home" : "monitor")} className="flex items-center gap-2.5 text-left">
          <Logo />
          <div>
            <div className="text-[15px] font-semibold leading-tight text-ink">Structural Instability Monitor</div>
            <div className="text-[11px] text-muted">Directed-dependence risk research · FSE 570 capstone</div>
          </div>
        </button>
        <span className="inline-flex items-center gap-1.5 rounded-md border border-warn/50 bg-warn/10 px-2 py-0.5 text-[11px] font-medium tracking-wide text-warn">
          HISTORICAL REPLAY <Info text={G.replay} />
        </span>
        <ModeToggle />
        <nav className="flex flex-wrap gap-1" aria-label="Main">
          {(modeOf(page) === "simple" ? SIMPLE_PAGES : ADVANCED_PAGES).map((p) => (
            <button key={p} type="button" onClick={() => go(p)} aria-current={page === p ? "page" : undefined}
              className={`rounded-md px-3 py-1.5 transition-colors ${modeOf(page) === "simple" ? "text-[15px]" : "text-[13px]"} ${page === p ? "bg-accent/20 text-ink" : "text-muted hover:bg-panel2 hover:text-ink"}`}>
              {TITLES[p]}
            </button>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-3">
          <span className="text-xs text-muted">Viewing <span className="font-mono text-ink">{longDate(date)}</span></span>
          <button type="button" onClick={onTour} className="rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-[#0b0f14] hover:bg-[#a59cf0]">
            ▶ Guided demo
          </button>
        </div>
      </div>
    </header>
  );
}

/** Historical replay: date picker, step, play/pause and episode shortcuts. Every control moves the shared date. */
export function ReplayBar() {
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
    }, 650);
    return () => { if (timer.current) window.clearInterval(timer.current); };
  }, [playing, date, series.dates, setDate]);

  const step = (n: number) => setDate(series.dates[Math.min(series.dates.length - 1, Math.max(0, idx + n))]);
  return (
    <div className="rounded-xl border border-line bg-panel px-4 py-3" data-testid="replay-bar">
      <div className="flex flex-wrap items-center gap-2">
        <span className="mr-1 text-[10.5px] font-medium uppercase tracking-[0.12em] text-muted">Replay date</span>
        <input type="date" aria-label="Replay date" value={series.dates[idx]} min={meta.start} max={meta.end}
          onChange={(e) => e.target.value && setDate(e.target.value)}
          className="rounded-md border border-line bg-panel2 px-2 py-1 font-mono text-xs text-ink [color-scheme:dark]" />
        <Button onClick={() => step(-5)} title="Back one week">◀ 1w</Button>
        <Button onClick={() => setPlaying((p) => !p)} active={playing} title="Advance one week every 0.65 s">{playing ? "❚❚ Pause" : "▶ Play"}</Button>
        <Button onClick={() => step(5)} title="Forward one week">1w ▶</Button>
        <span className="mx-2 hidden h-5 w-px bg-line md:block" />
        <span className="mr-1 text-[10.5px] font-medium uppercase tracking-[0.12em] text-muted">Replay an episode</span>
        {meta.events.filter((e) => e.available && (e.label.includes("COVID") || e.target === "exploratory")).map((e) => (
          <Button key={e.id} onClick={() => { setPlaying(false); setDate(e.replay_start!); setPlaying(true); }} title={e.note || e.period}>
            {e.label.replace(" (exploratory)", "")} · {e.period}
          </Button>
        ))}
        <EpisodePicker />
        {meta.events.filter((e) => !e.available).map((e) => (
          <Button key={e.id} disabled title={e.note}>{e.period} crisis — not in test span</Button>
        ))}
      </div>
    </div>
  );
}

function EpisodePicker() {
  const { meta } = useData();
  const { setDate } = useRoute();
  const onsets = meta.events.filter((e) => e.available && e.target === "primary" && !e.label.includes("COVID"));
  return (
    <select aria-label="Other stress episodes" defaultValue="" onChange={(e) => { const ev = onsets.find((o) => o.id === e.target.value); if (ev) setDate(ev.focus!); }}
      className="rounded-md border border-line bg-panel2 px-2 py-1 text-xs text-muted [color-scheme:dark]">
      <option value="" disabled>Other stress onsets ({onsets.length})…</option>
      {onsets.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
    </select>
  );
}

export function Footer() {
  return (
    <footer className="mx-auto max-w-[1400px] px-5 pb-10 pt-4 text-[11.5px] leading-relaxed text-dim">
      <span className="text-muted">About the data.</span> Historical replay of saved, point-in-time research outputs
      (prices from Yahoo Finance, macro series from FRED/ALFRED as published at the time, news headlines from the
      public FNSPID dataset up to June 2020). Nothing is live and nothing here is a forecast or investment advice. Graph
      arrows are estimated directed statistical dependence, not proven causal effects. In the project's locked
      evaluation the instability measures did not outperform the VIX.
    </footer>
  );
}
