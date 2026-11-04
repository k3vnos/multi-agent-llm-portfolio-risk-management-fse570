import { useEffect } from "react";
import type { Mode, Page } from "../state";
import { useRoute } from "../state";

export interface Step { page: Page; date: string; title: string; body: string }

// 3–5 minute walkthrough. Every step opens a real page on a real saved date.
export const STEPS: Step[] = [
  { page: "monitor", date: "2019-12-31", title: "1 · The monitor",
    body: "Historical replay of the saved research outputs. KPI cards show the Structural Instability Index (SII), the market regime as it was estimated at the time, the risk status (a fixed, documented rule) and how many estimated links changed this week. Hover any ? for a plain-English explanation." },
  { page: "monitor", date: "2020-02-24", title: "2 · A real stress episode: COVID-19",
    body: "We jump to 24 Feb 2020, the first trading day after the COVID-19 stress onset (21 Feb). The status turns to a model alert, the regime estimate flips to stressed and VIX jumps. Use ▶ Play in the replay bar to step forward week by week." },
  { page: "timeline", date: "2020-02-24", title: "3 · Indicators and alerts over time",
    body: "SII, VIX and NFCI as percentiles of their own previous 5 years, plus raw small multiples and the model's score vs threshold. Note the SII did not rise ahead of VIX here. 'Show what happened next' reveals hindsight, clearly marked." },
  { page: "network", date: "2020-02-24", title: "4 · How relationships changed",
    body: "Estimated directed dependence among 10 asset classes and 3 macro factors (VarLiNGAM on the previous 500 trading days). Switch Changes / Current links, compare vs 1 week or 1 month, click an asset or arrow to inspect it. These are statistical estimates, not causal proof." },
  { page: "agents", date: "2020-01-10", title: "5 · Multi-agent evidence",
    body: "Graph, Macro and News agents each see only evidence published before the 4pm close; the synthesis agent's deterministic briefing cites every source. The LLM draft for this alert is shown as an UNVERIFIED draft with its hand-check flags — the held-out check found 73% faithful claims, below our 90% bar." },
  { page: "research", date: "2020-01-10", title: "6 · What we found",
    body: "The locked evaluation (2014–2026, 21 stress onsets) did not support the hypothesis: instability-only models were significantly worse than VIX, and adding them to standard controls did not help. Leakage audit and bit-identical reproducibility are shown at the bottom." },
];

// Simple Mode walkthrough: written for someone with no background in finance, statistics or machine learning.
export const SIMPLE_STEPS: Step[] = [
  { page: "home", date: "2019-12-31", title: "1 · What this app is",
    body: "We replay real market data from the past, one day at a time, as if we were watching on that day. Nothing here is live. The screen answers three questions: what is happening, why our warning model did or did not raise an alert, and what evidence supports that." },
  { page: "home", date: "2020-02-24", title: "2 · A real crisis: COVID-19 (Feb 2020)",
    body: "Looking back, our stress labels mark 21 February 2020 as the start of a stress period. Here, at the close of Monday 24 February, the warning model's score is above its alert line, so it raises an alert — and the market mood is judged 'stressed'. Press 'Play forward' to watch the following weeks." },
  { page: "home", date: "2020-02-24", title: "3 · The idea we tested",
    body: "Our idea: when the way investments influence each other starts to rearrange, trouble may be coming. The chart compares our 'relationship-change index' (purple) with the market's standard fear gauge, the VIX (orange). If our idea worked, purple would usually rise first." },
  { page: "relations", date: "2020-02-24", title: "4 · Relationships between investments",
    body: "Each circle is a type of investment; arrows show which ones help explain the others' moves, estimated from two years of past data. By default we show only the 5 biggest changes this week. These are statistical estimates, not proof of cause and effect." },
  { page: "explain", date: "2020-01-10", title: "5 · The AI team",
    body: "Three agents each read one kind of evidence available before the market closed — relationships, economic indicators and news headlines — and a fourth combines them. Their summary is built by fixed rules so every sentence can be checked. A language-model draft is available but marked unverified." },
  { page: "findings", date: "2020-01-10", title: "6 · What we found",
    body: "Tested honestly on 21 stress episodes (2014–2026), our indicator did not give earlier or better warnings than the standard fear gauge, and adding it to standard data did not help. That negative result is the main finding. Switch to Advanced (top) for every technical detail." },
];

export function DemoTour({ mode = "advanced", step, setStep, onClose }: { mode?: Mode; step: number; setStep: (n: number) => void; onClose: () => void }) {
  const { go } = useRoute();
  const steps = mode === "simple" ? SIMPLE_STEPS : STEPS;
  const s = steps[step];
  useEffect(() => { go(s.page, { date: s.date, alert: null }); }, [step]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <aside role="dialog" aria-label="Guided demo" data-testid="tour"
      className={`fixed bottom-5 right-5 z-50 rounded-xl border border-accent/60 bg-panel2/95 p-4 shadow-2xl backdrop-blur ${mode === "simple" ? "w-[460px]" : "w-[380px]"}`}>
      <div className="flex items-center justify-between text-[10.5px] uppercase tracking-[0.14em] text-accent">
        <span>Guided demo · step {step + 1} of {steps.length}</span>
        <button type="button" onClick={onClose} className="text-muted hover:text-ink" aria-label="Close demo">✕</button>
      </div>
      <h3 className={`mt-1.5 font-semibold text-ink ${mode === "simple" ? "text-[19px]" : "text-[15px]"}`}>{s.title}</h3>
      <p className={`mt-1.5 leading-relaxed ${mode === "simple" ? "text-[16px] text-ink/90" : "text-[12.5px] text-muted"}`}>{s.body}</p>
      <div className="mt-3 flex items-center justify-between">
        <div className="flex gap-1">{steps.map((_, i) => <span key={i} className={`h-1.5 w-5 rounded-full ${i <= step ? "bg-accent" : "bg-line"}`} />)}</div>
        <div className="flex gap-2">
          <button type="button" disabled={step === 0} onClick={() => setStep(step - 1)} className="rounded-md border border-line px-3 py-1 text-xs text-muted disabled:opacity-40">Back</button>
          {step < steps.length - 1
            ? <button type="button" onClick={() => setStep(step + 1)} className="rounded-md bg-accent px-3 py-1 text-xs font-medium text-[#0b0f14]">Next</button>
            : <button type="button" onClick={onClose} className="rounded-md bg-accent px-3 py-1 text-xs font-medium text-[#0b0f14]">Finish</button>}
        </div>
      </div>
    </aside>
  );
}
