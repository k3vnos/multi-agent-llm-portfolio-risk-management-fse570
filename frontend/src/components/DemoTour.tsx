import { useEffect } from "react";
import type { Page } from "../state";
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

export function DemoTour({ step, setStep, onClose }: { step: number; setStep: (n: number) => void; onClose: () => void }) {
  const { go } = useRoute();
  const s = STEPS[step];
  useEffect(() => { go(s.page, { date: s.date, alert: null }); }, [step]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <aside role="dialog" aria-label="Guided demo" data-testid="tour"
      className="fixed bottom-5 right-5 z-50 w-[380px] rounded-xl border border-accent/60 bg-panel2/95 p-4 shadow-2xl backdrop-blur">
      <div className="flex items-center justify-between text-[10.5px] uppercase tracking-[0.14em] text-accent">
        <span>Guided demo · step {step + 1} of {STEPS.length}</span>
        <button type="button" onClick={onClose} className="text-muted hover:text-ink" aria-label="Close demo">✕</button>
      </div>
      <h3 className="mt-1.5 text-[15px] font-semibold text-ink">{s.title}</h3>
      <p className="mt-1.5 text-[12.5px] leading-relaxed text-muted">{s.body}</p>
      <div className="mt-3 flex items-center justify-between">
        <div className="flex gap-1">{STEPS.map((_, i) => <span key={i} className={`h-1.5 w-5 rounded-full ${i <= step ? "bg-accent" : "bg-line"}`} />)}</div>
        <div className="flex gap-2">
          <button type="button" disabled={step === 0} onClick={() => setStep(step - 1)} className="rounded-md border border-line px-3 py-1 text-xs text-muted disabled:opacity-40">Back</button>
          {step < STEPS.length - 1
            ? <button type="button" onClick={() => setStep(step + 1)} className="rounded-md bg-accent px-3 py-1 text-xs font-medium text-[#0b0f14]">Next</button>
            : <button type="button" onClick={onClose} className="rounded-md bg-accent px-3 py-1 text-xs font-medium text-[#0b0f14]">Finish</button>}
        </div>
      </div>
    </aside>
  );
}
