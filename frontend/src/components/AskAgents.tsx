import { useEffect, useRef, useState } from "react";
import { live, type AgentEvent, type AgentStatus, type Job } from "../api";
import { longDate } from "../format";
import { useAsync } from "../hooks";
import { useRoute } from "../state";
import { AGENT_COLOR, AGENT_NAME, AgentTrace } from "./AgentTrace";

const STEPS = ["graph", "macro", "news", "synthesis"] as const;
const DONE = new Set(["done", "error", "cancelled", "timeout"]);

/** "Ask the agents": runs the four v4 agents on the local Ollama model for the replay date and shows real progress
 * events from the backend. If the model is unavailable, the saved explanation (if any) is shown instead. */
export function AskAgents({ compact = false, portfolio = null }: { compact?: boolean; portfolio?: string | null }) {
  const { date, setDate } = useRoute();
  const [status, setStatus] = useState<AgentStatus | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [withPf, setWithPf] = useState(true);
  const timer = useRef<number | null>(null);
  const saved = useAsync(() => live.saved(date), [date]);

  useEffect(() => { live.status().then(setStatus).catch((e: Error) => setStatus({ available: false, model: "qwen2.5:7b", reason: e.message, busy: false, notice: "", fallback: "" })); }, []);
  useEffect(() => () => { if (timer.current) window.clearTimeout(timer.current); }, []);
  // a result belongs to one date/portfolio: changing either clears it (a running job keeps running until it ends)
  useEffect(() => { setJob((j) => (j && DONE.has(j.status) ? null : j)); setErr(null); }, [date, portfolio]);

  const poll = (id: string) => {
    live.job(id).then((j) => {
      setJob(j);
      if (!DONE.has(j.status)) timer.current = window.setTimeout(() => poll(id), 600);
      else live.status().then(setStatus).catch(() => undefined);
    }).catch((e: Error) => setErr(e.message));
  };
  const start = () => {
    setErr(null);
    live.run(date, withPf ? portfolio : null).then(({ job_id }) => poll(job_id)).catch((e: Error) => setErr(e.message));
  };
  const running = job && !DONE.has(job.status);
  const big = compact ? "text-[16px]" : "text-[13px]";
  const rec = job?.status === "done" ? job.result : null;
  return (
    <section className={`rounded-2xl border border-accent/50 bg-panel ${compact ? "p-6" : "p-4"}`} data-testid="ask-agents">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className={`${compact ? "text-[13px]" : "text-[11px]"} font-semibold uppercase tracking-[0.14em] text-accent`}>Live AI · historical data</div>
          <h2 className={`${compact ? "text-[22px]" : "text-[16px]"} font-semibold text-ink`}>Ask the agents about {longDate(date)}</h2>
          <p className={`${big} max-w-3xl text-muted`} data-testid="live-notice">
            Runs the Graph, Macro and News agents and then the Synthesis agent on a local language model (Qwen 2.5 7B
            through Ollama, on this computer), using only evidence available at the close of this past date. This is live AI
            processing of <b className="text-ink">historical</b> data — not real-time market monitoring. It never changes the risk score or alert.</p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <ModelBadge status={status} />
          {portfolio && <label className={`flex items-center gap-2 ${big} text-muted`}><input type="checkbox" checked={withPf} onChange={(e) => setWithPf(e.target.checked)} disabled={!!running} />Include my portfolio ({portfolio.replace(/,/g, ", ")})</label>}
          {!running ? (
            <button type="button" onClick={start} disabled={!status?.available || status.busy}
              className={`rounded-lg bg-accent px-4 py-2 ${compact ? "text-[16px]" : "text-[13px]"} font-semibold text-[#0b0f14] hover:bg-[#a59cf0] disabled:cursor-not-allowed disabled:opacity-40`}>
              ▶ Ask the agents (≈1 minute)</button>
          ) : (
            <button type="button" onClick={() => job && live.cancel(job.id)} className={`rounded-lg border border-crit/60 px-4 py-2 ${big} text-[#ff9c9c] hover:bg-crit/10`}>■ Cancel</button>
          )}
        </div>
      </div>
      {err && <div className={`mt-3 rounded-md border border-crit/50 bg-crit/10 p-2 ${big} text-[#ff9c9c]`} role="alert">{err}</div>}
      {job && <Progress job={job} compact={compact} />}
      {job && job.status === "error" && <div className={`mt-3 rounded-md border border-crit/50 bg-crit/10 p-3 ${big} text-[#ff9c9c]`} role="alert" data-testid="job-error">The agent run failed: {job.error}. Nothing was shown as an explanation; the saved explanation and the rule-based evidence below are unaffected.</div>}
      {job && (job.status === "cancelled" || job.status === "timeout") && <div className={`mt-3 rounded-md border border-warn/50 bg-warn/10 p-3 ${big} text-warn`} data-testid="job-stopped">{job.status === "timeout" ? "The run hit its time limit and was stopped." : "Run cancelled."} No partial result is shown.</div>}
      {rec && (
        <div className="mt-4" data-testid="live-result">
          <div className={`mb-2 ${big} text-muted`}>Live result · {longDate(rec.decision_date)} · {rec.secs}s · {rec.backend} {rec.model_digest?.slice(0, 12)} · prompt {rec.prompt_version}{rec.portfolio ? ` · portfolio ${rec.portfolio}` : ""}</div>
          <AgentTrace rec={rec} compact={compact} />
        </div>
      )}
      {!rec && !running && (
        <div className="mt-4" data-testid="saved-explanation">
          {saved.data?.record ? (
            <>
              <div className={`mb-2 ${big} text-muted`}>{status?.available ? "Saved explanation for this alert" : "The local model is not available — showing the saved explanation"} (generated offline with the same agents and checks on {saved.data.record.created_utc.slice(0, 10)}).</div>
              <AgentTrace rec={saved.data.record} compact={compact} />
            </>
          ) : saved.data && (
            <p className={`${big} text-muted`}>
              {status?.available ? "Press the button to run the agents on this date." : "The local model is not available, and there is no saved agent explanation for this date."}{" "}
              {saved.data.available_for.length > 0 && <>Saved explanations exist for the start date of every alert ({saved.data.available_for.length}), e.g.{" "}</>}
              {saved.data.available_for.filter((d) => d >= "2020").slice(0, 4).map((d) => (
                <button key={d} type="button" onClick={() => setDate(d)} className="mr-1 rounded border border-line px-1.5 font-mono text-[12px] hover:text-ink">{d}</button>))}
            </p>
          )}
        </div>
      )}
    </section>
  );
}

function ModelBadge({ status }: { status: AgentStatus | null }) {
  if (!status) return <span className="text-[12px] text-dim">checking local model…</span>;
  return status.available
    ? <span className="rounded-md border border-good/50 bg-good/10 px-2 py-0.5 text-[12px] text-[#6ee06e]" data-testid="model-status">● local model ready · {status.model}{status.busy ? " · busy" : ""}</span>
    : <span className="max-w-[320px] rounded-md border border-warn/50 bg-warn/10 px-2 py-0.5 text-[12px] text-warn" data-testid="model-status" title={status.reason}>○ local model not running — start Ollama (<code>ollama serve</code>) to run the agents live</span>;
}

function Progress({ job, compact }: { job: Job; compact: boolean }) {
  const ev = job.events;
  const state = (a: string) => {
    const done = ev.find((e) => e.type === "agent_done" && e.agent === a);
    if (done) return { s: "done" as const, e: done };
    if (ev.some((e) => e.type === "agent_start" && e.agent === a)) return { s: DONE.has(job.status) ? "stopped" as const : "running" as const, e: undefined };
    return { s: "waiting" as const, e: undefined };
  };
  const tok = [...ev].reverse().find((e) => e.type === "tokens");
  const stage = [...ev].reverse().find((e) => e.type === "stage");
  return (
    <div className="mt-3 rounded-xl border border-line bg-panel2 p-3" data-testid="agent-progress" aria-live="polite">
      <div className={`mb-2 flex items-center gap-2 ${compact ? "text-[14px]" : "text-[12px]"} text-muted`}>
        {!DONE.has(job.status) && <span className="h-4 w-4 animate-spin rounded-full border-2 border-dim border-t-accent" />}
        <span>Status: <b className="text-ink" data-testid="job-status">{job.status}</b> · {job.elapsed}s elapsed{stage ? ` · ${stage.detail}` : ""}</span>
      </div>
      <ol className="grid gap-2 md:grid-cols-4">
        {STEPS.map((a) => {
          const { s, e } = state(a);
          return (
            <li key={a} className={`rounded-lg border px-3 py-2 ${s === "running" ? "border-accent bg-accent/10" : "border-line"}`} data-testid={`progress-${a}`} data-state={s}>
              <div className="flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full" style={{ background: AGENT_COLOR[a] }} /><span className={`${compact ? "text-[15px]" : "text-[13px]"} font-medium text-ink`}>{AGENT_NAME[a]}</span></div>
              <div className={`${compact ? "text-[13px]" : "text-[11px]"} text-muted`}>{describe(s, e, a === tok?.agent ? tok : undefined)}</div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function describe(s: string, e: AgentEvent | undefined, tok: AgentEvent | undefined): string {
  if (s === "waiting") return "waiting";
  if (s === "stopped") return "stopped";
  if (s === "running") return tok ? `model writing… ${tok.tokens} chunks` : "reading its facts…";
  if (!e) return "";
  if (e.agent === "synthesis") return !e.called_model ? "not run (too few confirmed facts)" : e.validated ? `confirmed · ${e.secs}s` : `rejected → rule-based summary · ${e.secs}s`;
  if (!e.called_model) return "abstained — no data for this date";
  return `${e.n_valid} confirmed · ${e.n_rejected} rejected · ${e.secs}s`;
}
