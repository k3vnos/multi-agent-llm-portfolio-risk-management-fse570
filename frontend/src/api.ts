// Typed client for the local API. GET responses are cached; the only writes are starting / cancelling a live
// local-model agent job (`live.run`, `live.cancel`), which never change a score, alert or saved result.

export type Level = "alert" | "unusual" | "normal";

export interface ReplayEvent {
  id: string; label: string; period: string; available: boolean; replay_start?: string; focus?: string;
  target?: "primary" | "exploratory"; note?: string;
}
export interface Meta {
  mode: string; start: string; end: string; monitored_model: string; events: ReplayEvent[];
  nodes: { id: string; name: string; group: string }[]; universe_etfs: string[]; graph_window: number;
  graph_step: number; edge_threshold: number; news_coverage_end: string; status_rule: string;
}
export interface Series {
  dates: string[]; sii: (number | null)[]; sii_pct: (number | null)[]; vix: (number | null)[];
  vix_pct: (number | null)[]; nfci: (number | null)[]; nfci_pct: (number | null)[];
  model_score: (number | null)[]; model_threshold: (number | null)[]; regime_p_stress: (number | null)[];
  model_alert: boolean[]; vix_alert: boolean[]; retrospective_stress: (number | null)[];
}
export interface RecentAlert { method: string; start: string; end: string; ongoing: boolean; peak_score: number; threshold: number }
export interface Snapshot {
  date: string; requested: string;
  status: { level: Level; summary: string; unusual: { indicator: string; percentile: number }[]; rule: string };
  sii: { value: number | null; percentile: number | null; change_21d: number | null; spark: (number | null)[]; spark_dates: string[] };
  regime: { p_stress: number | null; state: "calm" | "stressed" | null; basis: string };
  vix: { value: number | null; percentile: number | null; change_5d: number | null; change_5d_percentile: number | null };
  nfci: { value: number | null; percentile: number | null };
  edge_turnover: { value: number | null; percentile: number | null; avg_4_windows: number | null };
  model: { method: string; score: number | null; threshold: number | null; alert: boolean };
  graph_window_end: string | null; recent_alerts: RecentAlert[];
}
export interface Edge {
  id: string; source: string; target: string; before: number; after: number; change: string; text: string;
  active: boolean; sign: string;
}
export interface Network {
  date: string; window_end: string; previous_window_end: string; lag_windows: number; edge_threshold: number;
  nodes: { id: string; name: string; group: string; out_degree: number; in_degree: number; changed: number }[];
  edges: Edge[]; B0: number[][]; B0_previous: number[][]; node_order: string[]; n_active: number;
  counts: Record<string, number>; caveat: string;
}
export interface EvidenceItem {
  id: string; kind: string; label: string; available: string; description: string; source_text: string;
  values: Record<string, number>;
}
export interface Claim { text: string; evidence_ids: string[] }
export interface Agents {
  date: string; method: string; cutoff_utc: string;
  agents: {
    graph: { title: string; role: string; evidence: EvidenceItem[] };
    macro: { title: string; role: string; evidence: EvidenceItem[] };
    news: { title: string; role: string; evidence: EvidenceItem[]; unavailable: string[]; abstained: boolean };
  };
  synthesis: { title: string; kind: string; briefing: Claim[] };
  llm_draft: null | {
    status: string; flags: string[]; claims: Claim[]; uncertainty: string; model: string; evaluation_set: string;
    label: string; specialists: Record<string, { abstain: boolean; abstain_reason: string; claims: Claim[]; n_rejected: number }>;
  };
  llm_draft_available_for: string[]; reliability_retrospective: string[];
}
export interface MethodRow {
  method: string; roc_auc: number | null; roc_ci95: [number, number] | null; pr_auc: number | null;
  pr_ci95: [number, number] | null; base_rate: number | null; n_onsets: number; n_detected: number;
  recall: number | null; recall_ci95: [number, number] | null; precision: number | null;
  precision_ci95: [number, number] | null; false_alarms_per_year: number | null; calm_alert_share: number | null;
  median_lead: number | null;
}
export interface Diff { comparison: string; delta_roc: number; roc_ci95: [number, number]; excludes_zero: boolean; delta_pr: number; pr_ci95: [number, number] }
export interface MetricsResp {
  meta: { run_id: string; git_sha: string; audit_passed: boolean; kind: string; role: string };
  data: { span: [string, string]; alert_rule: string; truth: string; ci_method: string; methods: MethodRow[];
          differences: Diff[]; audit: { check: string; passed: boolean; n_violations: number; n_checked: number }[];
          label_summary: { onsets: string[]; stress_share: number; regime_annualised_vol: number[] } };
}
export interface Research {
  primary: MetricsResp; exploratory: MetricsResp; lead_times: Record<string, number[]>;
  reproducibility: { status: string; twin_run_id?: string; files: Record<string, boolean> }; conclusion: string;
}
export interface Episode { start: string; end: string; outcome_retrospective: string; peak_score: number }
export interface RunsResp {
  runs: { role: string; run_id: string; profile: string; test_span: [string, string]; audit_passed: boolean; git_sha: string; target_series: string }[];
  explained_method: string; headline_methods: string[];
}

const cache = new Map<string, Promise<unknown>>();

export function get<T>(path: string): Promise<T> {
  if (!cache.has(path)) {
    const p = fetch(path, { headers: { Accept: "application/json" } }).then(async (r) => {
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        throw new Error((body as { detail?: string }).detail || `${r.status} ${r.statusText}`);
      }
      return r.json();
    });
    p.catch(() => cache.delete(path));
    cache.set(path, p);
  }
  return cache.get(path) as Promise<T>;
}

export const api = {
  meta: () => get<Meta>("/replay/meta"),
  series: () => get<Series>("/replay/series"),
  snapshot: (d: string) => get<Snapshot>(`/replay/snapshot/${d}`),
  network: (d: string, lag = 1) => get<Network>(`/replay/network/${d}?lag=${lag}`),
  agents: (d: string, method?: string) => get<Agents>(`/replay/agents/${d}${method ? `?method=${encodeURIComponent(method)}` : ""}`),
  research: () => get<Research>("/replay/research"),
  runs: () => get<RunsResp>("/runs"),
  episodes: (runId: string, method: string) =>
    get<{ data: { episodes: Episode[] } }>(`/runs/${runId}/alerts?method=${encodeURIComponent(method)}`),
};

// ---------------------------------------------------------------- M10: v4 agents, live runs, Portfolio Lens
export interface Fact {
  id: string; agent: string; kind: string; label: string; allowed: string[]; prompt: string; text: string;
  source: string; available: string; values: Record<string, unknown>;
}
export interface V4Claim {
  fact_id: string; label: string; note: string; source: string | null; kind?: string; text?: string; true_label?: string;
  reasons?: string[]; note_rejected?: { text: string; reasons: string[] };
}
export interface V4Specialist {
  role: "graph" | "macro" | "news"; inputs: Fact[]; claims: V4Claim[]; rejected: V4Claim[]; abstain: boolean;
  model_abstained: boolean; abstain_reason: string; called_model: boolean; prompt?: string; raw_output?: unknown;
  meta?: { wall_secs?: number; eval_count?: number; prompt_eval_count?: number };
}
export interface V4Synthesis {
  role: "synthesis"; called_model: boolean; validated?: boolean; abstain: boolean; abstain_reason?: string;
  inputs: { validated_facts: string[]; abstained: string[] }; fallback_text: string; true_overall: string;
  key_facts?: string[]; overall?: string; text?: string | null; note?: string;
  note_rejected?: { text: string; reasons: string[] }; rejected: { reasons: string[] }[];
  meta?: { wall_secs?: number };
}
export interface V4Record {
  alert_id: string; decision_date: string; cutoff_utc: string; method: string; backend: string;
  model_digest: string | null; prompt_version: string; created_utc: string; secs: number; facts: Fact[];
  coverage_notes: string[]; specialists: V4Specialist[]; synthesis: V4Synthesis; disclaimer: string;
  group?: string; portfolio?: string | null;
}
export interface AgentEvent { type: string; agent?: string; stage?: string; detail?: string; tokens?: number; t: number;
  called_model?: boolean; abstain?: boolean; n_valid?: number; n_rejected?: number; validated?: boolean; secs?: number; status?: string; n_inputs?: number }
export interface Job {
  id: string; date: string; method: string; portfolio: string | null;
  status: "queued" | "running" | "done" | "error" | "cancelled" | "timeout";
  error: string | null; result: V4Record | null; events: AgentEvent[]; elapsed: number; notice: string;
}
export interface AgentStatus { available: boolean; model: string; digest?: string; reason?: string; busy: boolean; notice: string; fallback: string }
export interface Holding { ticker: string; name: string; weight: number; risk_share: number; vol_63d: number; return_21d: number; node: string | null; node_is_proxy: boolean }
export interface PortfolioLink { id: string; source: string; target: string; before: number; after: number; change: string; text: string; holdings: string[]; weight_touched: number }
export interface PortfolioAnalysis {
  date: string; spec: string; label: string; data_through: string; n_sessions: number;
  metrics: { return_21d: number; return_21d_pct: number | null; return_252d: number; vol_63d: number; vol_63d_pct: number | null;
    vol_252d: number; var95_hist: number; cvar95_hist: number; var95_param: number; max_drawdown_252d: number;
    avg_comovement_63d: number | null; avg_comovement_prev_63d: number | null };
  holdings: Holding[];
  network: { window_end: string; previous_window_end: string; nodes: Record<string, string[]>; not_in_graph: string[];
    links: PortfolioLink[]; n_links: number; weight_on_changed_nodes: number };
  series: { dates: string[]; value: number[]; drawdown: number[] }; definitions: string;
}
export interface Universe { etfs: { ticker: string; name: string; node: string | null; first_date: string }[];
  examples: { id: string; name: string; spec: string; about: string }[]; label: string }

/** Uncached request (live job state and the two approved POST routes). */
async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, { ...init, headers: { Accept: "application/json", "Content-Type": "application/json" } });
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error((body as { detail?: string }).detail || `${r.status} ${r.statusText}`);
  }
  return r.json() as Promise<T>;
}

export const live = {
  status: () => req<AgentStatus>("/agents/status"),
  run: (date: string, portfolio?: string | null) =>
    req<{ job_id: string }>("/agents/run", { method: "POST", body: JSON.stringify({ date, portfolio: portfolio || null }) }),
  job: (id: string) => req<Job>(`/agents/jobs/${id}`),
  cancel: (id: string) => req<unknown>(`/agents/jobs/${id}/cancel`, { method: "POST" }),
  saved: (date: string) => get<{ date: string; available_for: string[]; record: V4Record | null }>(`/agents/saved/${date}`),
  universe: () => get<Universe>("/portfolio/universe"),
  analyze: (spec: string, date: string) =>
    get<PortfolioAnalysis>(`/portfolio/analyze?spec=${encodeURIComponent(spec)}&date=${date}`),
};
