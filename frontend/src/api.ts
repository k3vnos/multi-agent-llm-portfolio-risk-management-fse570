// Typed, cached GET client for the local read-only API. Nothing here writes or calls a model.

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
