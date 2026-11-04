import { describe, expect, it } from "vitest";
import type { Agents, Edge, Snapshot } from "./api";
import { edgePlain, evidenceRows, headline, newsRow, tone, whatIsHappening, whyAlert } from "./plain";

const snap = (over: Partial<Snapshot> = {}): Snapshot => ({
  date: "2020-02-24", requested: "2020-02-24",
  status: { level: "alert", summary: "", unusual: [{ indicator: "5-day VIX change", percentile: 90 }], rule: "" },
  sii: { value: 0.93, percentile: 85, change_21d: 0.9, spark: [], spark_dates: [] },
  regime: { p_stress: 0.9997, state: "stressed", basis: "" },
  vix: { value: 17.08, percentile: 78, change_5d: 2.9, change_5d_percentile: 90 },
  nfci: { value: -0.83, percentile: 30 }, edge_turnover: { value: 17, percentile: 73, avg_4_windows: 18.5 },
  model: { method: "bayes_logit:combined", score: 0.829, threshold: 0.317, alert: true },
  graph_window_end: "2020-02-13", recent_alerts: [], ...over,
});

describe("plain-language templates use the saved numbers only", () => {
  it("states score, alert line and the comparison", () => {
    expect(whyAlert(snap()).main).toBe("Its score was 0.83, above its alert line of 0.32 — so it raised an alert.");
    const no = snap({ model: { method: "m", score: 0.1, threshold: 0.3, alert: false } });
    expect(whyAlert(no).main).toContain("below its alert line of 0.30 — so no alert");
    expect(whyAlert(no).caveat).toContain("not more accurate than the fear gauge");
  });
  it("describes what is happening without inventing causes", () => {
    const t = whatIsHappening(snap()).join(" ");
    expect(t).toContain('"stressed" (100% chance');
    expect(t).toContain("17.1 — normal");
    expect(t).toContain("looser than the long-run average");
    expect(t).toContain("the weekly jump in the fear gauge (90th percentile)");
    expect(t).not.toMatch(/because|caused|will /);
    expect(headline(snap())).toBe("The warning model raised an alert");
  });
  it("uses the same 20/80 percentile bands as the backend", () => {
    expect([tone(85), tone(50), tone(10), tone(null)]).toEqual(["high", "normal", "low", "unknown"]);
  });
  it("flags the fear gauge when its weekly move is unusual even if the level is normal", () => {
    expect(evidenceRows(snap(), null, null)[0].tone).toBe("high");
  });
  it("treats missing news as missing data, never as neutral", () => {
    const a = { agents: { news: { evidence: [], unavailable: [], abstained: true } } } as unknown as Agents;
    expect(newsRow(a).text).toContain("missing data, not 'no news'");
    expect(newsRow(a).tone).toBeNull();
  });
  it("describes edges with direction and estimate language", () => {
    const e: Edge = { id: "TLT->LQD", source: "TLT", target: "LQD", before: 0, after: 0.4, change: "appeared", text: "", active: true, sign: "positive" };
    expect(edgePlain(e)).toBe("the link from Long-term US government bonds to High-quality corporate bonds is a new link (they tend to move in the same direction).");
    expect(edgePlain({ ...e, before: 0.5, after: 0, change: "disappeared" })).toContain("has disappeared.");
  });
});
