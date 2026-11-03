import { useMemo } from "react";
import type { Edge, Network } from "../api";

export type Mode = "current" | "change";
export type Selection = { type: "node"; id: string } | { type: "edge"; id: string } | null;

const GROUP_COLOR: Record<string, string> = {
  equity: "#9085e9", rates: "#3987e5", credit: "#d95926", commodity: "#c98500", fx: "#199e70", macro: "#8a93a0",
};
export const CHANGE_COLOR: Record<string, string> = {
  appeared: "#9085e9", disappeared: "#6b7480", strengthened: "#d95926", weakened: "#199e70", "sign flipped": "#eda100",
};
export const SIGN_COLOR = { positive: "#e66767", negative: "#3987e5" };

export const material = (e: Edge) =>
  ["appeared", "disappeared", "sign flipped"].includes(e.change) || Math.abs(e.after - e.before) > 0.1;

const label = (id: string) => id.replace("d_VIXCLS", "ΔVIX").replace("d_BAA10Y", "ΔBaa").replace("d_T10Y2Y", "ΔCurve");

export function NetworkGraph({ net, mode, selection, onSelect, size = 460 }: {
  net: Network; mode: Mode; selection: Selection; onSelect: (s: Selection) => void; size?: number;
}) {
  const R = size / 2 - 52, cx = size / 2, cy = size / 2;
  const pos = useMemo(() => {
    const p: Record<string, { x: number; y: number }> = {};
    net.nodes.forEach((n, i) => {
      const a = -Math.PI / 2 + (2 * Math.PI * i) / net.nodes.length;
      p[n.id] = { x: cx + R * Math.cos(a), y: cy + R * Math.sin(a) };
    });
    return p;
  }, [net.nodes, R, cx, cy]);

  const shown = net.edges.filter((e) => (mode === "current" ? e.active : material(e)));
  const selNode = selection?.type === "node" ? selection.id : null;
  const selEdge = selection?.type === "edge" ? selection.id : null;
  const touches = (e: Edge) => !selNode || e.source === selNode || e.target === selNode;

  const path = (e: Edge) => {
    const a = pos[e.source], b = pos[e.target];
    const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
    const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy) || 1;
    const bend = 0.18 * len;                                    // curve so A→B and B→A do not overlap
    const qx = mx + (-dy / len) * bend * 0.6 + (cx - mx) * 0.25, qy = my + (dx / len) * bend * 0.6 + (cy - my) * 0.25;
    const r = 17;                                               // stop at the node border
    const ex = b.x - ((b.x - qx) / Math.hypot(b.x - qx, b.y - qy)) * r, ey = b.y - ((b.y - qy) / Math.hypot(b.x - qx, b.y - qy)) * r;
    const sx = a.x + ((qx - a.x) / Math.hypot(qx - a.x, qy - a.y)) * r, sy = a.y + ((qy - a.y) / Math.hypot(qx - a.x, qy - a.y)) * r;
    return `M${sx},${sy} Q${qx},${qy} ${ex},${ey}`;
  };
  const color = (e: Edge) => (mode === "current" ? SIGN_COLOR[e.after >= 0 ? "positive" : "negative"] : CHANGE_COLOR[e.change] || "#6b7480");
  const width = (e: Edge) => 1 + 3.2 * Math.min(1, Math.abs(mode === "change" && e.change === "disappeared" ? e.before : e.after));

  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="h-auto w-full" role="img" aria-label="Estimated directed-dependence network"
      onClick={() => onSelect(null)} data-testid="network-svg">
      <defs>
        {[...Object.entries(CHANGE_COLOR), ...Object.entries(SIGN_COLOR)].map(([k, c]) => (
          <marker key={k} id={`arrow-${k.replace(" ", "-")}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill={c} />
          </marker>
        ))}
      </defs>
      <circle cx={cx} cy={cy} r={R} fill="none" stroke="#1a212b" strokeDasharray="2 5" />
      {shown.map((e) => {
        const key = mode === "current" ? (e.after >= 0 ? "positive" : "negative") : e.change;
        const dim = !touches(e) || (selEdge && selEdge !== e.id);
        return (
          <g key={e.id} onClick={(ev) => { ev.stopPropagation(); onSelect({ type: "edge", id: e.id }); }} className="cursor-pointer" data-edge={e.id}>
            <path d={path(e)} fill="none" stroke="transparent" strokeWidth={12} />
            <path d={path(e)} fill="none" stroke={color(e)} strokeWidth={selEdge === e.id ? width(e) + 1.5 : width(e)}
              strokeDasharray={mode === "change" && e.change === "disappeared" ? "4 4" : undefined}
              opacity={dim ? 0.08 : 0.85} markerEnd={`url(#arrow-${key.replace(" ", "-")})`} />
          </g>
        );
      })}
      {net.nodes.map((n) => {
        const p = pos[n.id];
        const on = selNode === n.id;
        const changed = mode === "change" && n.changed > 0;
        return (
          <g key={n.id} transform={`translate(${p.x},${p.y})`} className="cursor-pointer" data-node={n.id}
            onClick={(ev) => { ev.stopPropagation(); onSelect(on ? null : { type: "node", id: n.id }); }}>
            <title>{`${n.id} — ${n.name}`}</title>
            {changed && <circle r={21} fill="none" stroke="#eda100" strokeOpacity={0.5} />}
            <circle r={16} fill="#0f141b" stroke={GROUP_COLOR[n.group] || "#8a93a0"} strokeWidth={on ? 3 : 1.6} />
            <text textAnchor="middle" dy="3.5" fontSize={n.id.length > 4 ? 8 : 9.5} fill="#e6e8eb" fontFamily="ui-monospace,monospace">{label(n.id)}</text>
          </g>
        );
      })}
    </svg>
  );
}

export function NetworkLegend({ mode }: { mode: Mode }) {
  const arrow = (c: string, dash?: boolean) => (
    <svg width="22" height="8" aria-hidden><line x1="0" y1="4" x2="16" y2="4" stroke={c} strokeWidth="2.2" strokeDasharray={dash ? "3 3" : undefined} /><path d="M15,0 L22,4 L15,8 z" fill={c} /></svg>
  );
  const items = mode === "current"
    ? [["Positive dependence", SIGN_COLOR.positive], ["Negative dependence", SIGN_COLOR.negative]]
    : [["Newly appeared", CHANGE_COLOR.appeared], ["Strengthened", CHANGE_COLOR.strengthened], ["Weakened", CHANGE_COLOR.weakened],
       ["Sign flipped", CHANGE_COLOR["sign flipped"]], ["Disappeared", CHANGE_COLOR.disappeared]];
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted">
      {items.map(([t, c]) => <span key={t} className="flex items-center gap-1.5">{arrow(c, t === "Disappeared")}{t}</span>)}
      <span className="flex items-center gap-1.5">thicker = stronger</span>
    </div>
  );
}
