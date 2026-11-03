import { useState, type ReactNode } from "react";
import type { Level } from "../api";

export function Card({ title, subtitle, info, right, children, className = "" }: {
  title?: ReactNode; subtitle?: ReactNode; info?: string; right?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`rounded-xl border border-line bg-panel ${className}`}>
      {(title || right) && (
        <header className="flex flex-wrap items-center justify-between gap-2 px-4 pt-3.5">
          <div className="flex min-w-0 items-baseline gap-2">
            {title && <h2 className="text-[13px] font-semibold text-ink">{title}</h2>}
            {info && <Info text={info} />}
            {subtitle && <span className="truncate text-xs text-muted">{subtitle}</span>}
          </div>
          {right}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function Info({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <span className="relative inline-flex" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <button type="button" aria-label="What is this?" onFocus={() => setOpen(true)} onBlur={() => setOpen(false)}
        onClick={() => setOpen((o) => !o)}
        className="flex h-4 w-4 items-center justify-center rounded-full border border-dim text-[10px] leading-none text-muted hover:border-accent hover:text-ink">
        ?
      </button>
      {open && (
        <span role="tooltip" className="absolute left-1/2 top-6 z-50 w-72 -translate-x-1/2 rounded-lg border border-line bg-panel2 p-3 text-xs font-normal leading-relaxed text-ink shadow-xl">
          {text}
        </span>
      )}
    </span>
  );
}

export function Label({ children, info }: { children: ReactNode; info?: string }) {
  return (
    <div className="flex items-center gap-1.5 text-[10.5px] font-medium uppercase tracking-[0.12em] text-muted">
      {children}{info && <Info text={info} />}
    </div>
  );
}

const LEVEL = {
  alert: { cls: "border-crit/50 bg-crit/15 text-[#ff8a8a]", icon: "▲", text: "Model alert" },
  unusual: { cls: "border-warn/50 bg-warn/10 text-warn", icon: "◆", text: "Unusual activity" },
  normal: { cls: "border-good/50 bg-good/10 text-[#5bd65b]", icon: "●", text: "Normal conditions" },
} as const;

export function LevelBadge({ level, small = false }: { level: Level; small?: boolean }) {
  const l = LEVEL[level];
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-md border font-medium ${l.cls} ${small ? "px-1.5 py-0.5 text-[11px]" : "px-2 py-1 text-xs"}`}>
      <span aria-hidden>{l.icon}</span>{l.text}
    </span>
  );
}

export function Pill({ children, tone = "muted" }: { children: ReactNode; tone?: "muted" | "accent" | "warn" | "crit" | "good" }) {
  const t = { muted: "border-line text-muted", accent: "border-accent/50 text-accent", warn: "border-warn/50 text-warn",
              crit: "border-crit/50 text-[#ff8a8a]", good: "border-good/50 text-[#5bd65b]" }[tone];
  return <span className={`inline-flex items-center rounded border px-1.5 py-0.5 font-mono text-[10.5px] ${t}`}>{children}</span>;
}

export function Loading({ label = "Loading saved results…" }: { label?: string }) {
  return (
    <div className="flex min-h-[120px] items-center justify-center gap-3 text-sm text-muted" role="status">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-dim border-t-accent" />{label}
    </div>
  );
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div role="alert" className="rounded-lg border border-crit/40 bg-crit/10 p-4 text-sm text-[#ffb3b3]">
      Could not load this view: {message}. Is the local server running? (<span className="font-mono">./run.sh status</span>)
    </div>
  );
}

export function Unavailable({ children }: { children: ReactNode }) {
  return <div className="rounded-lg border border-dashed border-line p-4 text-sm text-muted">{children}</div>;
}

export function Button({ children, onClick, active = false, disabled = false, title }: {
  children: ReactNode; onClick?: () => void; active?: boolean; disabled?: boolean; title?: string;
}) {
  return (
    <button type="button" onClick={onClick} disabled={disabled} title={title}
      className={`rounded-md border px-2.5 py-1 text-xs transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
        active ? "border-accent bg-accent/20 text-ink" : "border-line bg-panel2 text-muted hover:border-dim hover:text-ink"}`}>
      {children}
    </button>
  );
}
