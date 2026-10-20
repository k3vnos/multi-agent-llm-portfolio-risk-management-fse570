"""Case-study and timeline figures (small multiples: one score per panel, no dual axes).

Figures are illustrative only (proposal §3: two events are not enough to conclude much). If the run is flagged
synthetic, every figure carries a SYNTHETIC DATA watermark.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from riskcausal.registry import read_json

SURFACE, INK, INK2, GRID, STRESS = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d3"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # reference categorical slots 1–3
CASES = {"case_2020": ("2019-10-01", "2020-06-30", "COVID-19 crash (Feb–Mar 2020)"),
         "case_2023": ("2022-10-01", "2023-06-30", "Regional-bank stress / SVB (Mar 2023)")}
SHOW = [("bayes_logit:combined", "Bayesian logit — instability + controls (P(onset within h))"),
        ("inst_composite", "Graph-instability composite (z-score, train-only scaling)"),
        ("vix", "VIX level baseline")]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)


def _shade(ax, labels: pd.DataFrame):
    st = labels["stress"].fillna(0)
    runs = (st != st.shift()).cumsum()
    for _, g in st[st == 1].groupby(runs[st == 1]):
        ax.axvspan(g.index[0], g.index[-1], color=STRESS, alpha=0.6, linewidth=0)
    for d in labels.index[labels["onset"] == 1]:
        ax.axvline(d, color=INK, linewidth=1, linestyle=(0, (2, 2)))


def _figure(run_dir: Path, start: str, end: str, title: str, out: Path, synthetic: bool):
    P = pd.read_parquet(run_dir / "predictions.parquet")
    L = pd.read_parquet(run_dir / "labels_eval.parquet").loc[start:end]
    F = pd.read_parquet(run_dir / "features.parquet").loc[start:end]
    fig, axes = plt.subplots(len(SHOW) + 1, 1, figsize=(10, 9), sharex=True, facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    _shade(ax, L)
    M = read_json(run_dir / "metrics.json")
    tgt = M.get("target_series", "SPY")
    rv = L["target_rv"] if "target_rv" in L else F["rv21"]
    ax.plot(rv.index, rv, color=INK2, linewidth=1.5)
    ax.set_title(f"{tgt} 21-day realised volatility (annualised) — shaded: retrospective stress regime; "
                 "dashed: onset", fontsize=9, color=INK, loc="left")
    for ax, (method, label), color in zip(axes[1:], SHOW, SERIES):
        _style(ax)
        _shade(ax, L)
        g = P[P["method"] == method].set_index("date").loc[start:end]
        if g.empty:
            ax.text(0.5, 0.5, "no out-of-sample predictions in this window", transform=ax.transAxes,
                    ha="center", color=INK2, fontsize=9)
        else:
            ax.plot(g.index, g["score"], color=color, linewidth=2)
            ax.step(g.index, g["threshold"], where="post", color=INK2, linewidth=1, linestyle="--")
            a = g[g["alert"]]
            ax.scatter(a.index, a["score"], s=10, color=color, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        ax.set_title(f"{label} — dashed: threshold chosen on validation data; dots: alert days",
                     fontsize=9, color=INK, loc="left")
    fig.suptitle(f"{title}: out-of-sample scores (illustrative, not evidence on its own)", fontsize=11,
                 color=INK, x=0.01, ha="left")
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC DATA", fontsize=48, color="#e34948", alpha=0.25, ha="center",
                 va="center", rotation=30)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def make_figures(run_dir: Path) -> list[Path]:
    run_dir = Path(run_dir)
    synthetic = bool(read_json(run_dir / "provenance.json").get("synthetic"))
    figdir = run_dir / "figures"
    figdir.mkdir(exist_ok=True)
    outs = []
    for name, (a, b, title) in CASES.items():
        out = figdir / f"{name}.png"
        _figure(run_dir, a, b, title, out, synthetic)
        outs.append(out)
    P = pd.read_parquet(run_dir / "predictions.parquet")
    out = figdir / "timeline_full_test.png"
    _figure(run_dir, str(P["date"].min().date()), str(P["date"].max().date()), "Full out-of-sample span", out,
            synthetic)
    outs.append(out)
    return outs
