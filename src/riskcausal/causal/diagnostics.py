"""Overlapping-window stability diagnostic (proposal Risk 2): is the change between consecutive rolling windows
larger than the estimator's own sampling noise?

For an evenly spaced sample of windows, the window's rows are resampled with a moving-block bootstrap and
VarLiNGAM is refit `n_boot` times. The noise floor is the B0 change (SHD, Frobenius) between pairs of refits of
the *same* window; it is compared with the actual change between that window and the previous one. Block joins
slightly distort lagged dependence, so the floor is an approximation (reported as such).
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from riskcausal.causal.varlingam_rolling import RollingGraphs, fit_window
from riskcausal.config import Config
from riskcausal.data.panel import processed_dir
from riskcausal.registry import write_json

log = logging.getLogger(__name__)


def block_bootstrap(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def _b0_change(a: np.ndarray, b: np.ndarray, thr: float) -> tuple[float, float]:
    return float(((np.abs(a) > thr) != (np.abs(b) > thr)).sum()), float(np.sqrt(((a - b) ** 2).sum()))


def noise_floor(cfg: Config, g: RollingGraphs, out_dir: Path, n_windows: int = 100, n_boot: int = 3,
                block: int = 10) -> Path:
    panel = pd.read_parquet(processed_dir(cfg) / "graph_panel.parquet")
    data = panel.to_numpy(float)
    W, thr = cfg.graph.window, cfg.graph.edge_threshold
    rng = np.random.default_rng(cfg.seed)
    picks = np.unique(np.linspace(1, len(g.ends) - 1, n_windows).astype(int))
    rows = []
    for w in picks:
        end = panel.index.get_loc(g.ends[w])
        X = data[end - W + 1: end + 1]
        fits = [fit_window(X[block_bootstrap(W, block, rng)], cfg.graph.max_lag, cfg.graph.lag_criterion,
                           cfg.graph.prune, cfg.seed)[0][0] for _ in range(n_boot)]
        noise = [_b0_change(a, b, thr) for a, b in itertools.combinations(fits, 2)]
        actual = _b0_change(g.B[w, 0], g.B[w - 1, 0], thr)
        rows.append({"window_end": g.ends[w], "actual_shd": actual[0], "actual_frob": actual[1],
                     "noise_shd_mean": float(np.mean([x[0] for x in noise])),
                     "noise_frob_mean": float(np.mean([x[1] for x in noise]))})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "noise_floor.csv", index=False)
    summary = {
        "n_windows": len(df), "n_boot": n_boot, "block": block,
        "median_actual_shd_consecutive": float(df["actual_shd"].median()),
        "median_noise_shd_same_window": float(df["noise_shd_mean"].median()),
        "median_actual_frob_consecutive": float(df["actual_frob"].median()),
        "median_noise_frob_same_window": float(df["noise_frob_mean"].median()),
        "share_windows_actual_frob_above_noise": float((df["actual_frob"] > df["noise_frob_mean"]).mean()),
        "share_windows_actual_shd_above_noise": float((df["actual_shd"] > df["noise_shd_mean"]).mean()),
    }
    write_json(out_dir / "noise_floor.json", summary)
    log.info("noise floor: %s", summary)
    return out_dir / "noise_floor.json"
