"""Exploratory banking/credit-stress check B2 (robustness.yaml): descriptive only, no models.

A 2-regime Markov model on the point-in-time BAA10Y level (Moody's Baa minus 10y Treasury, as known at each
session), fit on the full sample (retrospective). Reports stress onsets and the regime path around March 2023.
This never replaces or modifies the SPY-volatility target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskcausal.config import Config
from riskcausal.data.panel import processed_dir
from riskcausal.regimes.markov import cached_fit_ms
from riskcausal.regimes.targets import label_set


def credit_regimes(cfg: Config, series: str = "BAA10Y") -> dict:
    ctrl = pd.read_parquet(processed_dir(cfg) / "controls_pit.parquet")
    y = ctrl[series].dropna()
    fit = cached_fit_ms(y, 2, cfg.seed, cfg.data_path / "interim" / "markov_cache")
    L = label_set(fit.smoothed, cfg.regime.min_stress_days, cfg.regime.min_calm_days, cfg.target.horizon)
    win = L.loc["2022-10-01":"2023-06-30"]
    return {
        "series": series, "definition": "2-regime Markov on point-in-time level; stress = higher-mean regime",
        "regime_means": [float(m) for m in fit.regime_means], "stress_share": float(L["stress"].mean()),
        "onsets": [str(d.date()) for d in L.index[L["onset"] == 1]],
        "onsets_2022_10_to_2023_06": [str(d.date()) for d in win.index[win["onset"] == 1]],
        "stress_share_2022_10_to_2023_06": float(win["stress"].mean()),
        "level_2023": {d: float(y.asof(pd.Timestamp(d))) for d in
                       ("2023-01-03", "2023-03-01", "2023-03-08", "2023-03-13", "2023-03-20", "2023-04-03")},
        "p_stress_2023_03": {str(d.date()): round(float(v), 3)
                             for d, v in fit.smoothed.loc["2023-03-01":"2023-03-31"].iloc[::3].items()},
        "level_quantiles_full": dict(zip(["p50", "p90", "p99"], map(float, np.quantile(y, [.5, .9, .99])))),
    }
