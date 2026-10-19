"""Control features (all known at the close of t): VIX level/changes (PIT), realised vol, average correlation."""

from __future__ import annotations

import pandas as pd

from riskcausal.data.transforms import realized_vol, rolling_avg_corr


def control_features(returns: pd.DataFrame, controls_pit: pd.DataFrame, regime_series: str,
                     vol_window: int, corr_window: int = 63) -> pd.DataFrame:
    out = pd.DataFrame(index=returns.index)
    if "VIXCLS" in controls_pit:
        vix = controls_pit["VIXCLS"]
        out["vix_level"] = vix
        out["vix_chg_5d"] = vix - vix.shift(5)
    out["rv21"] = realized_vol(returns[regime_series], vol_window)
    out["avg_corr_63"] = rolling_avg_corr(returns, corr_window)
    return out
