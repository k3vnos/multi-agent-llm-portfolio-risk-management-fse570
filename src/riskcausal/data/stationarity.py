"""ADF + KPSS stationarity checks on graph inputs, full sample and sub-periods (descriptive report)."""

from __future__ import annotations

import warnings

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

SUBPERIODS = [("2005-01-01", "2009-12-31"), ("2010-01-01", "2014-12-31"),
              ("2015-01-01", "2019-12-31"), ("2020-01-01", "2030-12-31")]


def stationarity_report(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    periods = [("full", None, None)] + [(f"{a[:4]}-{b[:4]}", a, b) for a, b in SUBPERIODS]
    for col in panel.columns:
        for name, a, b in periods:
            x = panel[col].loc[a:b].dropna() if a else panel[col].dropna()
            if len(x) < 100:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # KPSS warns when p is outside its lookup table
                _adf_stat, adf_p, *_ = adfuller(x, autolag="AIC")
                _kpss_stat, kpss_p, *_ = kpss(x, regression="c", nlags="auto")
            rows.append({"series": col, "period": name, "n": len(x), "adf_p": round(adf_p, 4),
                         "kpss_p": round(kpss_p, 4),
                         "stationary_both": bool(adf_p < 0.05 and kpss_p > 0.05)})
    return pd.DataFrame(rows)
