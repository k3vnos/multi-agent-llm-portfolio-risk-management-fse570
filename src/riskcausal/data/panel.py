"""Build the processed daily panel for a profile from the latest raw snapshot.

Outputs (data/processed/<profile>_<snapshot>/):
- returns.parquet   : log returns of profile ETFs on NYSE sessions (NaN before listing — never back-filled)
- target_returns.parquet: log returns of the series that define regime targets (labels only, never features)
- graph_panel.parquet: graph nodes aligned by *observation date*: ETF log returns + first differences of daily
                       macro series. Graph-derived features are lagged by `graph_feature_lag` sessions downstream
                       so they only use values published by the decision date.
- graph_avail.parquet: per graph row, the first session at whose close every value in that row was published
                       (ETF closes: same session; FRED daily series: per-series `availability`, see
                       configs/universe.yaml and data/pit.py). A graph
                       window is usable only once all of its rows are (LEAKAGE_AUDIT row 2).
- controls_pit.parquet: macro levels as known at each session (PIT, release-based availability)
- stationarity.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from riskcausal.config import Config
from riskcausal.data.calendar import add_bdays, nyse_sessions
from riskcausal.data.pit import pit_from_observations, pit_from_vintages, release_availability
from riskcausal.data.stationarity import stationarity_report
from riskcausal.data.transforms import log_returns
from riskcausal.registry import read_json, write_json

log = logging.getLogger(__name__)


def latest_snapshot(cfg: Config) -> str:
    return read_json(cfg.data_path / "raw" / "latest_snapshot.json")["snapshot"]


def processed_dir(cfg: Config) -> Path:
    return cfg.data_path / "processed" / f"{cfg.profile}_{latest_snapshot(cfg)}"


def releases_dir(cfg: Config) -> Path:
    return cfg.data_path / "raw" / "fred_releases" / latest_snapshot(cfg)


def load_first_releases(cfg: Config, sid: str) -> pd.Series:
    path = releases_dir(cfg) / f"{sid}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no ALFRED first-release dates for {sid} at {path}; run "
                                f"`riskcausal fetch-releases --profile {cfg.profile}`")
    return pd.read_parquet(path)["first_release"]


def _availability(cfg: Config, sid: str, obs: pd.DatetimeIndex, sessions_all: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Session at whose close each observation of a non-revised daily series is known (universe `availability`)."""
    lag = cfg.pit.macro_release_lag_bdays
    if cfg.universe["fred"][sid].get("availability", "release") == "market_close":
        return add_bdays(obs, lag, sessions_all)
    return release_availability(obs, load_first_releases(cfg, sid), sessions_all, lag,
                                cfg.pit.daily_fallback_lag_bdays)


def _asof_obs(cfg: Config, sid: str, s: pd.Series, sessions_all: pd.DatetimeIndex) -> pd.DataFrame:
    if cfg.universe["fred"][sid].get("availability", "release") == "market_close":
        return pit_from_observations(s, sessions_all, cfg.pit.macro_release_lag_bdays)
    return pit_from_observations(s, sessions_all, cfg.pit.macro_release_lag_bdays,
                                 first_release=load_first_releases(cfg, sid),
                                 fallback_lag=cfg.pit.daily_fallback_lag_bdays)


def _lag_counts(avail: pd.Series, sessions_all: pd.DatetimeIndex) -> dict:
    """How many sessions after its own date each row becomes usable (distribution, for the audit)."""
    a = avail.dropna()
    lag = sessions_all.get_indexer(pd.DatetimeIndex(a.values)) - sessions_all.get_indexer(a.index)
    return {str(k): int(v) for k, v in pd.Series(lag).value_counts().sort_index().items()}


def target_returns_from_raw(cfg: Config, tickers: list[str], sessions_all: pd.DatetimeIndex,
                            sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Log returns of regime-target tickers from the raw snapshot (no fill: NaN before listing)."""
    raw = cfg.data_path / "raw" / "yfinance" / latest_snapshot(cfg)
    close = pd.DataFrame({t: pd.read_parquet(raw / f"{t}.parquet")["close"] for t in tickers})
    return log_returns(close.reindex(sessions_all)).reindex(sessions)


def build_panel(cfg: Config) -> Path:
    snap = latest_snapshot(cfg)
    raw = cfg.data_path / "raw"
    close = pd.DataFrame({t: pd.read_parquet(raw / "yfinance" / snap / f"{t}.parquet")["close"] for t in cfg.etfs})
    end = str(close.index.max().date())          # the snapshot's last market day — never today's date
    sessions_all = nyse_sessions("1990-01-01", end)
    sessions = sessions_all[sessions_all >= pd.Timestamp(cfg.start_date)]
    lag = cfg.pit.macro_release_lag_bdays

    close = close.reindex(sessions_all)  # no fill: a missing bar stays missing
    rets = log_returns(close).reindex(sessions)
    # series that define regime targets (may lie outside the profile's ETFs, e.g. KRE); never used as features
    target_rets = target_returns_from_raw(cfg, sorted({cfg.regime_series, cfg.regime.target_series or
                                                       cfg.regime_series}), sessions_all, sessions)

    graph = rets[cfg.graph_nodes.etfs].copy()
    row_avail = pd.Series(sessions, index=sessions)                    # ETF closes: known at that close
    for sid in cfg.graph_nodes.macro:
        s = pd.read_parquet(raw / "fred" / snap / f"{sid}.parquet")[sid]
        meta = cfg.universe["fred"][sid]
        if meta["revised"]:
            raise ValueError(f"{sid} is revised; graph nodes must be non-revised daily series")
        # Align by observation date on NYSE sessions. FRED has no value on bond-market-only holidays: carry the
        # previous observation forward (trailing fill only, max 3 sessions) so the daily change there is 0.
        lvl = s.reindex(sessions_all).ffill(limit=3)
        graph[f"d_{sid}"] = lvl.diff().reindex(sessions)
        obs = s.dropna()
        av = pd.Series(_availability(cfg, sid, obs.index, sessions_all), index=obs.index)
        # row d needs obs d and d-1 (first difference); a carried-forward row is known no earlier than d + lag
        av = av.reindex(sessions_all).ffill(limit=3)
        av = pd.concat([av, av.shift(1), pd.Series(add_bdays(sessions_all, lag, sessions_all), index=sessions_all)],
                       axis=1).max(axis=1, skipna=False)
        row_avail = pd.concat([row_avail, av.reindex(sessions)], axis=1).max(axis=1, skipna=False)

    controls = {}
    for sid in cfg.fred_series:
        meta = cfg.universe["fred"][sid]
        if meta["revised"]:
            vpath = raw / "alfred" / snap / f"{sid}_vintages.parquet"
            if not vpath.exists():
                raise FileNotFoundError(f"{sid} is revised but no ALFRED vintages at {vpath}; run verify-data")
            pit = pit_from_vintages(pd.read_parquet(vpath), sessions_all, lag)
        else:
            s = pd.read_parquet(raw / "fred" / snap / f"{sid}.parquet")[sid]
            pit = _asof_obs(cfg, sid, s, sessions_all)
        controls[sid] = pit["value"].reindex(sessions)
        controls[f"{sid}__obs_date"] = pit["obs_date"].reindex(sessions)
    controls = pd.DataFrame(controls)

    out = processed_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    rets.to_parquet(out / "returns.parquet")
    target_rets.to_parquet(out / "target_returns.parquet")
    graph.to_parquet(out / "graph_panel.parquet")
    row_avail.rename("available").to_frame().to_parquet(out / "graph_avail.parquet")
    controls.to_parquet(out / "controls_pit.parquet")
    st = stationarity_report(graph)
    st.to_csv(out / "stationarity.csv", index=False)
    write_json(out / "panel_meta.json", {
        "snapshot": snap, "profile": cfg.profile, "sessions": [str(sessions.min().date()), str(sessions.max().date())],
        "graph_nodes": list(graph.columns), "graph_rows_complete": int(graph.dropna().shape[0]),
        "macro_release_lag_bdays": lag, "daily_fallback_lag_bdays": cfg.pit.daily_fallback_lag_bdays,
        "graph_row_avail_lag_counts": _lag_counts(row_avail, sessions_all),
        "non_stationary_cases": st.loc[~st["stationary_both"], ["series", "period"]].to_dict("records"),
    })
    log.info("panel written to %s (%d sessions, %d graph nodes)", out, len(sessions), graph.shape[1])
    return out
