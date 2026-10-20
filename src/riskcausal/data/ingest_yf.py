"""Yahoo Finance daily bars via yfinance (free, no key). Prices are split/dividend adjusted (auto_adjust)."""

from __future__ import annotations

import logging
import time

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


def download_etf(ticker: str, start: str, end: str | None = None, retries: int = 3) -> pd.DataFrame:
    """Adjusted OHLCV for one ticker; raises on persistent failure (never returns fabricated data)."""
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False,
                             actions=False, threads=False)
            if df is None or df.empty:
                raise RuntimeError(f"yfinance returned no rows for {ticker}")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
            df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
            df.index.name = "date"
            return df
        except Exception as e:  # noqa: BLE001 — network library raises many types
            last_err = e
            log.warning("yfinance %s attempt %d/%d failed: %s", ticker, attempt, retries, e)
            time.sleep(2 * attempt)
    raise RuntimeError(f"yfinance download failed for {ticker}: {last_err}")
