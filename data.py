"""data.py
Data acquisition and caching utilities.

Functions:
- get_sp500_tickers(): fetch tickers from Wikipedia
- download_ohlcv(...): download OHLCV via yfinance (batching)
- load_or_download(...): caching wrapper (parquet)
- to_long_panel(...): convert yfinance-wide to long panel (date, ticker) index
"""
from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd
import yfinance as yf


import urllib.request


def get_sp500_tickers() -> List[str]:
    """
    Return current S&P 500 tickers from Wikipedia (with User-Agent).
    Avoids 403 Forbidden errors.
    """

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(req) as resp:
        html = resp.read()

    table = pd.read_html(html)[0]

    tickers = table["Symbol"].astype(str).tolist()

    # BRK.B -> BRK-B etc.
    return [t.replace(".", "-") for t in tickers]



def download_ohlcv(tickers: List[str], start: str, end: Optional[str] = None, threads: bool = True) -> pd.DataFrame:
    """Download OHLCV for tickers using yfinance.

    Returns a wide DataFrame as yfinance downloads (MultiIndex columns: ticker, field).
    """
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=threads,
    )
    return data


def to_long_panel(wide: pd.DataFrame) -> pd.DataFrame:
    """Convert yfinance 'wide' output to a long panel with MultiIndex (date, ticker).

    Keeps normalized column names: open, high, low, close, adj_close, volume.
    """
    if not isinstance(wide.columns, pd.MultiIndex):
        # single-ticker download: convert to 2-level columns
        wide.columns = pd.MultiIndex.from_product([ [list(wide.columns.name or "_single_")], wide.columns ])

    # stack tickers -> columns become field names
    long = wide.stack(level=0).rename_axis(index=["date", "ticker"]).reset_index()

    # normalize column names
    cols = {c: c.lower().replace(" ", "_") for c in long.columns if c not in ["date", "ticker"]}
    long = long.rename(columns=cols)

    keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    cols_present = [c for c in keep if c in long.columns]
    long = long[cols_present]

    long = long.sort_values(["ticker", "date"]).set_index(["date", "ticker"])  # MultiIndex
    return long


def load_or_download(cache_path: str, tickers: List[str], start: str, end: Optional[str] = None, force: bool = False) -> pd.DataFrame:
    """Load cached wide data or download and save as parquet.

    cache_path: path to parquet file.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path) and not force:
        try:
            wide = pd.read_parquet(cache_path)
            return wide
        except Exception:
            pass

    wide = download_ohlcv(tickers, start=start, end=end)
    # Save to parquet where possible (pandas handles MultiIndex columns)
    try:
        wide.to_parquet(cache_path)
    except Exception:
        # best-effort: ignore cache errors
        pass
    return wide
