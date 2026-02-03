"""features.py
Feature engineering from OHLCV panel.

Functions:
- add_features(panel): computes momentum, vol, px_vs_ma200, ret_5d, dd_63, log_dvol_21
- add_target(panel, horizon): forward return target y

Assumes input panel index = (date, ticker) and column 'close' present.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with added features.

    Input df: MultiIndex (date,ticker) -> columns include 'close' and optionally 'volume'.
    """
    df = df.copy()

    # daily return
    df["ret_1d"] = df.groupby(level="ticker")["close"].pct_change()

    # momentum windows (simple pct change over window)
    for w in [21, 63, 126, 252]:
        df[f"mom_{w}"] = df.groupby(level="ticker")["close"].pct_change(w)

    # realized volatility (std of daily returns)
    for w in [21, 63]:
        df[f"vol_{w}"] = (
            df.groupby(level="ticker")["ret_1d"].rolling(w).std().reset_index(level=0, drop=True)
        )

    # price vs MA200
    ma200 = df.groupby(level="ticker")["close"].rolling(200).mean().reset_index(level=0, drop=True)
    df["px_vs_ma200"] = df["close"] / ma200 - 1.0

    # short-term reversal
    df["ret_5d"] = df.groupby(level="ticker")["close"].pct_change(5)

    # rolling max drawdown over 63 days
    roll_max = df.groupby(level="ticker")["close"].rolling(63).max().reset_index(level=0, drop=True)
    df["dd_63"] = df["close"] / roll_max - 1.0

    # log dollar volume 21d
    if "volume" in df.columns:
        dollar_vol = df["close"] * df["volume"]
        dv21 = dollar_vol.groupby(level="ticker").rolling(21).mean().reset_index(level=0, drop=True)
        df["log_dvol_21"] = np.log1p(dv21)

    return df


def add_target(df: pd.DataFrame, horizon: int = 63) -> pd.DataFrame:
    """Add forward target y_fwd (close(t+h)/close(t) - 1) without look-ahead leakage.

    Shifts are done per ticker.
    """
    df = df.copy()
    future_close = df.groupby(level="ticker")["close"].shift(-horizon)
    df["y_fwd_3m"] = future_close / df["close"] - 1.0
    return df
