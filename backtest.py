"""backtest.py
Portfolio construction and backtest utilities.

Functions:
- month_ends(dates)
- backtest_positions(panel, monthly_positions, tc_bps)
- compute_perf_stats(returns, freq=252)

Outputs daily portfolio returns and diagnostics (turnover, equity, drawdown).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def month_ends(dates: pd.Index) -> pd.DatetimeIndex:
    s = pd.Series(pd.to_datetime(dates).unique()).sort_values()
    return s.groupby([s.dt.year, s.dt.month]).max().values


def compute_perf_stats(returns: pd.Series, freq: int = 252) -> Dict[str, float]:
    r = returns.dropna()
    if r.empty:
        return {"CAGR": 0.0, "Sharpe": 0.0, "MaxDD": 0.0}

    equity = (1 + r).cumprod()
    years = len(r) / float(freq) if freq > 0 else 0.0
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else 0.0
    sharpe = (r.mean() / (r.std() + 1e-12)) * np.sqrt(freq)

    peak = equity.cummax()
    dd = equity / peak - 1
    mdd = dd.min()

    return {"CAGR": float(cagr), "Sharpe": float(sharpe), "MaxDD": float(mdd)}


def backtest_positions(panel: pd.DataFrame, monthly_positions: Dict[pd.Timestamp, pd.Series], tc_bps: float = 5.0) -> Tuple[pd.Series, Dict[str, float]]:
    """Convert monthly_weights -> daily returns, accounting for simple transaction costs.

    - panel: must include 'close' column; index (date,ticker)
    - monthly_positions: dict mapping rebalance date -> Series(weights indexed by ticker)

    Returns (port_daily_returns, diagnostics)
    """
    if not monthly_positions:
        raise RuntimeError("No positions provided for backtest")

    # price panel: wide
    px = panel["close"].unstack("ticker").sort_index()
    daily_ret = px.pct_change()

    pos_df = pd.DataFrame(monthly_positions).T.sort_index()
    pos_df = pos_df.reindex(columns=daily_ret.columns).fillna(0.0)

    # forward-fill to daily index (apply from rebalance date forward)
    daily_w = pos_df.reindex(daily_ret.index, method="ffill").fillna(0.0)

    # turnover on rebalance days (sum abs changes)/2
    w_change = daily_w.diff().fillna(0.0)
    turnover = (w_change.abs().sum(axis=1) / 2.0)
    tc = turnover * (tc_bps / 10000.0)

    # apply yesterday's weights to today's returns (weights known at open)
    port_ret = (daily_w.shift(1).fillna(0.0) * daily_ret).sum(axis=1) - tc

    stats = compute_perf_stats(port_ret)
    # add turnover summary
    avg_turnover = turnover.mean()
    stats.update({"AvgTurnover": float(avg_turnover)})

    return port_ret.fillna(0.0), stats
