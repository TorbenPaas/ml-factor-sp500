"""run.py
Entry point to build features, train ML factor, backtest and report.

Usage examples in README.md
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from data import get_sp500_tickers, load_or_download, to_long_panel
from features import add_features, add_target
from model import generate_monthly_positions
from backtest import month_ends, backtest_positions
from report import plot_equity_curve, plot_drawdown


def _perf_stats(r: pd.Series, freq: int = 252) -> dict:
    """Simple performance stats helper for a daily return series."""
    r = r.dropna()
    if r.empty:
        return {"CAGR": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}

    eq = (1.0 + r).cumprod()
    cagr = float(eq.iloc[-1] ** (freq / len(r)) - 1.0)

    vol = float(r.std()) if float(r.std()) != 0.0 else np.nan
    sharpe = float((r.mean() / (r.std() + 1e-12)) * (freq ** 0.5))

    peak = eq.cummax()
    mdd = float((eq / peak - 1.0).min())

    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": mdd}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ML Factor backtest for S&P500")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--cache", default="./data/cache.parquet")
    parser.add_argument("--tc_bps", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=0, help="limit number of tickers for quick tests")
    args = parser.parse_args(argv)

    np.random.seed(42)

    tickers = get_sp500_tickers()
    if args.limit and args.limit > 0:
        tickers = tickers[: args.limit]

    print(f"Downloading {len(tickers)} tickers from {args.start} to {args.end}")
    wide = load_or_download(args.cache, tickers, start=args.start, end=args.end)

    panel = to_long_panel(wide)
    panel = add_features(panel)
    panel = add_target(panel, horizon=63)

    # determine rebal dates
    rebal_dates = month_ends(panel.index.get_level_values("date"))
    feature_cols = [c for c in panel.columns if c != "y_fwd_3m"]

    print("Generating monthly positions (walk-forward)...")
    monthly_positions = generate_monthly_positions(panel, feature_cols, rebal_dates, top_q=0.10, bot_q=0.10)
    print(f"Generated {len(monthly_positions)} monthly position sets")

    print("Running backtest...")
    port_ret, stats = backtest_positions(panel, monthly_positions, tc_bps=args.tc_bps)

    print("Strategy stats:")
    for k, v in stats.items():
        # stats sollten floats sein, aber wir casten zur Sicherheit
        try:
            print(f"{k}: {float(v):.4f}")
        except Exception:
            print(f"{k}: {v}")

    # ===============================
    # Benchmark: SPY (robust version)
    # ===============================
    spy = yf.download("SPY", start=args.start, end=args.end, progress=False)

    price_col = "Adj Close" if "Adj Close" in spy.columns else "Close"
    spy_price = spy[price_col]

    # Falls spy_price ein DataFrame ist (selten, aber möglich), nimm erste Spalte
    if isinstance(spy_price, pd.DataFrame):
        spy_price = spy_price.iloc[:, 0]

    spy_ret = spy_price.pct_change().reindex(port_ret.index).dropna()

    print("\nSPY Buy&Hold stats (aligned):")
    for k, v in _perf_stats(spy_ret).items():
        if np.isnan(v):
            print(f"{k}: nan")
        else:
            print(f"{k}: {float(v):.4f}")

    print("Plotting...")
    os.makedirs("outputs", exist_ok=True)
    plot_equity_curve(port_ret, title="ML Factor Equity", save_path="outputs/equity.png")
    plot_drawdown(port_ret, title="ML Factor Drawdown", save_path="outputs/drawdown.png")
    print("Saved plots to outputs/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
