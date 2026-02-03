"""report.py
Plotting and reporting utilities: equity curve and drawdown plots.
"""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(returns: pd.Series, title: Optional[str] = None, save_path: Optional[str] = None) -> None:
    eq = (1 + returns).cumprod()
    plt.figure(figsize=(10, 5))
    plt.plot(eq.index, eq.values, label="Equity")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    if title:
        plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def plot_drawdown(returns: pd.Series, title: Optional[str] = None, save_path: Optional[str] = None) -> None:
    eq = (1 + returns).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    plt.figure(figsize=(10, 4))
    plt.plot(dd.index, dd.values, color="red", label="Drawdown")
    plt.fill_between(dd.index, dd.values, 0, color="red", alpha=0.2)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    if title:
        plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
