"""model.py
Training and scoring ML model (Gradient Boosting) with walk-forward logic.

Functions / classes:
- MLRegressor: wrapper around sklearn's HistGradientBoostingRegressor
- make_walkforward_splits(rebal_dates, train_years, test_months)
- generate_monthly_positions(...): performs training and cross-sectional scoring
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


class MLRegressor:
    def __init__(self, seed: int = 42):
        self.seed = seed
        # small model suitable for factor modeling
        self.model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=400, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


def make_walkforward_splits(rebal_dates: pd.DatetimeIndex, train_years: int = 5, test_months: int = 6) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    rebal_dates = pd.to_datetime(pd.Index(rebal_dates)).sort_values()
    splits: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []

    for test_start in rebal_dates:
        train_end = test_start
        train_start = train_end - pd.DateOffset(years=train_years)
        test_end = test_start + pd.DateOffset(months=test_months)

        train_mask = (rebal_dates >= train_start) & (rebal_dates < train_end)
        test_mask = (rebal_dates >= test_start) & (rebal_dates < test_end)

        train_dates = rebal_dates[train_mask]
        test_dates = rebal_dates[test_mask]

        if len(train_dates) < 24 or len(test_dates) < 1:
            continue

        splits.append((train_dates, test_dates))

    return splits


def generate_monthly_positions(
    panel: pd.DataFrame,
    feature_cols: List[str],
    rebal_dates: pd.DatetimeIndex,
    top_q: float = 0.10,
    bot_q: float = 0.10,
    min_universe: int = 50,
    seed: int = 42,
) -> Dict[pd.Timestamp, pd.Series]:
    """Train on rolling windows and produce monthly equal-weighted long/short positions.

    Returns dict date -> Series(weights indexed by ticker).
    """
    usable = panel.dropna(subset=feature_cols + ["y_fwd_3m"]).copy()
    splits = make_walkforward_splits(rebal_dates)
    monthly_positions: Dict[pd.Timestamp, pd.Series] = {}

    model = MLRegressor(seed=seed)

    for train_dates, test_dates in splits:
        # boolean mask for train
        dates_index = usable.index.get_level_values("date")
        train_mask = dates_index.isin(train_dates)
        X_train = usable.loc[train_mask, feature_cols]
        y_train = usable.loc[train_mask, "y_fwd_3m"]

        if len(X_train) < 1000:
            # too small to meaningfully train
            continue

        model.fit(X_train.values, y_train.values)

        # score each test rebal date cross-sectionally
        for d in test_dates:
            X = usable.loc[usable.index.get_level_values("date") == d, feature_cols]
            if X.shape[0] < min_universe:
                continue
            tickers = X.index.get_level_values("ticker")
            preds = model.predict(X.values)
            s = pd.Series(preds, index=tickers).dropna()

            long_cut = s.quantile(1 - top_q)
            short_cut = s.quantile(bot_q)

            longs = s[s >= long_cut].index
            shorts = s[s <= short_cut].index

            w = pd.Series(0.0, index=s.index)
            if len(longs) > 0:
                w.loc[longs] = 1.0 / len(longs)
            if len(shorts) > 0:
                w.loc[shorts] = -1.0 / len(shorts)

            monthly_positions[pd.Timestamp(d)] = w

    return monthly_positions
