## Projektidee

Ich habe untersucht, ob man mit ML-Faktoren den S&P 500 schlagen kann.
Dazu habe ich Kursdaten verwendet und Momentum-/Volatilitätsfaktoren berechnet.

ML Factor (S&P500)

This project implements a simple ML factor that predicts 3-month forward returns
using only price-derived features, then backtests an equal-weighted long-short
portfolio with monthly rebalancing.

Files
- data.py: data download and caching (yfinance + Wikipedia tickers)
- features.py: feature engineering and forward target
- model.py: walk-forward training and cross-sectional scoring using gradient boosting
- backtest.py: daily portfolio construction, transaction costs, performance stats
- report.py: plotting (equity + drawdown)
- run.py: entry point that orchestrates everything
- requirements.txt: python deps

Quick start
1. (Optional) Create virtualenv and install requirements:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run with full universe (may take a while and download a lot):

```bash
python3 run.py --start 2010-01-01 --end 2026-01-31
```

3. For fast local tests, limit tickers:

```bash
python3 run.py --limit 100
```

Notes
- The project uses `HistGradientBoostingRegressor` from scikit-learn by default.
- Data is cached to `./data/cache.parquet` after first download.
- Running the full S&P500 download may be rate-limited by the data provider.
- This is a research-quality backtest, not production trading code.
