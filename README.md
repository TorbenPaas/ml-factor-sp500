# ML Factor Investing – S&P 500 Backtest

This project implements a machine-learning-based factor investing strategy
for stocks in the S&P 500 universe. It downloads historical market data,
computes quantitative features, trains a predictive model, and evaluates
the resulting portfolio using a walk-forward backtest.

The goal is to analyze whether machine learning can generate statistically
and economically meaningful alpha compared to a passive benchmark (SPY).

---

## 📌 Features

- Automatic download of S&P 500 constituents
- Historical OHLCV data via Yahoo Finance
- Feature engineering (momentum, volatility, moving averages, drawdowns, volume)
- Forward return prediction (3-month horizon)
- Walk-forward training and rebalancing
- Long/short portfolio construction
- Transaction cost modeling
- Performance evaluation (CAGR, Sharpe, Max Drawdown)
- Comparison with SPY benchmark
- Equity curve and drawdown plots

---

## 📊 Methodology

1. **Universe Selection**
   - Current S&P 500 constituents from Wikipedia  
   - Note: introduces survivorship bias

2. **Data Collection**
   - Daily OHLCV data from Yahoo Finance (`yfinance`)

3. **Feature Engineering**
   - Momentum indicators
   - Volatility measures
   - Moving averages
   - Volume-based features
   - Drawdown metrics

4. **Target Variable**
   - Forward 63-day (≈ 3 months) returns

5. **Model**
   - Gradient Boosting Regressor
   - Walk-forward training to avoid look-ahead bias

6. **Portfolio Construction**
   - Monthly rebalancing
   - Long top 10%, short bottom 10% predictions
   - Equal-weighted positions

7. **Backtesting**
   - Transaction costs in basis points
   - Performance statistics
   - Benchmark comparison (SPY)

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/ml-factor-sp500.git
cd ml-factor-sp500
