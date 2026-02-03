# ML Factor Investing – S&P 500 Backtest

This project implements a machine-learning-based factor investing strategy
for stocks in the S&P 500 universe. It provides a complete research pipeline
for quantitative equity analysis, including data acquisition, feature
engineering, predictive modeling, portfolio construction, backtesting, and
performance evaluation.

The objective of this project is to analyze whether machine learning models
can generate risk-adjusted excess returns compared to a passive benchmark
investment in the S&P 500 (SPY).

---

## Overview

The project follows a systematic quantitative research workflow:

- Retrieval of S&P 500 constituents
- Download of historical market data
- Feature engineering
- Machine learning model training
- Walk-forward evaluation
- Portfolio construction
- Backtesting and benchmarking
- Visualization and reporting

It is designed for educational and research purposes in quantitative finance
and machine learning.

---

## Data Sources

- Stock universe: Current S&P 500 constituents (Wikipedia)
- Price data: Yahoo Finance via yfinance
- Frequency: Daily OHLCV data

Note: The use of current index constituents introduces survivorship bias.

---

## Feature Engineering

The following quantitative features are computed:

- Momentum (21d, 63d, 126d, 252d returns)
- Volatility (rolling standard deviation)
- Moving average distance (MA200)
- Short-term reversal (5-day return)
- Maximum drawdown (63-day window)
- Volume-based liquidity measures

All features are computed using only historical information to avoid
look-ahead bias.

---

## Target Variable

The prediction target is the forward 63-day return:

y(t) = Close(t+63) / Close(t) - 1

This corresponds to an approximate three-month investment horizon.

---

## Machine Learning Model

- Model type: Gradient Boosting Regressor
- Library: scikit-learn
- Training method: Walk-forward validation
- Training window: Rolling historical window
- Testing window: Out-of-sample evaluation

The walk-forward setup reduces overfitting and improves realism.

---

## Portfolio Construction

The portfolio is constructed on a monthly basis:

- Rebalancing frequency: Monthly
- Long positions: Top 10 percent of predicted returns
- Short positions: Bottom 10 percent of predicted returns
- Weighting: Equal-weighted
- Transaction costs: Modeled in basis points

This results in a market-neutral long-short factor portfolio.

---

## Backtesting and Evaluation

The backtesting engine computes daily portfolio returns and evaluates
performance using standard metrics:

- Compound Annual Growth Rate (CAGR)
- Sharpe Ratio
- Maximum Drawdown
- Portfolio Turnover

The strategy is benchmarked against a buy-and-hold investment in SPY
with aligned return periods.

Equity curves and drawdown charts are generated automatically.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/<your-username>/ml-factor-sp500.git
cd ml-factor-sp500
