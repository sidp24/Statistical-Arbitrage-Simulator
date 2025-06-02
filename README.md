# 📊 Statistical Arbitrage Simulator

A modular Python framework for backtesting **cointegration-based statistical arbitrage strategies**, including pairs trading. Built to simulate real-world conditions, evaluate performance, and experiment with systematic trading ideas.

---

## 🚀 Features

- ✅ Pair selection via Engle-Granger cointegration tests
- ✅ Spread modeling using OLS regression
- ✅ Z-score signal generation with rolling statistics
- ✅ Rule-based trading logic (entry/exit thresholds)
- ✅ Backtesting engine with PnL tracking
- ✅ Portfolio metrics: Sharpe ratio, max drawdown, cumulative return
- ✅ Easily extensible to more pairs, signals, and cost models

---

## 🗂 Project Structure
    stat_arb_sim/
    ├── data/ # Historical price data and generated signal files
    │ └── .keep # Empty file to preserve folder in Git
    ├── pairs/ # Cointegration testing (Engle-Granger)
    │ └── find_pairs.py
    ├── signals/ # Spread + z-score calculation
    │ └── zscore_signal.py
    ├── backtest/ # Trade simulation logic
    │ └── backtester.py
    ├── analysis/ # Performance metrics and plots
    │ └── metrics.py
    ├── config.py # Strategy parameters (tickers, dates, thresholds)
    ├── main.py # End-to-end pipeline
    ├── requirements.txt
    └── .gitignore
---

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/stat-arb-simulator.git
cd stat-arb-simulator
pip install -r requirements.txt
🧪 Usage
Run the full backtesting pipeline:
python main.py

This will:

    Download historical prices via Yahoo Finance

    Find cointegrated pairs

    Generate z-score trading signals

    Simulate trades

    Output performance stats + cumulative PnL plot


Planned Improvements:

    Volatility-based position sizing

    Transaction cost and slippage modeling

    Parameter grid search for entry/exit thresholds

    Live dashboard using Streamlit or Jupyter

    Sector/industry filtering and cointegration clustering

📄 License
MIT © 2024 Siddharth Paul

---
Let me know if you'd like:
- A professional project **tagline/summary for GitHub**
- Help writing issues or milestones to track features
- To add badges (e.g., Python version, license) to the README
---
