# 📊 Statistical Arbitrage Simulator

A modular Python framework for backtesting **cointegration-based statistical arbitrage strategies**, including pairs trading. Built to simulate real-world conditions, evaluate performance, and experiment with systematic trading ideas.

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/stat-arb-simulator.git
cd stat-arb-simulator
pip install -r requirements.txt

Usage:
Run the full backtesting pipeline:
python main.py

This will:

    Download historical prices via Yahoo Finance

    Find co-integrated pairs

    Generate z-score trading signals

    Simulate trades

    Output performance stats + cumulative PnL plot


Planned Improvements:

    Volatility-based position sizing

    Transaction cost and slippage modeling

    Parameter grid search for entry/exit thresholds

    Live dashboard using Streamlit or Jupyter

    Sector/industry filtering and cointegration clustering

License
MIT © 20245Siddharth Paul
