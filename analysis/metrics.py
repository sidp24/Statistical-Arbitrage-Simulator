import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def sharpe_ratio(pnl_series, risk_free_rate=0.0):
    returns = pnl_series.diff().dropna()
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def max_drawdown(pnl_series):
    cumulative = pnl_series.cumsum()
    high_water_mark = cumulative.cummax()
    drawdown = cumulative - high_water_mark
    return drawdown.min()

def plot_cumulative_pnl(pnl_series):
    pnl_series.cumsum().plot(figsize=(10, 5), title="Cumulative PnL")
    plt.xlabel("Trade Number")
    plt.ylabel("Cumulative PnL")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("data/zscore_signals.csv", index_col=0, parse_dates=True)  # optional, if needed
    from backtest.backtester import PairTradingBacktester

    bt = PairTradingBacktester("data/zscore_signals.csv")
    bt.backtest()
    trades_df = bt.summary()

    pnl_series = trades_df["PnL"]

    print(f"Sharpe Ratio: {sharpe_ratio(pnl_series):.3f}")
    print(f"Max Drawdown: {max_drawdown(pnl_series):.3f}")

    plot_cumulative_pnl(pnl_series)
