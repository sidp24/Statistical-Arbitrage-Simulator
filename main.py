import pandas as pd
import numpy as np
import os

from data.download_data import download_prices
from pairs.find_pairs import find_cointegrated_pairs
from signals.zscore_signal import compute_zscore_spread
from backtest.backtester import PairTradingBacktester
from analysis.metrics import sharpe_ratio, max_drawdown, plot_cumulative_pnl
from config import *

def main():
    print("Downloading price data...")
    download_prices(TICKERS, START_DATE, END_DATE, PRICE_DATA_PATH)

    print("\nFinding cointegrated pairs...")
    price_df = pd.read_csv(PRICE_DATA_PATH, index_col=0, parse_dates=True)
    pairs, _ = find_cointegrated_pairs(price_df, significance=P_VALUE_THRESHOLD)

    if not pairs:
        print("No cointegrated pairs found.")
        print("\n" + "="*60)
        print("[INFO] WHY THIS HAPPENED & HOW TO FIX IT")
        print("="*60)
        print("""
[ANALYSIS] Common Reasons:
   1. Tickers not in the same sector (random stocks rarely cointegrate)
   2. Time period too short (need 1-2+ years of data)
   3. Market regime change broke historical relationships

[RECOMMENDED] Ticker Combinations:
   - Tech:          MSFT, GOOGL, AAPL, META
   - Banks:         JPM, BAC, GS, C
   - Oil & Gas:     XOM, CVX, COP
   - Airlines:      DAL, UAL, AAL
   - Retail:        WMT, TGT, COST
   - Semiconductors: AMD, NVDA, INTC

[ACTION] Quick Fix:
   1. Edit config.py and change TICKERS to stocks in the SAME sector
   2. Use at least 2 years of data (START_DATE to END_DATE)
   3. Example: TICKERS = ['JPM', 'BAC', 'GS', 'C']
""")
        print("="*60)
        return

    all_trades = []
    os.makedirs("data/signals", exist_ok=True)

    for pair in pairs:
        t1, t2, pval = pair
        print(f"\nPair: {t1} & {t2} (p={pval:.4f})")

        s1 = np.log(price_df[t1])
        s2 = np.log(price_df[t2])
        z, spread, hedge, df = compute_zscore_spread(s1, s2, window=ZSCORE_WINDOW, ticker1=t1, ticker2=t2)

        temp_path = f"data/signals/{t1}_{t2}_signals.csv"
        df.to_csv(temp_path)

        bt = PairTradingBacktester(temp_path, ENTRY_Z, EXIT_Z, INITIAL_CAPITAL, tickers=(t1, t2))
        bt.backtest()
        trades = bt.summary()

        trades["Pair"] = f"{t1}-{t2}"
        all_trades.append(trades)

    print("\nAggregating performance across all pairs...")
    all_trades_df = pd.concat(all_trades, ignore_index=True)
    all_trades_df.to_csv("data/all_trades.csv", index=False)

    pnl_series = all_trades_df["PnL"]
    print(f"\nTotal Pairs Traded: {len(pairs)}")
    print(f"Total Trades: {len(all_trades_df)}")
    print(f"Max Drawdown: {max_drawdown(pnl_series):.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio(pnl_series):.4f}")

    plot_cumulative_pnl(pnl_series)

if __name__ == "__main__":
    main()
