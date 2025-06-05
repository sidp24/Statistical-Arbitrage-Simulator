# app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from data.download_data import download_prices
from pairs.find_pairs import find_cointegrated_pairs
from signals.zscore_signal import compute_zscore_spread
from backtest.backtester import PairTradingBacktester
from analysis.metrics import sharpe_ratio, max_drawdown
from config import *

st.set_page_config(page_title="Stat Arb Simulator", layout="wide")
st.title("Statistical Arbitrage Simulator")

# Sidebar controls
st.sidebar.header("Configuration")
tickers = st.sidebar.multiselect("Select tickers", TICKERS, default=TICKERS[:6])
start_date = st.sidebar.date_input("Start Date", pd.to_datetime(START_DATE))
end_date = st.sidebar.date_input("End Date", pd.to_datetime(END_DATE))
entry_z = st.sidebar.slider("Entry Z-Score", 0.5, 3.0, ENTRY_Z, 0.1)
exit_z = st.sidebar.slider("Exit Z-Score", 0.1, 2.0, EXIT_Z, 0.1)
z_window = st.sidebar.slider("Z-Score Rolling Window", 10, 90, ZSCORE_WINDOW, 5)

if st.sidebar.button("Download + Analyze"):
    with st.spinner("Downloading price data and finding pairs..."):
        download_prices(tickers, str(start_date), str(end_date), PRICE_DATA_PATH)
        df = pd.read_csv(PRICE_DATA_PATH, index_col=0, parse_dates=True)
        pairs, _ = find_cointegrated_pairs(df, significance=P_VALUE_THRESHOLD)

    if not pairs:
        st.error("No cointegrated pairs found.")
    else:
        st.success(f"Found {len(pairs)} cointegrated pairs.")
        selected_pair = st.selectbox("Choose a pair to backtest", [f"{a} & {b}" for a, b, _ in pairs])

        t1, t2 = selected_pair.split(" & ")
        s1 = np.log(df[t1])
        s2 = np.log(df[t2])
        z, spread, hedge, signal_df = compute_zscore_spread(s1, s2, window=z_window, ticker1=t1, ticker2=t2)
        temp_path = f"data/signals/{t1}_{t2}_signals.csv"
        os.makedirs("data/signals", exist_ok=True)
        signal_df.to_csv(temp_path)

        bt = PairTradingBacktester(temp_path, entry_z, exit_z, INITIAL_CAPITAL, tickers=(t1, t2))
        bt.backtest()
        trades_df = bt.summary()
        pnl_series = trades_df["PnL"]

        st.subheader("Performance Metrics")
        col1, col2 = st.columns(2)
        col1.metric("Sharpe Ratio", f"{sharpe_ratio(pnl_series):.2f}")
        col2.metric("Max Drawdown", f"{max_drawdown(pnl_series):.4f}")

        st.subheader("Cumulative PnL")
        st.line_chart(pnl_series.cumsum())

        st.subheader("Spread and Z-Score")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(signal_df.index, signal_df["Z-Score"], label="Z-Score")
        ax.axhline(entry_z, color='red', linestyle='--', label="Entry Z")
        ax.axhline(-entry_z, color='red', linestyle='--')
        ax.axhline(exit_z, color='green', linestyle='--', label="Exit Z")
        ax.axhline(-exit_z, color='green', linestyle='--')
        ax.set_title("Z-Score Over Time")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Trade Log")
        st.dataframe(trades_df)

        st.download_button("Download Trade Log CSV", trades_df.to_csv(index=False), f"{t1}_{t2}_trades.csv")
