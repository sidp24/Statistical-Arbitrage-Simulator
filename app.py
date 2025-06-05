import streamlit as st
import pandas as pd
import numpy as np
import os

from data.download_data import download_prices
from pairs.find_pairs import find_cointegrated_pairs
from signals.zscore_signal import compute_zscore_spread
from backtest.backtester import PairTradingBacktester
from analysis.metrics import sharpe_ratio, max_drawdown, plot_cumulative_pnl, plot_spread_zscore
from config import *

st.set_page_config(page_title="Stat Arb Simulator", layout="wide")

st.sidebar.title("Configuration")

# --- INPUTS ---
tickers = st.sidebar.multiselect("Select tickers", options=TICKERS, default=["AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM"])
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))
entry_z = st.sidebar.slider("Entry Z-Score", 0.5, 3.0, 1.7)
exit_z = st.sidebar.slider("Exit Z-Score", 0.1, 2.0, 1.0)
zscore_window = st.sidebar.slider("Z-Score Rolling Window", 10, 90, 30)

if st.sidebar.button("Download + Analyze"):
    with st.spinner("Running cointegration and backtest..."):
        # Download prices
        os.makedirs("data", exist_ok=True)
        download_prices(tickers, str(start_date), str(end_date), PRICE_DATA_PATH)
        price_df = pd.read_csv(PRICE_DATA_PATH, index_col=0, parse_dates=True)

        # Find cointegrated pairs
        pairs, _ = find_cointegrated_pairs(price_df, significance=P_VALUE_THRESHOLD)

        if not pairs:
            st.session_state['pairs'] = []
            st.session_state['results'] = {}
            st.error("No cointegrated pairs found.")
        else:
            results = {}
            for t1, t2, pval in pairs:
                s1 = np.log(price_df[t1])
                s2 = np.log(price_df[t2])
                z, spread, hedge, df = compute_zscore_spread(s1, s2, window=zscore_window, ticker1=t1, ticker2=t2)

                temp_path = f"data/signals/{t1}_{t2}_signals.csv"
                df.to_csv(temp_path)

                bt = PairTradingBacktester(temp_path, entry_z, exit_z, INITIAL_CAPITAL, tickers=(t1, t2))
                bt.backtest()
                trades = bt.summary()

                trades["Pair"] = f"{t1} & {t2}"
                results[f"{t1} & {t2}"] = {
                    "trades": trades,
                    "spread_df": df,
                    "pnl_series": trades["PnL"]
                }

            st.session_state['pairs'] = list(results.keys())
            st.session_state['results'] = results

# --- MAIN DISPLAY ---
st.title("Statistical Arbitrage Simulator")

if 'pairs' in st.session_state and st.session_state['pairs']:
    st.success(f"Found {len(st.session_state['pairs'])} cointegrated pairs.")

    selected_pair = st.selectbox("Choose a pair to backtest", st.session_state['pairs'])
    if selected_pair:
        result = st.session_state['results'][selected_pair]
        trades = result['trades']
        pnl_series = result['pnl_series']
        spread_df = result['spread_df']

        col1, col2 = st.columns(2)
        col1.metric("Sharpe Ratio", f"{sharpe_ratio(pnl_series):.2f}")
        col2.metric("Max Drawdown", f"{max_drawdown(pnl_series):.4f}")

        st.markdown("### Cumulative PnL")
        st.pyplot(plot_cumulative_pnl(pnl_series))

        st.markdown("### Spread and Z-Score")
        st.pyplot(plot_spread_zscore(spread_df))

else:
    st.info("Use the sidebar to configure your simulation and click **Download + Analyze**.")

