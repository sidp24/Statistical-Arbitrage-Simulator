import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Try to import data functions with fallback
try:
    from data.download_data import download_prices, create_sample_data
    DATA_DOWNLOAD_AVAILABLE = True
except ImportError as e:
    st.error(f"Data download module unavailable: {e}")
    DATA_DOWNLOAD_AVAILABLE = False
    
    # Create a dummy function
    def download_prices(*args, **kwargs):
        raise ImportError("yfinance not available")
    
    def create_sample_data(*args, **kwargs):
        raise ImportError("Sample data creation not available")

from pairs.find_pairs import find_cointegrated_pairs
from signals.zscore_signal import compute_zscore_spread
from backtest.backtester import PairTradingBacktester
from backtest.enhanced_backtester import EnhancedPairTradingBacktester
from analysis.metrics import sharpe_ratio, max_drawdown, plot_cumulative_pnl, plot_spread_zscore
from config import *

st.set_page_config(page_title="Statistical Arbitrage Simulator", layout="wide")

# Main title
st.title("Statistical Arbitrage Simulator")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose Function", ["Backtesting", "Testing & Validation", "Configuration"])

if page == "Configuration":
    st.header("System Configuration")
    
    st.subheader("Dependencies")
    if DATA_DOWNLOAD_AVAILABLE:
        st.success("Data Download: yfinance available - real market data can be downloaded")
    else:
        st.warning("Data Download: yfinance not available - will use sample data")
        st.info("To install yfinance for real market data: `pip install yfinance`")
    
    st.subheader("Data Settings")
    st.write(f"Price data path: {PRICE_DATA_PATH}")
    st.write(f"Signals path: data/signals/")
    st.write(f"Default tickers: {TICKERS}")
    
    st.subheader("Trading Parameters")
    st.write(f"Initial capital: ${INITIAL_CAPITAL:,}")
    st.write(f"P-value threshold: {P_VALUE_THRESHOLD}")
    
    st.subheader("Enhanced Features Available")
    try:
        from trading.transaction_costs import TransactionCostModel
        st.success("Transaction Cost Modeling: Available")
    except ImportError:
        st.warning("Transaction Cost Modeling: Not Available")
    
    try:
        from risk.risk_management import RiskManager
        st.success("Risk Management: Available")
    except ImportError:
        st.warning("Risk Management: Not Available")
    
    try:
        from optimization.parameter_optimization import WalkForwardOptimizer
        st.success("Parameter Optimization: Available")
    except ImportError:
        st.warning("Parameter Optimization: Not Available")

elif page == "Testing & Validation":
    st.header("Testing & Validation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Quick System Test")
        if st.button("Run System Test"):
            with st.spinner("Testing system components..."):
                test_results = {}
                
                # Test basic imports
                try:
                    import pandas as pd
                    import numpy as np
                    test_results["Basic Libraries"] = "OK"
                except Exception as e:
                    test_results["Basic Libraries"] = f"ERROR: {e}"
                
                # Test enhanced backtester
                try:
                    from backtest.enhanced_backtester import EnhancedPairTradingBacktester
                    test_results["Enhanced Backtester"] = "OK"
                except Exception as e:
                    test_results["Enhanced Backtester"] = f"ERROR: {e}"
                
                # Test enhanced modules
                try:
                    from trading.transaction_costs import TransactionCostModel
                    test_results["Transaction Costs"] = "OK"
                except Exception as e:
                    test_results["Transaction Costs"] = f"Not Available: {e}"
                
                try:
                    from risk.risk_management import RiskManager
                    test_results["Risk Management"] = "OK"
                except Exception as e:
                    test_results["Risk Management"] = f"Not Available: {e}"
                
                try:
                    from optimization.parameter_optimization import WalkForwardOptimizer
                    test_results["Parameter Optimization"] = "OK"
                except Exception as e:
                    test_results["Parameter Optimization"] = f"Not Available: {e}"
                
                st.subheader("Test Results:")
                for component, status in test_results.items():
                    if "OK" in status:
                        st.success(f"{component}: {status}")
                    elif "ERROR" in status:
                        st.error(f"{component}: {status}")
                    else:
                        st.warning(f"{component}: {status}")
    
    with col2:
        st.subheader("Demo Backtest")
        if st.button("Run Demo"):
            with st.spinner("Running demonstration backtest..."):
                # Create demo data
                dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')
                np.random.seed(42)
                
                # Generate synthetic pair data
                price_changes = np.random.normal(0, 0.015, len(dates))
                spread = np.cumsum(price_changes)
                spread = spread - spread.rolling(window=20, min_periods=1).mean()
                
                zscore_window = 15
                spread_mean = spread.rolling(window=zscore_window).mean()
                spread_std = spread.rolling(window=zscore_window).std()
                zscore = (spread - spread_mean) / spread_std
                
                demo_data = pd.DataFrame({
                    'spread': spread,
                    'zscore': zscore.fillna(0)
                }, index=dates)
                
                # Run basic vs enhanced comparison
                try:
                    # Basic backtest
                    backtester_basic = EnhancedPairTradingBacktester(
                        signal_data=demo_data,
                        entry_z=1.5,
                        exit_z=0.5,
                        initial_capital=100000,
                        enable_costs=False,
                        enable_risk_management=False
                    )
                    results_basic = backtester_basic.backtest()
                    
                    # Enhanced backtest
                    backtester_enhanced = EnhancedPairTradingBacktester(
                        signal_data=demo_data,
                        entry_z=1.5,
                        exit_z=0.5,
                        initial_capital=100000,
                        tickers=('DEMO_A', 'DEMO_B'),
                        enable_costs=True,
                        enable_risk_management=True
                    )
                    results_enhanced = backtester_enhanced.backtest()
                    
                    st.subheader("Demo Results Comparison:")
                    
                    col_basic, col_enhanced = st.columns(2)
                    
                    with col_basic:
                        st.write("**Basic Backtest:**")
                        st.write(f"Total Return: {results_basic['total_return']:.2%}")
                        st.write(f"Number of Trades: {len(results_basic['trades'])}")
                    
                    with col_enhanced:
                        st.write("**Enhanced Backtest:**")
                        st.write(f"Total Return: {results_enhanced['total_return']:.2%}")
                        st.write(f"Number of Trades: {len(results_enhanced['trades'])}")
                        if 'performance_metrics' in results_enhanced and results_enhanced['performance_metrics']:
                            if 'total_costs' in results_enhanced['performance_metrics']:
                                st.write(f"Total Costs: ${results_enhanced['performance_metrics']['total_costs']:.2f}")
                    
                    st.success("Demo completed successfully!")
                    
                except Exception as e:
                    st.error(f"Demo failed: {e}")

else:  # Backtesting page
    st.sidebar.title("Backtesting Configuration")
    
    # Enhanced vs Basic toggle
    use_enhanced = st.sidebar.checkbox("Use Enhanced Features", value=True, help="Enable transaction costs and risk management")
    
    # --- INPUTS ---
    tickers = st.sidebar.multiselect("Select tickers", options=TICKERS, default=["AAPL", "MSFT", "GOOGL", "AMZN"])
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))
    entry_z = st.sidebar.slider("Entry Z-Score", 0.5, 3.0, 1.7)
    exit_z = st.sidebar.slider("Exit Z-Score", 0.1, 2.0, 1.0)
    zscore_window = st.sidebar.slider("Z-Score Rolling Window", 10, 90, 30)
    
    # Enhanced features configuration
    if use_enhanced:
        st.sidebar.subheader("Enhanced Features")
        enable_costs = st.sidebar.checkbox("Enable Transaction Costs", value=True)
        enable_risk_mgmt = st.sidebar.checkbox("Enable Risk Management", value=True)
        initial_capital = st.sidebar.number_input("Initial Capital", value=100000, min_value=10000, step=10000)

    if st.sidebar.button("Run Backtest"):
        with st.spinner("Running analysis..."):
            # Download or create sample prices
            os.makedirs("data", exist_ok=True)
            
            try:
                if DATA_DOWNLOAD_AVAILABLE:
                    download_prices(tickers, str(start_date), str(end_date), PRICE_DATA_PATH)
                else:
                    st.warning("yfinance not available. Creating sample data for demonstration.")
                    create_sample_data(tickers, str(start_date), str(end_date), PRICE_DATA_PATH)
                
                price_df = pd.read_csv(PRICE_DATA_PATH, index_col=0, parse_dates=True)
                
            except Exception as e:
                st.error(f"Error downloading/creating data: {e}")
                st.info("To use real market data, please install yfinance: pip install yfinance")
                
                # Create minimal sample data as fallback
                st.info("Creating minimal sample data for demonstration...")
                sample_dates = pd.date_range(start=start_date, end=end_date, freq='D')
                np.random.seed(42)
                
                sample_data = {}
                for ticker in tickers[:4]:  # Limit to 4 tickers for demo
                    prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(sample_dates))))
                    sample_data[ticker] = prices
                
                price_df = pd.DataFrame(sample_data, index=sample_dates)
                price_df.to_csv(PRICE_DATA_PATH)
            
            # Find cointegrated pairs
            pairs, _ = find_cointegrated_pairs(price_df, significance=P_VALUE_THRESHOLD)

            if not pairs:
                st.session_state['pairs'] = []
                st.session_state['results'] = {}
                st.error("No cointegrated pairs found with the current parameters.")
            else:
                results = {}
                for t1, t2, pval in pairs:
                    s1 = np.log(price_df[t1])
                    s2 = np.log(price_df[t2])
                    z, spread, hedge, df = compute_zscore_spread(s1, s2, window=zscore_window, ticker1=t1, ticker2=t2)

                    temp_path = f"data/signals/{t1}_{t2}_signals.csv"
                    os.makedirs("data/signals", exist_ok=True)
                    df.to_csv(temp_path)

                    if use_enhanced:
                        # Use enhanced backtester
                        bt = EnhancedPairTradingBacktester(
                            path_to_signals=temp_path,
                            entry_z=entry_z,
                            exit_z=exit_z,
                            initial_capital=initial_capital,
                            tickers=(t1, t2),
                            enable_costs=enable_costs,
                            enable_risk_management=enable_risk_mgmt
                        )
                        result = bt.backtest()
                        
                        # Extract trade data for display
                        if 'trades' in result and len(result['trades']) > 0:
                            trades_df = pd.DataFrame(result['trades'])
                            trades_df["Pair"] = f"{t1} & {t2}"
                            
                            results[f"{t1} & {t2}"] = {
                                "trades": trades_df,
                                "spread_df": df,
                                "pnl_series": trades_df.get("pnl", []),
                                "result": result
                            }
                    else:
                        # Use basic backtester
                        bt = PairTradingBacktester(temp_path, entry_z, exit_z, initial_capital, tickers=(t1, t2))
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
                st.session_state['use_enhanced'] = use_enhanced

    # --- MAIN DISPLAY ---
    if 'pairs' in st.session_state and st.session_state['pairs']:
        st.success(f"Found {len(st.session_state['pairs'])} cointegrated pairs.")

        selected_pair = st.selectbox("Choose a pair to analyze", st.session_state['pairs'])
        if selected_pair:
            result = st.session_state['results'][selected_pair]
            trades = result['trades']
            pnl_series = result['pnl_series']
            spread_df = result['spread_df']
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            if len(pnl_series) > 0:
                col1.metric("Sharpe Ratio", f"{sharpe_ratio(pnl_series):.3f}")
                col2.metric("Max Drawdown", f"{max_drawdown(pnl_series):.3f}")
                col3.metric("Total Trades", len(trades))
            
            # Enhanced results display
            if st.session_state.get('use_enhanced', False) and 'result' in result:
                enhanced_result = result['result']
                
                if 'performance_metrics' in enhanced_result and enhanced_result['performance_metrics']:
                    st.subheader("Enhanced Metrics")
                    metrics = enhanced_result['performance_metrics']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Return", f"{enhanced_result.get('total_return', 0):.2%}")
                    col2.metric("Win Rate", f"{metrics.get('win_rate', 0):.1%}")
                    
                    if 'total_costs' in metrics:
                        col3.metric("Total Costs", f"${metrics['total_costs']:.2f}")
                    if 'sharpe_ratio' in metrics:
                        col4.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")

            # Charts
            st.subheader("Cumulative PnL")
            if len(pnl_series) > 0:
                st.pyplot(plot_cumulative_pnl(pnl_series))
            else:
                st.warning("No trades generated for this pair.")

            st.subheader("Spread and Z-Score")
            st.pyplot(plot_spread_zscore(spread_df))
            
            # Trade details
            st.subheader("Trade Details")
            st.dataframe(trades)

    else:
        st.info("Configure your parameters in the sidebar and click 'Run Backtest' to begin.")

