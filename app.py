"""
Statistical Arbitrage Simulator - Main Application

Web application for backtesting cointegration-based statistical arbitrage strategies.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Load configuration first
from config import *

# Try to import data functions with fallback
try:
    from data.download_data import download_prices, create_sample_data
    DATA_DOWNLOAD_AVAILABLE = True
except ImportError as e:
    DATA_DOWNLOAD_AVAILABLE = False
    
    def download_prices(*args, **kwargs):
        raise ImportError("yfinance not available")
    
    def create_sample_data(*args, **kwargs):
        raise ImportError("Sample data creation not available")

from pairs.find_pairs import find_cointegrated_pairs
from signals.zscore_signal import compute_zscore_spread
from backtest.backtester import PairTradingBacktester
from backtest.enhanced_backtester import EnhancedPairTradingBacktester
from analysis.metrics import sharpe_ratio, max_drawdown

# Try to import enhanced features
try:
    from trading.paper_trading import PaperTradingEngine, paper_trading_engine
    PAPER_TRADING_AVAILABLE = True
except ImportError:
    PAPER_TRADING_AVAILABLE = False

try:
    from database.models import init_db, get_db, BacktestRepository
    DATABASE_AVAILABLE = True
    init_db()
except ImportError:
    DATABASE_AVAILABLE = False

try:
    from auth.authentication import init_session_state, show_login_form, logout, auth_service
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

try:
    from notifications.alerts import alert_manager, email_notifier
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Statistical Arbitrage Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f1f1f;
        margin-bottom: 1rem;
    }
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .main-header {
            color: #ffffff;
        }
    }
    /* Streamlit dark mode */
    [data-testid="stAppViewContainer"][data-theme="dark"] .main-header {
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def plot_cumulative_pnl_plotly(pnl_series):
    """Create interactive cumulative PnL chart using Plotly."""
    cumulative = pnl_series.cumsum()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=cumulative,
        mode='lines',
        name='Cumulative PnL',
        line=dict(color='#667eea', width=2),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))
    
    fig.update_layout(
        title="Cumulative P&L",
        xaxis_title="Trade Number",
        yaxis_title="Cumulative PnL ($)",
        template="plotly_white",
        height=400,
        hovermode="x unified"
    )
    
    return fig


def plot_spread_zscore_plotly(df):
    """Create interactive spread and z-score chart using Plotly."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.1,
                       subplot_titles=('Spread', 'Z-Score'))
    
    # Spread
    fig.add_trace(go.Scatter(
        x=df.index, y=df['spread'],
        mode='lines', name='Spread',
        line=dict(color='#667eea', width=1.5)
    ), row=1, col=1)
    
    # Z-Score
    fig.add_trace(go.Scatter(
        x=df.index, y=df['zscore'],
        mode='lines', name='Z-Score',
        line=dict(color='#764ba2', width=1.5)
    ), row=2, col=1)
    
    # Add threshold lines
    if 'zscore' in df.columns:
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        fig.add_hline(y=1.5, line_dash="dot", line_color="red", row=2, col=1,
                     annotation_text="Entry (Short)")
        fig.add_hline(y=-1.5, line_dash="dot", line_color="green", row=2, col=1,
                     annotation_text="Entry (Long)")
    
    fig.update_layout(
        height=500,
        template="plotly_white",
        showlegend=True,
        hovermode="x unified"
    )
    
    return fig


def create_performance_dashboard(results):
    """Create a performance summary dashboard."""
    metrics = results.get('performance_metrics', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = results.get('total_return', 0) * 100
        st.metric(
            "Total Return",
            f"{total_return:.2f}%",
            delta=f"{'▲' if total_return > 0 else '▼'}"
        )
    
    with col2:
        sharpe = metrics.get('sharpe_ratio', 0)
        st.metric("Sharpe Ratio", f"{sharpe:.3f}")
    
    with col3:
        win_rate = metrics.get('win_rate', 0) * 100
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with col4:
        num_trades = len(results.get('trades', []))
        st.metric("Total Trades", num_trades)


# ============================================================================
# PAGE: BACKTESTING
# ============================================================================

def page_backtesting():
    st.header("Strategy Backtesting")
    
    # Sidebar configuration
    st.sidebar.subheader("Backtest Configuration")
    
    use_enhanced = st.sidebar.checkbox(
        "Enhanced Features",
        value=True,
        help="Enable transaction costs and risk management"
    )
    
    tickers = st.sidebar.multiselect(
        "Select Tickers",
        options=TICKERS + ['BAC', 'GS', 'C', 'XOM', 'CVX', 'DAL', 'UAL'],
        default=["AAPL", "MSFT", "GOOGL", "AMZN"]
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
    with col2:
        end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))
    
    entry_z = st.sidebar.slider("Entry Z-Score", 0.5, 3.0, 1.5, 0.1)
    exit_z = st.sidebar.slider("Exit Z-Score", 0.1, 2.0, 0.5, 0.1)
    zscore_window = st.sidebar.slider("Z-Score Window", 10, 90, 30)
    
    if use_enhanced:
        st.sidebar.subheader("Cost Settings")
        enable_costs = st.sidebar.checkbox("Transaction Costs", value=True)
        enable_risk = st.sidebar.checkbox("Risk Management", value=True)
        initial_capital = st.sidebar.number_input(
            "Initial Capital ($)",
            value=100000, min_value=10000, step=10000
        )
    else:
        enable_costs = False
        enable_risk = False
        initial_capital = 100000
    
    if st.sidebar.button("Run Backtest", type="primary"):
        run_backtest(
            tickers, start_date, end_date, entry_z, exit_z,
            zscore_window, use_enhanced, enable_costs, enable_risk,
            initial_capital
        )
    
    # Display results
    display_backtest_results()


def run_backtest(tickers, start_date, end_date, entry_z, exit_z,
                 zscore_window, use_enhanced, enable_costs, enable_risk,
                 initial_capital):
    """Execute the backtest with given parameters."""
    with st.spinner("Running backtest..."):
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/signals", exist_ok=True)
        
        # Get price data
        try:
            if DATA_DOWNLOAD_AVAILABLE:
                download_prices(tickers, str(start_date), str(end_date), PRICE_DATA_PATH)
            else:
                st.warning("Using sample data (yfinance not available)")
                create_sample_data(tickers, str(start_date), str(end_date), PRICE_DATA_PATH)
            
            price_df = pd.read_csv(PRICE_DATA_PATH, index_col=0, parse_dates=True)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            # Create fallback data
            sample_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            np.random.seed(42)
            sample_data = {t: 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(sample_dates))))
                         for t in tickers[:4]}
            price_df = pd.DataFrame(sample_data, index=sample_dates)
        
        # Find cointegrated pairs
        pairs, _ = find_cointegrated_pairs(price_df, significance=P_VALUE_THRESHOLD)
        
        if not pairs:
            st.session_state['pairs'] = []
            st.session_state['results'] = {}
            st.error("No cointegrated pairs found. Try different tickers or time period.")
            show_cointegration_help()
            return
        
        # Run backtests
        results = {}
        progress_bar = st.progress(0)
        
        for i, (t1, t2, pval) in enumerate(pairs):
            s1 = np.log(price_df[t1])
            s2 = np.log(price_df[t2])
            z, spread, hedge, df = compute_zscore_spread(
                s1, s2, window=zscore_window, ticker1=t1, ticker2=t2
            )
            
            temp_path = f"data/signals/{t1}_{t2}_signals.csv"
            df.to_csv(temp_path)
            
            if use_enhanced:
                bt = EnhancedPairTradingBacktester(
                    path_to_signals=temp_path,
                    entry_z=entry_z,
                    exit_z=exit_z,
                    initial_capital=initial_capital,
                    tickers=(t1, t2),
                    enable_costs=enable_costs,
                    enable_risk_management=enable_risk
                )
                result = bt.backtest()
                
                if 'trades' in result and len(result['trades']) > 0:
                    trades_df = pd.DataFrame(result['trades'])
                    trades_df["Pair"] = f"{t1} & {t2}"
                    
                    results[f"{t1} & {t2}"] = {
                        "trades": trades_df,
                        "spread_df": df,
                        "pnl_series": trades_df.get("pnl", pd.Series()),
                        "result": result,
                        "pvalue": pval
                    }
            else:
                bt = PairTradingBacktester(temp_path, entry_z, exit_z, initial_capital, tickers=(t1, t2))
                bt.backtest()
                trades = bt.summary()
                trades["Pair"] = f"{t1} & {t2}"
                
                results[f"{t1} & {t2}"] = {
                    "trades": trades,
                    "spread_df": df,
                    "pnl_series": trades["PnL"],
                    "pvalue": pval
                }
            
            progress_bar.progress((i + 1) / len(pairs))
        
        st.session_state['pairs'] = list(results.keys())
        st.session_state['results'] = results
        st.session_state['use_enhanced'] = use_enhanced
        
        st.success(f"Backtest complete! Found {len(results)} tradeable pairs.")


def display_backtest_results():
    """Display backtest results with interactive charts."""
    if 'pairs' not in st.session_state or not st.session_state['pairs']:
        st.info("Configure parameters in the sidebar and click 'Run Backtest' to begin.")
        return
    
    # Pair selector
    selected_pair = st.selectbox(
        "Select Pair to Analyze",
        st.session_state['pairs'],
        format_func=lambda x: f"{x} (p={st.session_state['results'][x].get('pvalue', 0):.4f})"
    )
    
    if not selected_pair:
        return
    
    result = st.session_state['results'][selected_pair]
    trades = result['trades']
    pnl_series = result['pnl_series']
    spread_df = result['spread_df']
    
    # Metrics
    st.subheader("Performance Summary")
    
    if st.session_state.get('use_enhanced') and 'result' in result:
        create_performance_dashboard(result['result'])
    else:
        col1, col2, col3, col4 = st.columns(4)
        if len(pnl_series) > 0:
            col1.metric("Sharpe Ratio", f"{sharpe_ratio(pnl_series):.3f}")
            col2.metric("Max Drawdown", f"${max_drawdown(pnl_series):.2f}")
            col3.metric("Total Trades", len(trades))
            col4.metric("Total P&L", f"${pnl_series.sum():.2f}")
    
    # Charts
    tabs = st.tabs(["P&L Analysis", "Spread & Z-Score", "Trade Details"])
    
    with tabs[0]:
        if len(pnl_series) > 0:
            st.plotly_chart(plot_cumulative_pnl_plotly(pnl_series), use_container_width=True)
            
            # P&L distribution
            fig_hist = px.histogram(
                pnl_series, nbins=30,
                title="P&L Distribution",
                labels={'value': 'P&L ($)', 'count': 'Frequency'}
            )
            fig_hist.update_layout(template="plotly_white")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("No trades generated for this pair.")
    
    with tabs[1]:
        st.plotly_chart(plot_spread_zscore_plotly(spread_df), use_container_width=True)
    
    with tabs[2]:
        st.dataframe(trades, use_container_width=True)
        
        # Download button
        csv = trades.to_csv(index=False)
        st.download_button(
            label="Download Trade Data",
            data=csv,
            file_name=f"{selected_pair.replace(' & ', '_')}_trades.csv",
            mime="text/csv"
        )


def show_cointegration_help():
    """Show help for cointegration failures."""
    with st.expander("Why no pairs found? Click for tips", expanded=True):
        st.markdown("""
        ### Common Reasons
        
        1. **Tickers not in same sector** - Cointegration works best with similar companies
        2. **Time period too short** - Need 1-2+ years of data
        3. **Market regime change** - Historical relationships may have broken
        
        ### Recommended Combinations
        
        | Sector | Tickers |
        |--------|---------|
        | Tech | MSFT, GOOGL, AAPL, META |
        | Banks | JPM, BAC, GS, C |
        | Oil | XOM, CVX |
        | Airlines | DAL, UAL |
        """)


# ============================================================================
# PAGE: PAPER TRADING
# ============================================================================

def page_paper_trading():
    st.header("Paper Trading")
    
    if not PAPER_TRADING_AVAILABLE:
        st.warning("Paper trading module not available. Please check installation.")
        return
    
    # Configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Portfolio Overview")
        
        summary = paper_trading_engine.get_portfolio_summary()
        
        # Portfolio metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Capital", f"${summary['current_capital']:,.2f}")
        m2.metric("Unrealized P&L", f"${summary['total_unrealized_pnl']:,.2f}")
        m3.metric("Realized P&L", f"${summary['total_realized_pnl']:,.2f}")
        m4.metric("Total Return", f"{summary['total_return']:.2%}")
        
        # Open positions
        st.subheader("Open Positions")
        if summary['positions']:
            positions_data = []
            for pos in summary['positions']:
                positions_data.append({
                    "Pair": pos.pair,
                    "Direction": pos.direction,
                    "Entry Date": pos.entry_date.strftime("%Y-%m-%d"),
                    "Entry Z": f"{pos.entry_zscore:.2f}",
                    "Current Z": f"{pos.current_zscore:.2f}",
                    "Unrealized P&L": f"${pos.unrealized_pnl:.2f}",
                    "Size": f"${pos.position_size:,.2f}"
                })
            st.dataframe(pd.DataFrame(positions_data), use_container_width=True)
        else:
            st.info("No open positions")
        
        # Recent trades
        st.subheader("Recent Closed Trades")
        if summary['recent_trades']:
            st.dataframe(pd.DataFrame(summary['recent_trades']), use_container_width=True)
        else:
            st.info("No closed trades yet")
    
    with col2:
        st.subheader("Open New Position")
        
        with st.form("new_position"):
            ticker1 = st.text_input("Ticker 1", value="AAPL")
            ticker2 = st.text_input("Ticker 2", value="MSFT")
            direction = st.selectbox("Direction", ["long_spread", "short_spread"])
            size = st.number_input("Position Size ($)", value=10000, min_value=1000)
            
            if st.form_submit_button("Open Position"):
                try:
                    pos = paper_trading_engine.open_position(
                        ticker1, ticker2, direction, size
                    )
                    st.success(f"Opened {direction} position on {ticker1}/{ticker2}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        
        st.subheader("Close Position")
        open_pairs = list(paper_trading_engine.positions.keys())
        if open_pairs:
            pair_to_close = st.selectbox("Select Position", open_pairs)
            if st.button("Close Position"):
                try:
                    trade = paper_trading_engine.close_position(pair_to_close)
                    st.success(f"Closed position with P&L: ${trade['realized_pnl']:.2f}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")


# ============================================================================
# PAGE: ALERTS
# ============================================================================

def page_alerts():
    st.header("Alerts & Notifications")
    
    if not NOTIFICATIONS_AVAILABLE:
        st.warning("Notifications module not available.")
        st.info("To enable: Install email dependencies and configure SMTP in .env")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Active Alerts")
        
        # Display active alerts (would need to load from database)
        st.info("No active alerts. Create one using the form on the right.")
    
    with col2:
        st.subheader("Create Alert")
        
        with st.form("create_alert"):
            pair = st.text_input("Pair (e.g., AAPL_MSFT)")
            alert_type = st.selectbox("Alert Type", ["zscore_entry", "zscore_exit", "price_target"])
            condition = st.selectbox("Condition", ["above", "below", "crosses"])
            threshold = st.number_input("Threshold", value=1.5)
            
            if st.form_submit_button("Create Alert"):
                st.success(f"Alert created for {pair}")
                st.info("Note: Email notifications require SMTP configuration in .env")


# ============================================================================
# PAGE: CONFIGURATION
# ============================================================================

def page_configuration():
    st.header("System Configuration")
    
    tabs = st.tabs(["Status", "Settings", "About"])
    
    with tabs[0]:
        st.subheader("System Status")
        
        status_items = [
            ("Data Download (yfinance)", DATA_DOWNLOAD_AVAILABLE),
            ("Paper Trading", PAPER_TRADING_AVAILABLE),
            ("Database", DATABASE_AVAILABLE),
            ("Authentication", AUTH_AVAILABLE),
            ("Notifications", NOTIFICATIONS_AVAILABLE),
        ]
        
        for name, available in status_items:
            if available:
                st.success(f"✓ {name}: Available")
            else:
                st.warning(f"○ {name}: Not configured")
        
        st.subheader("Module Tests")
        if st.button("Run System Tests"):
            run_system_tests()
    
    with tabs[1]:
        st.subheader("Current Settings")
        
        st.json({
            "tickers": TICKERS,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "p_value_threshold": P_VALUE_THRESHOLD,
            "zscore_window": ZSCORE_WINDOW,
            "entry_z": ENTRY_Z,
            "exit_z": EXIT_Z,
            "initial_capital": INITIAL_CAPITAL,
            "debug": DEBUG,
            "log_level": LOG_LEVEL
        })
        
        st.info("Edit .env file to change settings, then restart the app.")
    
    with tabs[2]:
        st.subheader("About")
        st.markdown("""
        **Statistical Arbitrage Simulator**
        
        A framework for backtesting cointegration-based pairs trading strategies.
        
        **Features:**
        - Cointegration testing and pair identification
        - Z-score based signal generation
        - Transaction cost modeling
        - Risk management
        - Paper trading simulation
        
        **Source:** [GitHub](https://github.com/sidp24/Statistical-Arbitrage-Simulator)
        """)


def run_system_tests():
    """Run system component tests."""
    test_results = {}
    
    # Test imports
    try:
        import pandas, numpy
        test_results["Core Libraries"] = "OK"
    except Exception as e:
        test_results["Core Libraries"] = f"ERROR: {e}"
    
    try:
        from backtest.enhanced_backtester import EnhancedPairTradingBacktester
        test_results["Enhanced Backtester"] = "OK"
    except Exception as e:
        test_results["Enhanced Backtester"] = f"ERROR: {e}"
    
    try:
        from trading.transaction_costs import TransactionCostModel
        test_results["Transaction Costs"] = "OK"
    except Exception as e:
        test_results["Transaction Costs"] = f"Not Available"
    
    try:
        from risk.risk_management import RiskManager
        test_results["Risk Management"] = "OK"
    except Exception as e:
        test_results["Risk Management"] = f"Not Available"
    
    for component, status in test_results.items():
        if "OK" in status:
            st.success(f"✓ {component}: {status}")
        elif "ERROR" in status:
            st.error(f"✗ {component}: {status}")
        else:
            st.warning(f"○ {component}: {status}")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Title
    st.title("Statistical Arbitrage Simulator")
    
    # Navigation
    st.sidebar.title("Navigation")
    
    pages = {
        "Backtesting": page_backtesting,
        "Paper Trading": page_paper_trading,
        "Alerts": page_alerts,
        "Configuration": page_configuration,
    }
    
    page = st.sidebar.radio("Select Page", list(pages.keys()))
    
    # Separator
    st.sidebar.markdown("---")
    
    # Run selected page
    pages[page]()


if __name__ == "__main__":
    main()

