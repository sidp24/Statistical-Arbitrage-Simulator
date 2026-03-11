import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_price_data():
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2022-01-01', periods=n, freq='D')
    
    # Create cointegrated series
    common_factor = np.cumsum(np.random.normal(0, 1, n))
    
    stock_a = 100 + common_factor + np.random.normal(0, 0.5, n)
    stock_b = 50 + 0.5 * common_factor + np.random.normal(0, 0.3, n)
    
    return pd.DataFrame({
        'STOCK_A': stock_a,
        'STOCK_B': stock_b
    }, index=dates)


@pytest.fixture
def sample_signal_data():
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')
    
    # Generate synthetic spread data
    price_changes = np.random.normal(0, 0.015, len(dates))
    spread = pd.Series(np.cumsum(price_changes), index=dates)
    spread = spread - spread.rolling(window=20, min_periods=1).mean()
    
    # Calculate z-score
    zscore_window = 15
    spread_mean = spread.rolling(window=zscore_window).mean()
    spread_std = spread.rolling(window=zscore_window).std()
    zscore = (spread - spread_mean) / spread_std
    
    return pd.DataFrame({
        'spread': spread,
        'zscore': zscore.fillna(0)
    }, index=dates)


@pytest.fixture
def sample_trades():
    return pd.DataFrame({
        'entry_date': pd.date_range('2023-01-01', periods=10, freq='15D'),
        'exit_date': pd.date_range('2023-01-10', periods=10, freq='15D'),
        'direction': ['long', 'short'] * 5,
        'pnl': np.random.randn(10) * 1000,
        'entry_zscore': np.random.randn(10) * 1.5,
        'exit_zscore': np.random.randn(10) * 0.5
    })


@pytest.fixture
def temp_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    signals_dir = data_dir / "signals"
    signals_dir.mkdir()
    return data_dir


@pytest.fixture(scope="session")
def database_url():
    return "sqlite:///:memory:"
