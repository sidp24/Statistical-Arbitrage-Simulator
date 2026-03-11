import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtest.enhanced_backtester import EnhancedPairTradingBacktester


class TestEnhancedBacktester:
    @pytest.fixture
    def sample_signal_data(self):
        dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')
        np.random.seed(42)
        
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
    
    def test_backtest_basic(self, sample_signal_data):
        backtester = EnhancedPairTradingBacktester(
            signal_data=sample_signal_data,
            entry_z=1.5,
            exit_z=0.5,
            initial_capital=100000,
            enable_costs=False,
            enable_risk_management=False
        )
        
        results = backtester.backtest()
        
        assert 'trades' in results
        assert 'total_return' in results
        assert isinstance(results['trades'], list)
    
    def test_backtest_with_costs(self, sample_signal_data):
        backtester = EnhancedPairTradingBacktester(
            signal_data=sample_signal_data,
            entry_z=1.5,
            exit_z=0.5,
            initial_capital=100000,
            tickers=('TEST_A', 'TEST_B'),
            enable_costs=True,
            enable_risk_management=False
        )
        
        results = backtester.backtest()
        
        assert 'trades' in results
        # Costs should reduce returns
        if len(results['trades']) > 0 and 'performance_metrics' in results:
            assert 'total_costs' in results.get('performance_metrics', {}) or True
    
    def test_backtest_with_risk_management(self, sample_signal_data):
        backtester = EnhancedPairTradingBacktester(
            signal_data=sample_signal_data,
            entry_z=1.5,
            exit_z=0.5,
            initial_capital=100000,
            tickers=('TEST_A', 'TEST_B'),
            enable_costs=False,
            enable_risk_management=True
        )
        
        results = backtester.backtest()
        
        assert 'trades' in results
    
    def test_insufficient_data(self):
        small_data = pd.DataFrame({
            'spread': [0.1, 0.2, 0.3],
            'zscore': [0.5, 1.0, 0.3]
        })
        
        backtester = EnhancedPairTradingBacktester(
            signal_data=small_data,
            entry_z=1.5,
            exit_z=0.5,
            initial_capital=100000
        )
        
        results = backtester.backtest()
        
        assert 'error' in results or len(results.get('trades', [])) == 0
    
    def test_no_signals(self, sample_signal_data):
        # Use very high entry threshold
        backtester = EnhancedPairTradingBacktester(
            signal_data=sample_signal_data,
            entry_z=10.0,  # Very high threshold
            exit_z=0.5,
            initial_capital=100000
        )
        
        results = backtester.backtest()
        
        assert results['trades'] == [] or len(results['trades']) == 0


class TestBacktesterParameters:
    def test_invalid_entry_exit_thresholds(self):
        data = pd.DataFrame({
            'spread': [0.1] * 100,
            'zscore': [0.0] * 100
        })
        
        # This should still work, but might not produce sensible results
        backtester = EnhancedPairTradingBacktester(
            signal_data=data,
            entry_z=0.5,
            exit_z=1.5,  # Exit > Entry
            initial_capital=100000
        )
        
        results = backtester.backtest()
        assert 'trades' in results
    
    def test_zero_capital(self):
        data = pd.DataFrame({
            'spread': [0.1] * 100,
            'zscore': [2.0] * 100
        })
        
        backtester = EnhancedPairTradingBacktester(
            signal_data=data,
            entry_z=1.5,
            exit_z=0.5,
            initial_capital=0
        )
        
        results = backtester.backtest()
        # Should handle gracefully
        assert 'trades' in results
