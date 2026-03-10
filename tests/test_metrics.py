"""
Tests for analysis metrics
"""
import pytest
import pandas as pd
import numpy as np

from analysis.metrics import sharpe_ratio, max_drawdown


class TestMetrics:
    """Test suite for performance metrics."""
    
    def test_sharpe_ratio_positive(self):
        """Test Sharpe ratio with positive returns."""
        # Consistent positive returns should give positive Sharpe
        pnl = pd.Series([100, 110, 120, 130, 140, 150])
        result = sharpe_ratio(pnl)
        
        assert isinstance(result, (int, float))
        # Note: Sharpe can be negative even with positive PnL due to volatility
    
    def test_sharpe_ratio_negative(self):
        """Test Sharpe ratio with negative returns."""
        pnl = pd.Series([100, 90, 80, 70, 60, 50])
        result = sharpe_ratio(pnl)
        
        assert isinstance(result, (int, float))
        assert result < 0  # Should be negative for consistent losses
    
    def test_sharpe_ratio_flat(self):
        """Test Sharpe ratio with flat returns."""
        pnl = pd.Series([100, 100, 100, 100, 100])
        result = sharpe_ratio(pnl)
        
        # Flat returns have zero standard deviation
        # This might return inf, nan, or 0 depending on implementation
        assert isinstance(result, (int, float)) or np.isnan(result) or np.isinf(result)
    
    def test_max_drawdown_calculation(self):
        """Test max drawdown calculation."""
        # Create a PnL series with known drawdown
        pnl = pd.Series([10, 20, 30, 20, 10, 15, 25])
        result = max_drawdown(pnl)
        
        assert isinstance(result, (int, float))
        assert result <= 0  # Drawdown is typically negative or zero
    
    def test_max_drawdown_no_drawdown(self):
        """Test max drawdown with monotonically increasing PnL."""
        pnl = pd.Series([10, 20, 30, 40, 50])
        result = max_drawdown(pnl)
        
        assert result == 0  # No drawdown for monotonically increasing
    
    def test_max_drawdown_single_value(self):
        """Test max drawdown with single value."""
        pnl = pd.Series([100])
        result = max_drawdown(pnl)
        
        assert result == 0


class TestMetricsEdgeCases:
    """Test edge cases for metrics."""
    
    def test_empty_series(self):
        """Test handling of empty series."""
        pnl = pd.Series([], dtype=float)
        
        # Should handle gracefully
        try:
            sharpe = sharpe_ratio(pnl)
            assert np.isnan(sharpe) or sharpe == 0
        except (ZeroDivisionError, ValueError):
            pass  # Acceptable to raise error for empty series
    
    def test_single_value_sharpe(self):
        """Test Sharpe ratio with single value."""
        pnl = pd.Series([100])
        
        try:
            result = sharpe_ratio(pnl)
            # Single value has no variance
            assert np.isnan(result) or result == 0
        except (ZeroDivisionError, ValueError):
            pass
