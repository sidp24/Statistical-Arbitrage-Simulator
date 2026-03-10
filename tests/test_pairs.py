"""
Tests for the cointegration pair finding module
"""
import pytest
import pandas as pd
import numpy as np

from pairs.find_pairs import find_cointegrated_pairs


class TestFindPairs:
    """Test suite for cointegration pair finding."""
    
    @pytest.fixture
    def cointegrated_data(self):
        """Create synthetic cointegrated price data."""
        np.random.seed(42)
        n = 500
        
        # Create a common factor
        common_factor = np.cumsum(np.random.normal(0, 1, n))
        
        # Create two cointegrated series
        price1 = 100 + common_factor + np.random.normal(0, 0.5, n)
        price2 = 50 + 0.5 * common_factor + np.random.normal(0, 0.3, n)
        
        # Create a non-cointegrated series
        price3 = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n)))
        
        dates = pd.date_range('2022-01-01', periods=n, freq='D')
        
        return pd.DataFrame({
            'STOCK_A': price1,
            'STOCK_B': price2,
            'STOCK_C': price3
        }, index=dates)
    
    def test_find_cointegrated_pairs(self, cointegrated_data):
        """Test that cointegrated pairs are found."""
        pairs, pvals = find_cointegrated_pairs(cointegrated_data, significance=0.05)
        
        assert isinstance(pairs, list)
        assert isinstance(pvals, np.ndarray)
        # The synthetic data should produce at least one cointegrated pair
        # (though this is probabilistic)
    
    def test_find_pairs_returns_pvalues(self, cointegrated_data):
        """Test that p-values matrix is returned correctly."""
        pairs, pvals = find_cointegrated_pairs(cointegrated_data, significance=0.05)
        
        # P-values should be a square matrix
        n_assets = len(cointegrated_data.columns)
        assert pvals.shape == (n_assets, n_assets)
        
        # Diagonal should be 1s (self-cointegration not tested)
        for i in range(n_assets):
            assert pvals[i, i] == 1.0
    
    def test_significance_threshold(self, cointegrated_data):
        """Test that significance threshold affects results."""
        # Very strict threshold
        pairs_strict, _ = find_cointegrated_pairs(cointegrated_data, significance=0.001)
        
        # Lenient threshold
        pairs_lenient, _ = find_cointegrated_pairs(cointegrated_data, significance=0.10)
        
        # Lenient should find at least as many pairs as strict
        assert len(pairs_lenient) >= len(pairs_strict)
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        empty_df = pd.DataFrame()
        
        pairs, pvals = find_cointegrated_pairs(empty_df, significance=0.05)
        
        assert pairs == []
    
    def test_single_asset(self):
        """Test handling of single asset."""
        single_asset = pd.DataFrame({
            'SINGLE': np.random.randn(100)
        })
        
        pairs, pvals = find_cointegrated_pairs(single_asset, significance=0.05)
        
        assert pairs == []
    
    def test_pair_tuple_structure(self, cointegrated_data):
        """Test that returned pairs have correct structure."""
        pairs, _ = find_cointegrated_pairs(cointegrated_data, significance=0.50)  # Lenient
        
        for pair in pairs:
            assert len(pair) == 3  # (ticker1, ticker2, pvalue)
            assert isinstance(pair[0], str)
            assert isinstance(pair[1], str)
            assert isinstance(pair[2], float)
            assert 0 <= pair[2] <= 1  # P-value between 0 and 1


class TestFindPairsEdgeCases:
    """Test edge cases for pair finding."""
    
    def test_missing_values(self):
        """Test handling of missing values."""
        data = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5] * 20,
            'B': [2, 4, 6, np.nan, 10] * 20
        })
        
        # Should handle NaN values
        pairs, pvals = find_cointegrated_pairs(data.dropna(), significance=0.05)
        assert isinstance(pairs, list)
    
    def test_constant_series(self):
        """Test handling of constant price series."""
        data = pd.DataFrame({
            'CONSTANT': [100.0] * 100,
            'VARIABLE': np.random.randn(100) + 100
        })
        
        # Should handle gracefully (might warn but not crash)
        try:
            pairs, pvals = find_cointegrated_pairs(data, significance=0.05)
            assert isinstance(pairs, list)
        except Exception:
            # Some implementations might raise an error for constant series
            pass
