"""
Tests for transaction cost model
"""
import pytest
import numpy as np

from trading.transaction_costs import TransactionCostModel


class TestTransactionCosts:
    """Test suite for transaction cost modeling."""
    
    @pytest.fixture
    def cost_model(self):
        """Create a default cost model."""
        return TransactionCostModel()
    
    def test_commission_calculation(self, cost_model):
        """Test commission calculation."""
        # Test with enough shares to exceed minimum
        commission = cost_model.calculate_commission(1000)
        
        assert commission >= cost_model.commission_min
        assert commission == max(cost_model.commission_min, 1000 * cost_model.commission_per_share)
    
    def test_commission_minimum(self, cost_model):
        """Test minimum commission is applied."""
        # Very small trade should still incur minimum commission
        commission = cost_model.calculate_commission(1)
        
        assert commission == cost_model.commission_min
    
    def test_bid_ask_cost(self, cost_model):
        """Test bid-ask spread cost calculation."""
        notional = 10000
        cost = cost_model.calculate_bid_ask_cost(notional)
        
        expected = notional * (cost_model.bid_ask_spread_bps / 10000)
        assert cost == pytest.approx(expected)
    
    def test_market_impact(self, cost_model):
        """Test market impact calculation."""
        shares = 1000
        volume = 1000000
        volatility = 0.02
        
        impact = cost_model.calculate_market_impact(shares, volume, volatility)
        
        assert impact >= 0
        assert isinstance(impact, float)
    
    def test_financing_cost(self, cost_model):
        """Test short financing cost calculation."""
        short_value = 100000
        days_held = 30
        
        cost = cost_model.calculate_financing_cost(short_value, days_held)
        
        expected = short_value * (cost_model.financing_rate / 365) * days_held
        assert cost == pytest.approx(expected)
    
    def test_total_transaction_cost(self, cost_model):
        """Test total transaction cost calculation."""
        costs = cost_model.total_transaction_cost(
            shares=1000,
            price=100,
            avg_volume=1000000,
            volatility=0.02,
            days_held=5,
            is_short=False
        )
        
        assert 'total' in costs
        assert 'commission' in costs
        assert 'bid_ask' in costs
        assert 'market_impact' in costs
        assert 'slippage' in costs
        assert 'financing' in costs
        assert 'total_bps' in costs
        
        # Total should be sum of components
        component_sum = (
            costs['commission'] + 
            costs['bid_ask'] + 
            costs['market_impact'] + 
            costs['slippage'] + 
            costs['financing']
        )
        assert costs['total'] == pytest.approx(component_sum)
    
    def test_short_financing_applied(self, cost_model):
        """Test that financing cost is applied for short positions."""
        costs_long = cost_model.total_transaction_cost(
            shares=1000, price=100, is_short=False, days_held=30
        )
        costs_short = cost_model.total_transaction_cost(
            shares=1000, price=100, is_short=True, days_held=30
        )
        
        assert costs_short['financing'] > 0
        assert costs_long['financing'] == 0
        assert costs_short['total'] > costs_long['total']


class TestTransactionCostsEdgeCases:
    """Test edge cases for transaction costs."""
    
    def test_zero_shares(self):
        """Test handling of zero shares."""
        model = TransactionCostModel()
        costs = model.total_transaction_cost(shares=0, price=100)
        
        # Should still have minimum commission
        assert costs['commission'] == model.commission_min
    
    def test_zero_volume(self):
        """Test handling of zero average volume."""
        model = TransactionCostModel()
        
        # Should use default volume
        impact = model.calculate_market_impact(1000, 0, 0.02)
        assert isinstance(impact, float)
    
    def test_negative_shares(self):
        """Test handling of negative shares (indicating sell)."""
        model = TransactionCostModel()
        costs = model.total_transaction_cost(shares=-1000, price=100)
        
        # Should handle absolute value
        assert costs['total'] > 0
