"""
Trading Costs and Slippage Models for Statistical Arbitrage

This module implements realistic transaction cost models including:
- Commission costs
- Bid-ask spread costs
- Market impact (slippage)
- Financing costs for short positions
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class TransactionCostModel:
    """
    Comprehensive transaction cost model for pairs trading
    """
    
    def __init__(self, 
                 commission_per_share: float = 0.005,  # $0.005 per share (typical retail)
                 commission_min: float = 1.0,          # Minimum commission per trade
                 bid_ask_spread_bps: float = 5.0,      # 5 basis points typical for liquid stocks
                 market_impact_coeff: float = 0.1,     # Market impact coefficient
                 financing_rate: float = 0.02,         # 2% annual rate for short financing
                 slippage_factor: float = 0.5):        # Additional slippage factor
        
        self.commission_per_share = commission_per_share
        self.commission_min = commission_min
        self.bid_ask_spread_bps = bid_ask_spread_bps
        self.market_impact_coeff = market_impact_coeff
        self.financing_rate = financing_rate
        self.slippage_factor = slippage_factor
    
    def calculate_commission(self, shares: float) -> float:
        """Calculate commission cost"""
        return max(self.commission_min, abs(shares) * self.commission_per_share)
    
    def calculate_bid_ask_cost(self, notional_value: float) -> float:
        """Calculate bid-ask spread cost"""
        return abs(notional_value) * (self.bid_ask_spread_bps / 10000)
    
    def calculate_market_impact(self, shares: float, avg_volume: float, 
                              volatility: float) -> float:
        """
        Calculate market impact (temporary + permanent)
        
        Model: Impact = coefficient * (shares/volume)^0.5 * volatility
        """
        if avg_volume <= 0:
            avg_volume = 1000000  # Default volume if missing
            
        volume_ratio = abs(shares) / avg_volume
        impact = self.market_impact_coeff * np.sqrt(volume_ratio) * volatility
        return abs(shares) * impact
    
    def calculate_financing_cost(self, short_value: float, days_held: int) -> float:
        """Calculate cost of financing short positions"""
        if short_value <= 0:
            return 0.0
        return short_value * (self.financing_rate / 365) * days_held
    
    def calculate_slippage(self, notional_value: float, volatility: float) -> float:
        """Calculate additional slippage due to execution timing"""
        return abs(notional_value) * self.slippage_factor * volatility * 0.01
    
    def total_transaction_cost(self, 
                             shares: float, 
                             price: float,
                             avg_volume: float = 1000000,
                             volatility: float = 0.02,
                             days_held: int = 1,
                             is_short: bool = False) -> Dict[str, float]:
        """
        Calculate total transaction costs for a trade
        
        Returns breakdown of all cost components
        """
        notional_value = abs(shares * price)
        
        costs = {
            'commission': self.calculate_commission(shares),
            'bid_ask': self.calculate_bid_ask_cost(notional_value),
            'market_impact': self.calculate_market_impact(shares, avg_volume, volatility),
            'slippage': self.calculate_slippage(notional_value, volatility),
            'financing': 0.0
        }
        
        if is_short and days_held > 0:
            costs['financing'] = self.calculate_financing_cost(notional_value, days_held)
        
        costs['total'] = sum(costs.values())
        costs['total_bps'] = (costs['total'] / notional_value) * 10000 if notional_value > 0 else 0
        
        return costs


class PairTradingCosts:
    """
    Specialized cost calculator for pairs trading
    """
    
    def __init__(self, cost_model: TransactionCostModel):
        self.cost_model = cost_model
    
    def calculate_pair_entry_costs(self, 
                                 long_shares: float, long_price: float,
                                 short_shares: float, short_price: float,
                                 market_data: Dict) -> Dict[str, float]:
        """Calculate costs for entering a pairs trade"""
        
        # Long leg costs
        long_costs = self.cost_model.total_transaction_cost(
            shares=long_shares,
            price=long_price,
            avg_volume=market_data.get('long_volume', 1000000),
            volatility=market_data.get('long_vol', 0.02),
            days_held=0,
            is_short=False
        )
        
        # Short leg costs
        short_costs = self.cost_model.total_transaction_cost(
            shares=-short_shares,  # Negative for short
            price=short_price,
            avg_volume=market_data.get('short_volume', 1000000),
            volatility=market_data.get('short_vol', 0.02),
            days_held=0,
            is_short=True
        )
        
        return {
            'long_leg': long_costs,
            'short_leg': short_costs,
            'total_entry_cost': long_costs['total'] + short_costs['total'],
            'total_entry_bps': ((long_costs['total'] + short_costs['total']) / 
                              (abs(long_shares * long_price) + abs(short_shares * short_price))) * 10000
        }
    
    def calculate_pair_exit_costs(self, 
                                long_shares: float, long_price: float,
                                short_shares: float, short_price: float,
                                days_held: int,
                                market_data: Dict) -> Dict[str, float]:
        """Calculate costs for exiting a pairs trade"""
        
        # Exit long position (sell)
        long_exit_costs = self.cost_model.total_transaction_cost(
            shares=-long_shares,  # Selling long position
            price=long_price,
            avg_volume=market_data.get('long_volume', 1000000),
            volatility=market_data.get('long_vol', 0.02),
            days_held=days_held,
            is_short=False
        )
        
        # Exit short position (cover)
        short_exit_costs = self.cost_model.total_transaction_cost(
            shares=short_shares,  # Covering short position
            price=short_price,
            avg_volume=market_data.get('short_volume', 1000000),
            volatility=market_data.get('short_vol', 0.02),
            days_held=days_held,
            is_short=True
        )
        
        # Add financing costs for short position
        short_financing = self.cost_model.calculate_financing_cost(
            short_shares * short_price, days_held
        )
        
        return {
            'long_leg': long_exit_costs,
            'short_leg': short_exit_costs,
            'financing_cost': short_financing,
            'total_exit_cost': long_exit_costs['total'] + short_exit_costs['total'] + short_financing,
            'total_exit_bps': ((long_exit_costs['total'] + short_exit_costs['total'] + short_financing) / 
                             (abs(long_shares * long_price) + abs(short_shares * short_price))) * 10000
        }


def estimate_market_volatility(price_series: pd.Series, window: int = 20) -> float:
    """Estimate volatility from price series"""
    returns = price_series.pct_change().dropna()
    return returns.rolling(window=window).std().iloc[-1] * np.sqrt(252)


def estimate_avg_volume(volume_series: pd.Series, window: int = 20) -> float:
    """Estimate average volume"""
    if volume_series is None or volume_series.empty:
        return 1000000  # Default volume
    return volume_series.rolling(window=window).mean().iloc[-1]


# Example usage and testing
if __name__ == "__main__":
    # Create cost model with typical retail parameters
    cost_model = TransactionCostModel(
        commission_per_share=0.005,
        commission_min=1.0,
        bid_ask_spread_bps=5.0,
        market_impact_coeff=0.1,
        financing_rate=0.02,
        slippage_factor=0.5
    )
    
    # Test pair trading costs
    pair_costs = PairTradingCosts(cost_model)
    
    market_data = {
        'long_volume': 2000000,
        'long_vol': 0.25,
        'short_volume': 1500000,
        'short_vol': 0.22
    }
    
    # Example trade: Long 100 AAPL @ $150, Short 95 MSFT @ $300
    entry_costs = pair_costs.calculate_pair_entry_costs(
        long_shares=100, long_price=150.0,
        short_shares=95, short_price=300.0,
        market_data=market_data
    )
    
    print("Entry Costs Breakdown:")
    print(f"Total Entry Cost: ${entry_costs['total_entry_cost']:.2f}")
    print(f"Total Entry Cost (bps): {entry_costs['total_entry_bps']:.1f}")
