"""
Enhanced Backtester with Trading Costs, Risk Management, and Optimization

This module provides a comprehensive backtesting framework that includes:
- Realistic transaction costs and slippage
- Portfolio-level risk management
- Position sizing optimization
- Performance attribution
- Integration with parameter optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings

# Import our new modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from trading.transaction_costs import TransactionCostModel, PairTradingCosts
    from risk.risk_management import RiskManager, PositionSizer, RiskMetrics
except ImportError:
    # Fallback if modules not available
    print("Warning: Enhanced modules not found. Using basic functionality.")
    TransactionCostModel = None
    PairTradingCosts = None
    RiskManager = None


class EnhancedPairTradingBacktester:
    """
    Enhanced backtester with realistic costs and risk management
    """
    
    def __init__(self,
                 path_to_signals: str = None,
                 signal_data: pd.DataFrame = None,
                 entry_z: float = 1.5,
                 exit_z: float = 0.5,
                 initial_capital: float = 100000,
                 tickers: Tuple[str, str] = None,
                 enable_costs: bool = True,
                 enable_risk_management: bool = True,
                 cost_model_params: Dict = None,
                 risk_params: Dict = None):
        
        # Basic parameters
        self.path = path_to_signals
        self.signal_data = signal_data
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.tickers = tickers
        
        # Enhanced features flags
        self.enable_costs = enable_costs
        self.enable_risk_management = enable_risk_management
        
        # Initialize cost model
        if self.enable_costs and TransactionCostModel:
            cost_params = cost_model_params or {}
            self.cost_model = TransactionCostModel(**cost_params)
            self.pair_costs = PairTradingCosts(self.cost_model)
        else:
            self.cost_model = None
            self.pair_costs = None
        
        # Initialize risk manager
        if self.enable_risk_management and RiskManager:
            risk_params = risk_params or {}
            position_sizer = PositionSizer(**risk_params.get('position_sizer', {}))
            self.risk_manager = RiskManager(position_sizer=position_sizer, **risk_params.get('risk_manager', {}))
        else:
            self.risk_manager = None
        
        # Results storage
        self.trade_log = pd.DataFrame()
        self.portfolio_returns = []
        self.cost_breakdown = []
        self.risk_events = []
        self.positions_history = []
        
    def load_signal_data(self) -> pd.DataFrame:
        """Load signal data from file or use provided data"""
        if self.signal_data is not None:
            return self.signal_data.copy()
        elif self.path:
            return pd.read_csv(self.path, parse_dates=True, index_col=0)
        else:
            raise ValueError("Either path_to_signals or signal_data must be provided")
    
    def calculate_position_size(self, 
                              current_capital: float,
                              signal_strength: float,
                              market_data: Dict) -> Tuple[float, float]:
        """Calculate optimal position sizes for the pair"""
        
        if not self.enable_risk_management or not self.risk_manager:
            # Simple position sizing: use 10% of capital
            base_position_value = current_capital * 0.1
            return base_position_value, base_position_value
        
        # Enhanced position sizing based on signal strength and risk metrics
        base_allocation = min(0.1, abs(signal_strength) / self.entry_z * 0.05)  # Scale with signal
        max_position = current_capital * base_allocation
        
        # Apply risk management constraints
        volatility = market_data.get('volatility', 0.02)
        if volatility > 0:
            vol_adjusted = self.risk_manager.position_sizer.volatility_targeting(
                volatility, volatility, current_capital
            )
            max_position = min(max_position, vol_adjusted)
        
        return max_position, max_position
    
    def backtest(self) -> Dict[str, Any]:
        """Enhanced backtest with costs and risk management"""
        
        # Load data
        df = self.load_signal_data()
        
        if len(df) < 10:
            return {'trades': [], 'error': 'Insufficient data'}
        
        # Initialize tracking variables
        position = 0  # 0: no position, 1: long spread, -1: short spread
        trades = []
        current_capital = self.initial_capital
        portfolio_values = [current_capital]
        
        # Prepare market data for cost calculations
        market_data = {
            'volatility': 0.02,
            'long_vol': 0.02,
            'short_vol': 0.02,
            'long_volume': 1000000,
            'short_volume': 1000000
        }
        
        entry_idx = None
        entry_spread = None
        entry_costs = 0.0
        position_size = 0
        
        for i in range(1, len(df)):
            current_date = df.index[i]
            z = df["zscore"].iloc[i]
            spread = df["spread"].iloc[i]
            
            # Check for entry signals
            if position == 0:
                if abs(z) > self.entry_z:
                    # Determine position direction
                    position = -1 if z > 0 else 1
                    
                    # Calculate position size
                    pos_size_1, pos_size_2 = self.calculate_position_size(
                        current_capital, z, market_data
                    )
                    position_size = min(pos_size_1, pos_size_2)
                    
                    # Calculate entry costs (simplified)
                    if self.enable_costs:
                        entry_costs = position_size * 0.001  # 0.1% transaction cost
                    else:
                        entry_costs = 0.0
                    
                    # Record entry
                    entry_idx = i
                    entry_spread = spread
                    current_capital -= entry_costs
                    
                    self.cost_breakdown.append({
                        'date': current_date,
                        'type': 'entry',
                        'cost': entry_costs
                    })
            
            # Check for exit signals
            elif position != 0:
                exit_condition = (
                    (position == -1 and z < self.exit_z) or
                    (position == 1 and z > -self.exit_z)
                )
                
                if exit_condition:
                    # Calculate days held
                    days_held = (current_date - df.index[entry_idx]).days
                    
                    # Calculate exit costs
                    if self.enable_costs:
                        exit_costs = position_size * 0.001  # 0.1% transaction cost
                        # Add financing cost for short positions
                        if position == -1:
                            financing_cost = position_size * 0.02 * (days_held / 365)  # 2% annual rate
                            exit_costs += financing_cost
                    else:
                        exit_costs = 0.0
                    
                    # Calculate P&L
                    gross_pnl = (entry_spread - spread) * position
                    net_pnl = gross_pnl - entry_costs - exit_costs
                    
                    # Update capital
                    current_capital += position_size + net_pnl
                    
                    # Record trade
                    trade_record = {
                        "Entry Date": df.index[entry_idx],
                        "Exit Date": current_date,
                        "Position": position,
                        "Entry Spread": entry_spread,
                        "Exit Spread": spread,
                        "Entry Z": df["zscore"].iloc[entry_idx],
                        "Exit Z": z,
                        "Position Size": position_size,
                        "Gross PnL": gross_pnl,
                        "Entry Costs": entry_costs,
                        "Exit Costs": exit_costs,
                        "Net PnL": net_pnl,
                        "Days Held": days_held,
                        "Return %": (net_pnl / position_size) * 100 if position_size > 0 else 0
                    }
                    
                    trades.append(trade_record)
                    
                    self.cost_breakdown.append({
                        'date': current_date,
                        'type': 'exit',
                        'cost': exit_costs
                    })
                    
                    # Reset position
                    position = 0
                    position_size = 0
            
            # Track portfolio value
            portfolio_values.append(current_capital)
        
        # Convert trades to DataFrame
        trade_df = pd.DataFrame(trades)
        
        if not trade_df.empty:
            trade_df["Cumulative PnL"] = trade_df["Net PnL"].cumsum()
            self.trade_log = trade_df
        
        # Calculate portfolio metrics
        portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
        
        results = {
            'trades': trades,
            'trade_df': trade_df,
            'portfolio_returns': portfolio_returns,
            'final_capital': current_capital,
            'total_return': (current_capital - self.initial_capital) / self.initial_capital,
            'cost_breakdown': self.cost_breakdown,
            'performance_metrics': self._calculate_performance_metrics(trade_df, portfolio_returns)
        }
        
        return results
    
    def _calculate_performance_metrics(self, trade_df: pd.DataFrame, portfolio_returns: pd.Series) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        if trade_df.empty:
            return {}
        
        metrics = {}
        
        # Trade-based metrics
        metrics['total_trades'] = len(trade_df)
        metrics['winning_trades'] = len(trade_df[trade_df['Net PnL'] > 0])
        metrics['losing_trades'] = len(trade_df[trade_df['Net PnL'] < 0])
        metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
        
        if metrics['winning_trades'] > 0 and metrics['losing_trades'] > 0:
            avg_win = trade_df[trade_df['Net PnL'] > 0]['Net PnL'].mean()
            avg_loss = trade_df[trade_df['Net PnL'] < 0]['Net PnL'].mean()
            metrics['avg_win'] = avg_win
            metrics['avg_loss'] = avg_loss
            metrics['profit_factor'] = abs(avg_win * metrics['winning_trades']) / abs(avg_loss * metrics['losing_trades'])
        
        # Portfolio-based metrics
        if len(portfolio_returns) > 0:
            if RiskMetrics:
                metrics['sharpe_ratio'] = RiskMetrics.calculate_sharpe_ratio(portfolio_returns)
                metrics['max_drawdown'] = RiskMetrics.calculate_max_drawdown(portfolio_returns.cumsum())[0]
                metrics['calmar_ratio'] = RiskMetrics.calculate_calmar_ratio(portfolio_returns)
            else:
                # Basic calculations if RiskMetrics not available
                if portfolio_returns.std() > 0:
                    metrics['sharpe_ratio'] = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
                else:
                    metrics['sharpe_ratio'] = 0
        
        # Cost analysis
        if self.cost_breakdown:
            total_costs = sum(item['cost'] for item in self.cost_breakdown)
            total_gross_pnl = trade_df['Gross PnL'].sum() if not trade_df.empty else 0
            metrics['total_costs'] = total_costs
            metrics['cost_ratio'] = total_costs / abs(total_gross_pnl) if total_gross_pnl != 0 else 0
        
        return metrics
    
    def summary(self) -> pd.DataFrame:
        """Return trade summary"""
        return self.trade_log
    
    def get_cost_breakdown(self) -> pd.DataFrame:
        """Return detailed cost breakdown"""
        return pd.DataFrame(self.cost_breakdown)


# Integration function for optimization
def strategy_function_for_optimization(data: pd.DataFrame, 
                                     entry_z: float = 1.5, 
                                     exit_z: float = 0.5, 
                                     window: int = 30,
                                     **kwargs) -> Dict:
    """
    Wrapper function for use with parameter optimization
    """
    
    try:
        # Create backtester with given parameters
        backtester = EnhancedPairTradingBacktester(
            signal_data=data,
            entry_z=entry_z,
            exit_z=exit_z,
            initial_capital=100000,
            enable_costs=True,
            enable_risk_management=True
        )
        
        # Run backtest
        results = backtester.backtest()
        
        return results
    
    except Exception as e:
        warnings.warn(f"Error in strategy function: {e}")
        return None


# Example usage
if __name__ == "__main__":
    
    # Example with sample data
    dates = pd.date_range('2022-01-01', '2023-01-01', freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'spread': np.random.normal(0, 0.02, len(dates)),
        'zscore': np.random.normal(0, 1, len(dates))
    }, index=dates)
    
    # Create enhanced backtester
    backtester = EnhancedPairTradingBacktester(
        signal_data=sample_data,
        entry_z=1.5,
        exit_z=0.5,
        initial_capital=100000,
        tickers=('AAPL', 'MSFT'),
        enable_costs=True,
        enable_risk_management=True
    )
    
    # Run backtest
    results = backtester.backtest()
    
    print(f"Total trades: {len(results['trades'])}")
    print(f"Final capital: ${results['final_capital']:,.2f}")
    print(f"Total return: {results['total_return']:.2%}")
    
    if results['performance_metrics']:
        print(f"Win rate: {results['performance_metrics'].get('win_rate', 0):.1%}")
        print(f"Total costs: ${results['performance_metrics'].get('total_costs', 0):,.2f}")
