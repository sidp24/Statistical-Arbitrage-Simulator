"""
Enhanced Risk Management for Statistical Arbitrage Portfolio

This module implements risk management including:
- Portfolio-level risk controls
- Position sizing using Kelly criterion and volatility targeting
- Correlation limits and sector concentration limits
- Dynamic risk adjustment
- VaR and CVaR calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize_scalar
import warnings


class RiskMetrics:
    """Calculate various risk metrics for portfolio management"""
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk"""
        if len(returns) < 10:
            return 0.0
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = RiskMetrics.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_max_drawdown(cumulative_returns: pd.Series) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration"""
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_dd = drawdown.min()
        
        # Calculate drawdown duration
        dd_duration = 0
        current_dd = 0
        max_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_dd += 1
                max_duration = max(max_duration, current_dd)
            else:
                current_dd = 0
        
        return max_dd, max_duration
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series) -> float:
        """Calculate Calmar ratio (Annual return / Max Drawdown)"""
        if len(returns) == 0:
            return 0.0
        
        annual_return = returns.mean() * 252
        cumulative = (1 + returns).cumprod()
        max_dd, _ = RiskMetrics.calculate_max_drawdown(cumulative)
        
        if max_dd == 0:
            return float('inf')
        return annual_return / abs(max_dd)


class PositionSizer:
    """Advanced position sizing using various methodologies"""
    
    def __init__(self, 
                 max_position_pct: float = 0.10,    # Max 10% per position
                 max_sector_pct: float = 0.30,      # Max 30% per sector
                 volatility_target: float = 0.15,   # 15% annual volatility target
                 kelly_fraction: float = 0.25):     # Use 25% of Kelly size
        
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.volatility_target = volatility_target
        self.kelly_fraction = kelly_fraction
    
    def kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly fraction for position sizing
        Kelly = (bp - q) / b
        where: b = odds received (avg_win/avg_loss), p = win rate, q = loss rate
        """
        if avg_loss >= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        b = abs(avg_win / avg_loss)  # Odds ratio
        p = win_rate
        q = 1 - win_rate
        
        kelly_f = (b * p - q) / b
        return max(0, min(kelly_f * self.kelly_fraction, self.max_position_pct))
    
    def volatility_targeting(self, 
                           strategy_vol: float, 
                           portfolio_vol: float,
                           current_capital: float) -> float:
        """
        Calculate position size based on volatility targeting
        """
        if strategy_vol <= 0 or portfolio_vol <= 0:
            return 0.0
        
        # Target leverage to achieve desired portfolio volatility
        vol_scalar = self.volatility_target / strategy_vol
        position_size = current_capital * vol_scalar
        
        # Apply maximum position size constraint
        max_position = current_capital * self.max_position_pct
        return min(position_size, max_position)
    
    def risk_parity_sizing(self, 
                          strategy_vol: float, 
                          correlations: Dict[str, float],
                          current_positions: Dict[str, float]) -> float:
        """
        Size position to contribute equally to portfolio risk
        """
        if strategy_vol <= 0:
            return 0.0
        
        # Simplified risk parity: inverse volatility weighting
        total_inv_vol = sum(1/vol for vol in correlations.values() if vol > 0)
        if total_inv_vol == 0:
            return 0.0
        
        weight = (1/strategy_vol) / total_inv_vol
        return weight * self.max_position_pct


class RiskManager:
    """
    Portfolio-level risk management system
    """
    
    def __init__(self,
                 max_portfolio_var: float = 0.02,     # Max 2% daily VaR
                 max_correlation: float = 0.7,        # Max correlation between strategies
                 max_drawdown_limit: float = 0.15,    # Stop if drawdown > 15%
                 position_sizer: Optional[PositionSizer] = None):
        
        self.max_portfolio_var = max_portfolio_var
        self.max_correlation = max_correlation
        self.max_drawdown_limit = max_drawdown_limit
        self.position_sizer = position_sizer or PositionSizer()
        
        # Portfolio state
        self.portfolio_returns = []
        self.positions = {}
        self.sector_exposure = {}
        self.strategy_correlations = {}
    
    def check_correlation_limit(self, 
                              new_strategy_returns: pd.Series,
                              existing_strategies: Dict[str, pd.Series]) -> bool:
        """Check if adding new strategy violates correlation limits"""
        
        for strategy_name, returns in existing_strategies.items():
            if len(returns) > 10 and len(new_strategy_returns) > 10:
                # Align series for correlation calculation
                common_dates = returns.index.intersection(new_strategy_returns.index)
                if len(common_dates) > 10:
                    corr = returns.loc[common_dates].corr(new_strategy_returns.loc[common_dates])
                    if abs(corr) > self.max_correlation:
                        return False
        return True
    
    def check_sector_limits(self, 
                          new_position_sector: str, 
                          new_position_size: float) -> bool:
        """Check if new position violates sector concentration limits"""
        
        current_sector_exposure = self.sector_exposure.get(new_position_sector, 0)
        projected_exposure = current_sector_exposure + abs(new_position_size)
        
        return projected_exposure <= self.position_sizer.max_sector_pct
    
    def calculate_portfolio_var(self, confidence_level: float = 0.05) -> float:
        """Calculate portfolio Value at Risk"""
        if len(self.portfolio_returns) < 10:
            return 0.0
        
        returns_series = pd.Series(self.portfolio_returns)
        return RiskMetrics.calculate_var(returns_series, confidence_level)
    
    def check_drawdown_limit(self) -> bool:
        """Check if current drawdown exceeds limits"""
        if len(self.portfolio_returns) < 2:
            return True
        
        cumulative = pd.Series(self.portfolio_returns).cumsum()
        max_dd, _ = RiskMetrics.calculate_max_drawdown(cumulative)
        
        return abs(max_dd) <= self.max_drawdown_limit
    
    def validate_new_position(self,
                            strategy_name: str,
                            position_size: float,
                            sector: str,
                            strategy_returns: pd.Series,
                            existing_strategies: Dict[str, pd.Series]) -> Dict[str, bool]:
        """
        Comprehensive validation of new position against all risk limits
        """
        
        checks = {
            'correlation_check': self.check_correlation_limit(strategy_returns, existing_strategies),
            'sector_check': self.check_sector_limits(sector, position_size),
            'drawdown_check': self.check_drawdown_limit(),
            'var_check': True  # Will be calculated after position is added
        }
        
        # Simulate adding position to check VaR
        if all(checks.values()):
            # Temporarily add position to calculate projected VaR
            temp_returns = self.portfolio_returns.copy()
            if len(strategy_returns) > 0:
                temp_returns.extend(strategy_returns.iloc[-5:].tolist())  # Add recent returns
            
            temp_var = RiskMetrics.calculate_var(pd.Series(temp_returns))
            checks['var_check'] = abs(temp_var) <= self.max_portfolio_var
        
        checks['overall_approved'] = all(checks.values())
        return checks
    
    def calculate_optimal_position_size(self,
                                      strategy_returns: pd.Series,
                                      current_capital: float,
                                      strategy_metadata: Dict) -> float:
        """
        Calculate optimal position size using multiple methodologies
        """
        
        if len(strategy_returns) < 20:  # Need minimum history
            return current_capital * 0.01  # Very small position
        
        # Method 1: Kelly Criterion
        returns = strategy_returns.dropna()
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) > 0 and len(losses) > 0:
            win_rate = len(wins) / len(returns)
            avg_win = wins.mean()
            avg_loss = losses.mean()
            kelly_size = self.position_sizer.kelly_criterion(win_rate, avg_win, avg_loss)
        else:
            kelly_size = 0.01
        
        # Method 2: Volatility Targeting
        strategy_vol = returns.std() * np.sqrt(252)
        portfolio_vol = pd.Series(self.portfolio_returns).std() * np.sqrt(252) if self.portfolio_returns else strategy_vol
        vol_size = self.position_sizer.volatility_targeting(strategy_vol, portfolio_vol, current_capital)
        
        # Use more conservative of the two approaches
        optimal_size = min(kelly_size * current_capital, vol_size)
        
        return max(optimal_size, current_capital * 0.005)  # Minimum 0.5% position
    
    def update_portfolio_state(self, 
                             new_return: float,
                             position_updates: Dict[str, float],
                             sector_updates: Dict[str, float]):
        """Update internal portfolio state"""
        
        self.portfolio_returns.append(new_return)
        
        # Keep only recent history to avoid memory issues
        if len(self.portfolio_returns) > 1000:
            self.portfolio_returns = self.portfolio_returns[-500:]
        
        # Update positions and sector exposure
        for position, size in position_updates.items():
            self.positions[position] = size
        
        for sector, exposure in sector_updates.items():
            self.sector_exposure[sector] = exposure
    
    def generate_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        
        if len(self.portfolio_returns) < 10:
            return {"error": "Insufficient data for risk analysis"}
        
        returns_series = pd.Series(self.portfolio_returns)
        cumulative = returns_series.cumsum()
        
        report = {
            'portfolio_metrics': {
                'total_return': cumulative.iloc[-1],
                'sharpe_ratio': RiskMetrics.calculate_sharpe_ratio(returns_series),
                'calmar_ratio': RiskMetrics.calculate_calmar_ratio(returns_series),
                'volatility': returns_series.std() * np.sqrt(252),
                'var_95': RiskMetrics.calculate_var(returns_series, 0.05),
                'cvar_95': RiskMetrics.calculate_cvar(returns_series, 0.05)
            },
            'risk_limits': {
                'max_drawdown': self.max_drawdown_limit,
                'current_drawdown': RiskMetrics.calculate_max_drawdown(cumulative)[0],
                'max_var': self.max_portfolio_var,
                'current_var': abs(RiskMetrics.calculate_var(returns_series, 0.05))
            },
            'position_summary': {
                'active_positions': len(self.positions),
                'sector_exposure': self.sector_exposure,
                'total_exposure': sum(abs(pos) for pos in self.positions.values())
            }
        }
        
        return report


# Example usage
if __name__ == "__main__":
    # Initialize risk management system
    position_sizer = PositionSizer(max_position_pct=0.05, volatility_target=0.12)
    risk_manager = RiskManager(position_sizer=position_sizer)
    
    # Simulate some strategy returns
    np.random.seed(42)
    strategy_returns = pd.Series(np.random.normal(0.0005, 0.02, 100))  # Daily returns
    
    # Calculate optimal position size
    optimal_size = risk_manager.calculate_optimal_position_size(
        strategy_returns, 
        current_capital=100000,
        strategy_metadata={'sector': 'Technology'}
    )
    
    print(f"Optimal position size: ${optimal_size:,.2f}")
    print(f"Risk report: {risk_manager.generate_risk_report()}")
