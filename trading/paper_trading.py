import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

from database.models import PaperTrade, get_db, Session
from utils.logging_config import trading_logger


@dataclass
class Position:
    pair: str
    ticker1: str
    ticker2: str
    direction: str  # 'long_spread' or 'short_spread'
    entry_date: datetime
    entry_zscore: float
    entry_price1: float
    entry_price2: float
    position_size: float
    entry_z_threshold: float
    exit_z_threshold: float
    stop_loss_z: Optional[float] = None
    
    # Current state
    current_price1: float = 0.0
    current_price2: float = 0.0
    current_zscore: float = 0.0
    unrealized_pnl: float = 0.0


class PaperTradingEngine:
    def __init__(
        self,
        initial_capital: float = 100000,
        entry_z: float = 1.5,
        exit_z: float = 0.5,
        stop_loss_z: float = 3.0,
        max_position_pct: float = 0.1,
        zscore_window: int = 30
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_loss_z = stop_loss_z
        self.max_position_pct = max_position_pct
        self.zscore_window = zscore_window
        
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Dict] = []
        self.trade_history: List[Dict] = []
    
    def get_current_prices(self, ticker1: str, ticker2: str) -> Tuple[float, float]:
        if not YF_AVAILABLE:
            # Return dummy prices for demo
            return 100.0, 100.0
        
        try:
            data = yf.download([ticker1, ticker2], period="1d", progress=False)
            if len(data) > 0:
                price1 = data['Close'][ticker1].iloc[-1]
                price2 = data['Close'][ticker2].iloc[-1]
                return float(price1), float(price2)
        except Exception as e:
            trading_logger.logger.error(f"Error fetching prices: {e}")
        
        return 0.0, 0.0
    
    def calculate_zscore(self, ticker1: str, ticker2: str) -> Tuple[float, float, float]:
        if not YF_AVAILABLE:
            return 0.0, 0.0, 1.0
        
        try:
            # Get historical data
            data = yf.download(
                [ticker1, ticker2],
                period=f"{self.zscore_window * 2}d",
                progress=False
            )
            
            if len(data) < self.zscore_window:
                return 0.0, 0.0, 1.0
            
            s1 = np.log(data['Close'][ticker1])
            s2 = np.log(data['Close'][ticker2])
            
            # Calculate hedge ratio using OLS
            from scipy import stats
            slope, intercept, _, _, _ = stats.linregress(s2, s1)
            hedge_ratio = slope
            
            # Calculate spread
            spread = s1 - hedge_ratio * s2
            
            # Calculate z-score
            spread_mean = spread.rolling(window=self.zscore_window).mean().iloc[-1]
            spread_std = spread.rolling(window=self.zscore_window).std().iloc[-1]
            
            if spread_std > 0:
                zscore = (spread.iloc[-1] - spread_mean) / spread_std
            else:
                zscore = 0.0
            
            return float(zscore), float(spread.iloc[-1]), float(hedge_ratio)
            
        except Exception as e:
            trading_logger.logger.error(f"Error calculating z-score: {e}")
            return 0.0, 0.0, 1.0
    
    def check_entry_signals(self, watchlist: List[Tuple[str, str]]) -> List[Dict]:
        signals = []
        
        for ticker1, ticker2 in watchlist:
            pair_key = f"{ticker1}_{ticker2}"
            
            # Skip if already in position
            if pair_key in self.positions:
                continue
            
            zscore, spread, hedge_ratio = self.calculate_zscore(ticker1, ticker2)
            
            signal = {
                "pair": pair_key,
                "ticker1": ticker1,
                "ticker2": ticker2,
                "zscore": zscore,
                "spread": spread,
                "hedge_ratio": hedge_ratio,
                "signal": None,
                "timestamp": datetime.now()
            }
            
            if zscore > self.entry_z:
                signal["signal"] = "short_spread"
                signal["action"] = f"SHORT {ticker1}, LONG {ticker2}"
                signals.append(signal)
                trading_logger.signal(pair_key, zscore, "SHORT SPREAD")
                
            elif zscore < -self.entry_z:
                signal["signal"] = "long_spread"
                signal["action"] = f"LONG {ticker1}, SHORT {ticker2}"
                signals.append(signal)
                trading_logger.signal(pair_key, zscore, "LONG SPREAD")
        
        return signals
    
    def check_exit_signals(self) -> List[Dict]:
        exit_signals = []
        
        for pair_key, position in self.positions.items():
            zscore, _, _ = self.calculate_zscore(position.ticker1, position.ticker2)
            position.current_zscore = zscore
            
            should_exit = False
            exit_reason = ""
            
            # Check exit conditions
            if position.direction == "long_spread":
                if zscore >= -self.exit_z:
                    should_exit = True
                    exit_reason = "Target reached"
                elif self.stop_loss_z and zscore < -self.stop_loss_z:
                    should_exit = True
                    exit_reason = "Stop loss"
            else:  # short_spread
                if zscore <= self.exit_z:
                    should_exit = True
                    exit_reason = "Target reached"
                elif self.stop_loss_z and zscore > self.stop_loss_z:
                    should_exit = True
                    exit_reason = "Stop loss"
            
            if should_exit:
                exit_signals.append({
                    "pair": pair_key,
                    "position": position,
                    "current_zscore": zscore,
                    "reason": exit_reason
                })
                trading_logger.signal(pair_key, zscore, f"EXIT ({exit_reason})")
        
        return exit_signals
    
    def open_position(
        self,
        ticker1: str,
        ticker2: str,
        direction: str,
        position_size: Optional[float] = None
    ) -> Position:
        pair_key = f"{ticker1}_{ticker2}"
        
        if pair_key in self.positions:
            raise ValueError(f"Position already exists for {pair_key}")
        
        # Get current prices and z-score
        price1, price2 = self.get_current_prices(ticker1, ticker2)
        zscore, spread, hedge_ratio = self.calculate_zscore(ticker1, ticker2)
        
        # Calculate position size
        if position_size is None:
            position_size = self.current_capital * self.max_position_pct
        
        position = Position(
            pair=pair_key,
            ticker1=ticker1,
            ticker2=ticker2,
            direction=direction,
            entry_date=datetime.now(),
            entry_zscore=zscore,
            entry_price1=price1,
            entry_price2=price2,
            position_size=position_size,
            entry_z_threshold=self.entry_z,
            exit_z_threshold=self.exit_z,
            stop_loss_z=self.stop_loss_z,
            current_price1=price1,
            current_price2=price2,
            current_zscore=zscore
        )
        
        self.positions[pair_key] = position
        
        trading_logger.trade_entry(
            pair_key, direction, position_size, 
            (price1 + price2) / 2  # Average price for logging
        )
        
        return position
    
    def close_position(self, pair_key: str) -> Dict:
        if pair_key not in self.positions:
            raise ValueError(f"No position found for {pair_key}")
        
        position = self.positions[pair_key]
        
        # Get current prices
        price1, price2 = self.get_current_prices(position.ticker1, position.ticker2)
        
        # Calculate P&L
        if position.direction == "long_spread":
            # Long spread = long ticker1, short ticker2
            pnl1 = (price1 - position.entry_price1) / position.entry_price1
            pnl2 = (position.entry_price2 - price2) / position.entry_price2
        else:
            # Short spread = short ticker1, long ticker2
            pnl1 = (position.entry_price1 - price1) / position.entry_price1
            pnl2 = (price2 - position.entry_price2) / position.entry_price2
        
        total_pnl_pct = (pnl1 + pnl2) / 2
        realized_pnl = position.position_size * total_pnl_pct
        
        # Update capital
        self.current_capital += realized_pnl
        
        # Record trade
        trade_record = {
            "pair": pair_key,
            "ticker1": position.ticker1,
            "ticker2": position.ticker2,
            "direction": position.direction,
            "entry_date": position.entry_date,
            "exit_date": datetime.now(),
            "entry_zscore": position.entry_zscore,
            "exit_zscore": position.current_zscore,
            "entry_price1": position.entry_price1,
            "entry_price2": position.entry_price2,
            "exit_price1": price1,
            "exit_price2": price2,
            "position_size": position.position_size,
            "realized_pnl": realized_pnl,
            "return_pct": total_pnl_pct
        }
        
        self.closed_trades.append(trade_record)
        
        # Remove position
        del self.positions[pair_key]
        
        duration = (trade_record["exit_date"] - trade_record["entry_date"]).days
        trading_logger.trade_exit(pair_key, realized_pnl, duration)
        
        return trade_record
    
    def update_positions(self):
        for pair_key, position in self.positions.items():
            price1, price2 = self.get_current_prices(position.ticker1, position.ticker2)
            zscore, _, _ = self.calculate_zscore(position.ticker1, position.ticker2)
            
            position.current_price1 = price1
            position.current_price2 = price2
            position.current_zscore = zscore
            
            # Calculate unrealized P&L
            if position.direction == "long_spread":
                pnl1 = (price1 - position.entry_price1) / position.entry_price1
                pnl2 = (position.entry_price2 - price2) / position.entry_price2
            else:
                pnl1 = (position.entry_price1 - price1) / position.entry_price1
                pnl2 = (price2 - position.entry_price2) / position.entry_price2
            
            total_pnl_pct = (pnl1 + pnl2) / 2
            position.unrealized_pnl = position.position_size * total_pnl_pct
    
    def get_portfolio_summary(self) -> Dict:
        self.update_positions()
        
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        total_realized = sum(t["realized_pnl"] for t in self.closed_trades)
        
        return {
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "total_unrealized_pnl": total_unrealized,
            "total_realized_pnl": total_realized,
            "total_pnl": total_unrealized + total_realized,
            "total_return": (self.current_capital + total_unrealized - self.initial_capital) / self.initial_capital,
            "open_positions": len(self.positions),
            "closed_trades": len(self.closed_trades),
            "positions": list(self.positions.values()),
            "recent_trades": self.closed_trades[-10:] if self.closed_trades else []
        }
    
    def save_to_database(self, user_id: int):
        with get_db() as db:
            # Save open positions
            for pair_key, position in self.positions.items():
                paper_trade = PaperTrade(
                    user_id=user_id,
                    pair=position.pair,
                    ticker1=position.ticker1,
                    ticker2=position.ticker2,
                    status="open",
                    direction=position.direction,
                    entry_date=position.entry_date,
                    entry_zscore=position.entry_zscore,
                    current_zscore=position.current_zscore,
                    entry_price1=position.entry_price1,
                    entry_price2=position.entry_price2,
                    current_price1=position.current_price1,
                    current_price2=position.current_price2,
                    position_size=position.position_size,
                    unrealized_pnl=position.unrealized_pnl,
                    entry_z_threshold=position.entry_z_threshold,
                    exit_z_threshold=position.exit_z_threshold,
                    stop_loss_z=position.stop_loss_z
                )
                db.add(paper_trade)


# Singleton instance
paper_trading_engine = PaperTradingEngine()
