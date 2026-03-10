"""
Logging Configuration for Statistical Arbitrage Simulator

Provides structured logging with file and console output,
with support for different log levels and formats.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import LOG_LEVEL, DEBUG


def setup_logger(
    name: str = "stat_arb",
    log_file: Optional[str] = None,
    level: Optional[str] = None
) -> logging.Logger:
    """
    Set up and return a configured logger.
    
    Args:
        name: Logger name
        log_file: Optional file path for log output
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Set level
    log_level = getattr(logging, (level or LOG_LEVEL).upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Create formatter
    if DEBUG:
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "stat_arb") -> logging.Logger:
    """Get or create a logger with the given name."""
    return logging.getLogger(name)


# Create default application logger
logger = setup_logger(
    name="stat_arb",
    log_file="logs/app.log" if not DEBUG else None
)


class TradingLogger:
    """Specialized logger for trading operations with structured output."""
    
    def __init__(self, name: str = "trading"):
        self.logger = setup_logger(f"stat_arb.{name}")
    
    def trade_entry(self, pair: str, direction: str, size: float, price: float):
        """Log trade entry."""
        self.logger.info(
            f"TRADE ENTRY | Pair: {pair} | Direction: {direction} | "
            f"Size: {size:.2f} | Price: {price:.4f}"
        )
    
    def trade_exit(self, pair: str, pnl: float, duration_days: int):
        """Log trade exit."""
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        self.logger.info(
            f"TRADE EXIT | Pair: {pair} | PnL: {pnl_str} | Duration: {duration_days} days"
        )
    
    def signal(self, pair: str, zscore: float, action: str):
        """Log trading signal."""
        self.logger.debug(
            f"SIGNAL | Pair: {pair} | Z-Score: {zscore:.3f} | Action: {action}"
        )
    
    def risk_alert(self, message: str, level: str = "WARNING"):
        """Log risk management alert."""
        log_func = getattr(self.logger, level.lower(), self.logger.warning)
        log_func(f"RISK ALERT | {message}")
    
    def backtest_result(self, pair: str, total_return: float, sharpe: float, trades: int):
        """Log backtest results."""
        self.logger.info(
            f"BACKTEST | Pair: {pair} | Return: {total_return:.2%} | "
            f"Sharpe: {sharpe:.3f} | Trades: {trades}"
        )


# Singleton trading logger
trading_logger = TradingLogger()
