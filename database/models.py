"""
Database Models and ORM Layer for Statistical Arbitrage Simulator

Uses SQLAlchemy for database abstraction with support for SQLite and PostgreSQL.
"""
import os
from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from contextlib import contextmanager

from config import DATABASE_URL

# Create engine and session
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


@contextmanager
def get_db() -> Session:
    """Context manager for database sessions."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


class User(Base):
    """User model for authentication."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    backtests = relationship("BacktestResult", back_populates="user")
    alerts = relationship("Alert", back_populates="user")
    paper_trades = relationship("PaperTrade", back_populates="user")


class BacktestResult(Base):
    """Store backtest results for historical analysis."""
    __tablename__ = "backtest_results"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    pair = Column(String(50), nullable=False)
    ticker1 = Column(String(10), nullable=False)
    ticker2 = Column(String(10), nullable=False)
    
    # Parameters
    entry_z = Column(Float, nullable=False)
    exit_z = Column(Float, nullable=False)
    zscore_window = Column(Integer, nullable=False)
    initial_capital = Column(Float, nullable=False)
    
    # Date range
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    
    # Results
    total_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    total_trades = Column(Integer)
    win_rate = Column(Float)
    total_costs = Column(Float, default=0.0)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    parameters_json = Column(JSON, nullable=True)
    results_json = Column(JSON, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="backtests")
    trades = relationship("Trade", back_populates="backtest")


class Trade(Base):
    """Individual trade records."""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    backtest_id = Column(Integer, ForeignKey("backtest_results.id"), nullable=True)
    
    pair = Column(String(50), nullable=False)
    entry_date = Column(DateTime, nullable=False)
    exit_date = Column(DateTime, nullable=True)
    direction = Column(String(10), nullable=False)  # 'long' or 'short'
    
    entry_zscore = Column(Float)
    exit_zscore = Column(Float)
    entry_spread = Column(Float)
    exit_spread = Column(Float)
    
    position_size = Column(Float)
    pnl = Column(Float)
    costs = Column(Float, default=0.0)
    net_pnl = Column(Float)
    
    # Relationships
    backtest = relationship("BacktestResult", back_populates="trades")


class PaperTrade(Base):
    """Paper trading positions for simulation."""
    __tablename__ = "paper_trades"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    pair = Column(String(50), nullable=False)
    ticker1 = Column(String(10), nullable=False)
    ticker2 = Column(String(10), nullable=False)
    
    status = Column(String(20), default="open")  # open, closed
    direction = Column(String(10), nullable=False)
    
    entry_date = Column(DateTime, default=datetime.utcnow)
    exit_date = Column(DateTime, nullable=True)
    
    entry_zscore = Column(Float)
    current_zscore = Column(Float, nullable=True)
    
    entry_price1 = Column(Float)
    entry_price2 = Column(Float)
    current_price1 = Column(Float, nullable=True)
    current_price2 = Column(Float, nullable=True)
    
    position_size = Column(Float)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, nullable=True)
    
    # Target levels
    entry_z_threshold = Column(Float)
    exit_z_threshold = Column(Float)
    stop_loss_z = Column(Float, nullable=True)
    
    notes = Column(Text, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="paper_trades")


class Alert(Base):
    """Trading alerts and notifications."""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    pair = Column(String(50), nullable=False)
    alert_type = Column(String(50), nullable=False)  # zscore_entry, zscore_exit, price_target, etc.
    
    condition = Column(String(20), nullable=False)  # above, below, crosses
    threshold = Column(Float, nullable=False)
    
    is_active = Column(Boolean, default=True)
    is_triggered = Column(Boolean, default=False)
    triggered_at = Column(DateTime, nullable=True)
    
    notification_method = Column(String(20), default="email")  # email, sms, webhook
    notification_sent = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="alerts")


class WatchlistItem(Base):
    """User watchlist for pair monitoring."""
    __tablename__ = "watchlist"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    ticker1 = Column(String(10), nullable=False)
    ticker2 = Column(String(10), nullable=False)
    
    last_pvalue = Column(Float, nullable=True)
    last_zscore = Column(Float, nullable=True)
    last_checked = Column(DateTime, nullable=True)
    
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def drop_db():
    """Drop all database tables (use with caution!)."""
    Base.metadata.drop_all(bind=engine)


# Repository classes for CRUD operations
class BacktestRepository:
    """Repository for backtest operations."""
    
    @staticmethod
    def save_backtest(
        db: Session,
        pair: str,
        ticker1: str,
        ticker2: str,
        params: dict,
        results: dict,
        user_id: Optional[int] = None
    ) -> BacktestResult:
        """Save a backtest result to the database."""
        backtest = BacktestResult(
            user_id=user_id,
            pair=pair,
            ticker1=ticker1,
            ticker2=ticker2,
            entry_z=params.get("entry_z", 1.5),
            exit_z=params.get("exit_z", 0.5),
            zscore_window=params.get("zscore_window", 30),
            initial_capital=params.get("initial_capital", 100000),
            start_date=params.get("start_date", datetime.now()),
            end_date=params.get("end_date", datetime.now()),
            total_return=results.get("total_return", 0.0),
            sharpe_ratio=results.get("sharpe_ratio", 0.0),
            max_drawdown=results.get("max_drawdown", 0.0),
            total_trades=results.get("total_trades", 0),
            win_rate=results.get("win_rate", 0.0),
            total_costs=results.get("total_costs", 0.0),
            parameters_json=params,
            results_json=results,
        )
        db.add(backtest)
        db.flush()
        return backtest
    
    @staticmethod
    def get_user_backtests(db: Session, user_id: int, limit: int = 50) -> List[BacktestResult]:
        """Get recent backtests for a user."""
        return db.query(BacktestResult).filter(
            BacktestResult.user_id == user_id
        ).order_by(BacktestResult.created_at.desc()).limit(limit).all()
    
    @staticmethod
    def get_backtest_by_id(db: Session, backtest_id: int) -> Optional[BacktestResult]:
        """Get a specific backtest by ID."""
        return db.query(BacktestResult).filter(BacktestResult.id == backtest_id).first()


class AlertRepository:
    """Repository for alert operations."""
    
    @staticmethod
    def create_alert(
        db: Session,
        user_id: int,
        pair: str,
        alert_type: str,
        condition: str,
        threshold: float,
        notification_method: str = "email"
    ) -> Alert:
        """Create a new alert."""
        alert = Alert(
            user_id=user_id,
            pair=pair,
            alert_type=alert_type,
            condition=condition,
            threshold=threshold,
            notification_method=notification_method,
        )
        db.add(alert)
        db.flush()
        return alert
    
    @staticmethod
    def get_active_alerts(db: Session, user_id: Optional[int] = None) -> List[Alert]:
        """Get all active alerts."""
        query = db.query(Alert).filter(Alert.is_active == True)
        if user_id:
            query = query.filter(Alert.user_id == user_id)
        return query.all()
    
    @staticmethod
    def trigger_alert(db: Session, alert_id: int) -> Optional[Alert]:
        """Mark an alert as triggered."""
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if alert:
            alert.is_triggered = True
            alert.triggered_at = datetime.utcnow()
        return alert


# Initialize database on import
init_db()
