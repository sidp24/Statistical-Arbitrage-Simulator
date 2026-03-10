-- PostgreSQL initialization script for Statistical Arbitrage Simulator
-- This runs when the Docker container is first created

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- Backtest results table
CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    pair VARCHAR(50) NOT NULL,
    ticker1 VARCHAR(10) NOT NULL,
    ticker2 VARCHAR(10) NOT NULL,
    entry_z FLOAT NOT NULL,
    exit_z FLOAT NOT NULL,
    zscore_window INTEGER NOT NULL,
    initial_capital FLOAT NOT NULL,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    total_return FLOAT,
    sharpe_ratio FLOAT,
    max_drawdown FLOAT,
    total_trades INTEGER,
    win_rate FLOAT,
    total_costs FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    parameters_json JSONB,
    results_json JSONB
);

-- Trades table
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    backtest_id INTEGER REFERENCES backtest_results(id),
    pair VARCHAR(50) NOT NULL,
    entry_date TIMESTAMP NOT NULL,
    exit_date TIMESTAMP,
    direction VARCHAR(10) NOT NULL,
    entry_zscore FLOAT,
    exit_zscore FLOAT,
    entry_spread FLOAT,
    exit_spread FLOAT,
    position_size FLOAT,
    pnl FLOAT,
    costs FLOAT DEFAULT 0.0,
    net_pnl FLOAT
);

-- Paper trades table
CREATE TABLE IF NOT EXISTS paper_trades (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) NOT NULL,
    pair VARCHAR(50) NOT NULL,
    ticker1 VARCHAR(10) NOT NULL,
    ticker2 VARCHAR(10) NOT NULL,
    status VARCHAR(20) DEFAULT 'open',
    direction VARCHAR(10) NOT NULL,
    entry_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    exit_date TIMESTAMP,
    entry_zscore FLOAT,
    current_zscore FLOAT,
    entry_price1 FLOAT,
    entry_price2 FLOAT,
    current_price1 FLOAT,
    current_price2 FLOAT,
    position_size FLOAT,
    unrealized_pnl FLOAT DEFAULT 0.0,
    realized_pnl FLOAT,
    entry_z_threshold FLOAT,
    exit_z_threshold FLOAT,
    stop_loss_z FLOAT,
    notes TEXT
);

-- Alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) NOT NULL,
    pair VARCHAR(50) NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    condition VARCHAR(20) NOT NULL,
    threshold FLOAT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_triggered BOOLEAN DEFAULT FALSE,
    triggered_at TIMESTAMP,
    notification_method VARCHAR(20) DEFAULT 'email',
    notification_sent BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

-- Watchlist table
CREATE TABLE IF NOT EXISTS watchlist (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) NOT NULL,
    ticker1 VARCHAR(10) NOT NULL,
    ticker2 VARCHAR(10) NOT NULL,
    last_pvalue FLOAT,
    last_zscore FLOAT,
    last_checked TIMESTAMP,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, ticker1, ticker2)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_backtest_user ON backtest_results(user_id);
CREATE INDEX IF NOT EXISTS idx_backtest_pair ON backtest_results(pair);
CREATE INDEX IF NOT EXISTS idx_trades_backtest ON trades(backtest_id);
CREATE INDEX IF NOT EXISTS idx_paper_trades_user ON paper_trades(user_id);
CREATE INDEX IF NOT EXISTS idx_alerts_user ON alerts(user_id);
CREATE INDEX IF NOT EXISTS idx_alerts_active ON alerts(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_watchlist_user ON watchlist(user_id);

-- Insert default admin user (password: changeme123)
-- Hash generated with bcrypt, rounds=12
INSERT INTO users (email, username, hashed_password, is_admin)
VALUES (
    'admin@example.com',
    'admin',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4.QJQQ5UQ3P8d.Hy',
    TRUE
) ON CONFLICT (email) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
