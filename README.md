# Statistical Arbitrage Simulator

A Python framework for backtesting cointegration-based statistical arbitrage (pairs trading) strategies. Includes realistic transaction costs, risk management, paper trading, and parameter optimization.

## Features

- **Strategy Development**: Cointegration testing and z-score based signal generation
- **Transaction Costs**: Realistic commission, spread, and slippage modeling
- **Risk Management**: Kelly criterion position sizing and portfolio controls
- **Parameter Optimization**: Walk-forward analysis and out-of-sample testing
- **Paper Trading**: Simulated trading with real-time price tracking
- **Alerts & Notifications**: Email notifications for trading signals
- **Interactive Dashboard**: Streamlit-based web interface with Plotly charts
- **Database Storage**: SQLite/PostgreSQL for storing backtests and trades
- **Authentication**: Optional user authentication with JWT tokens

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/sidp24/Statistical-Arbitrage-Simulator.git
cd Statistical-Arbitrage-Simulator

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings (optional)
```

### 3. Run the Application

```bash
streamlit run app.py
```

Access the dashboard at `http://localhost:8501`

## Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f app
```

### Manual Deployment

```bash
# Install production dependencies
pip install -r requirements.txt

# Run with Streamlit
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

### Environment Variables

Key configuration options (set in `.env`):

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | `sqlite:///data/arbitrage.db` |
| `AUTH_ENABLED` | Enable user authentication | `false` |
| `SECRET_KEY` | JWT secret key | (generate one) |
| `SMTP_HOST` | Email server for notifications | - |
| `LOG_LEVEL` | Logging level | `INFO` |

## Project Structure

```
Statistical-Arbitrage-Simulator/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration (env-based)
├── main.py                   # CLI interface
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Package configuration
├── Dockerfile                # Docker build file
├── docker-compose.yml        # Docker orchestration
├── .env.example              # Environment template
│
├── auth/                     # Authentication module
│   └── authentication.py     # JWT/bcrypt auth
│
├── backtest/                 # Backtesting framework
│   ├── backtester.py         # Basic backtester
│   └── enhanced_backtester.py # Enhanced with costs
│
├── database/                 # Database layer
│   ├── models.py             # SQLAlchemy models
│   └── init.sql              # PostgreSQL schema
│
├── trading/                  # Trading modules
│   ├── transaction_costs.py  # Cost modeling
│   └── paper_trading.py      # Paper trading engine
│
├── notifications/            # Alerts system
│   └── alerts.py             # Email/webhook alerts
│
├── risk/                     # Risk management
│   └── risk_management.py    # Position sizing, VAR
│
├── tests/                    # Test suite
│   ├── conftest.py           # Pytest fixtures
│   ├── test_backtester.py    # Backtester tests
│   └── ...
│
└── .github/workflows/        # CI/CD
    └── ci.yml                # GitHub Actions
```

## Usage Examples

### Basic Backtesting

```python
from backtest.enhanced_backtester import EnhancedPairTradingBacktester

backtester = EnhancedPairTradingBacktester(
    path_to_signals="data/signals/AAPL_MSFT_signals.csv",
    entry_z=1.5,
    exit_z=0.5,
    initial_capital=100000,
    enable_costs=True,
    enable_risk_management=True
)

results = backtester.backtest()
print(f"Total Return: {results['total_return']:.2%}")
```

### Paper Trading

```python
from trading.paper_trading import PaperTradingEngine

engine = PaperTradingEngine(initial_capital=100000)

# Open a position
position = engine.open_position("AAPL", "MSFT", "long_spread", 10000)

# Check for exit signals
exit_signals = engine.check_exit_signals()

# Get portfolio summary
summary = engine.get_portfolio_summary()
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_backtester.py -v
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `pytest tests/`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
)

results = backtester.backtest()
print(f"Total return: {results['total_return']:.2%}")
```

### Parameter Optimization
```python
from optimization.parameter_optimization import WalkForwardOptimizer

optimizer = WalkForwardOptimizer(
    strategy_function=your_strategy_function,
    parameter_ranges={
        'entry_z': [1.0, 1.5, 2.0, 2.5],
        'exit_z': [0.5, 0.75, 1.0]
    }
)

best_params = optimizer.optimize(data, walk_forward_months=6)
```

## Core Components

### Strategy Development
- Engle-Granger and Johansen cointegration tests
- Z-score based entry/exit signal generation
- Dynamic hedge ratio calculation
- Multiple timeframe analysis

### Transaction Cost Modeling
- Commission structures (per-share and minimum fees)
- Bid-ask spread costs (configurable basis points)
- Market impact (temporary and permanent)
- Short financing costs
- Execution slippage simulation

### Risk Management
- Kelly Criterion position sizing
- Volatility targeting
- Portfolio-level risk controls
- Correlation and concentration limits
- Drawdown protection mechanisms
- VaR and CVaR calculations

### Optimization Framework
- Grid search and random search methods
- Walk-forward analysis for robustness
- Out-of-sample testing validation
- Parameter stability assessment
- Cross-validation framework

## Performance Analytics

- **Return Metrics**: Sharpe, Calmar, Sortino ratios
- **Risk Metrics**: Maximum drawdown, volatility, VaR
- **Trade Analytics**: Win rate, profit factor, average trade metrics
- **Cost Analysis**: Detailed transaction cost attribution
- **Visual Analytics**: PnL charts, spread analysis, signal visualization

## Configuration

Key parameters in `config.py`:
- `TICKERS`: Available securities for analysis
- `INITIAL_CAPITAL`: Default starting capital
- `P_VALUE_THRESHOLD`: Cointegration significance level
- `PRICE_DATA_PATH`: Data storage location

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computing
- matplotlib/seaborn: Visualization
- streamlit: Web interface
- yfinance: Market data
- scipy: Scientific computing
- statsmodels: Statistical analysis

## Dashboard Features

The Streamlit dashboard provides three main sections:

1. **Backtesting**: Configure and run strategy backtests
2. **Testing & Validation**: System health checks and demos
3. **Configuration**: View system settings and module status

### Backtesting Interface
- Ticker selection and date range configuration
- Entry/exit z-score threshold adjustment
- Enhanced features toggle (costs and risk management)
- Real-time results display with performance metrics
- Interactive charts for PnL and spread analysis

### Testing Interface
- System component health checks
- Demo backtests with synthetic data
- Module availability verification
- Performance comparison (basic vs enhanced)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is available under the MIT License.

## Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Trading involves substantial risk and may not be suitable for all investors. Use at your own discretion.
