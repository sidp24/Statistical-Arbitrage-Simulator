# Statistical Arbitrage Simulator

A Python framework for backtesting cointegration-based statistical arbitrage strategies with enhanced features including realistic transaction costs, risk management, and parameter optimization.

## Features

- **Strategy Development**: Cointegration testing and z-score based signals
- **Transaction Costs**: Realistic commission, spread, and slippage modeling
- **Risk Management**: Kelly criterion position sizing and portfolio controls
- **Parameter Optimization**: Walk-forward analysis and out-of-sample testing
- **Interactive Dashboard**: Streamlit-based web interface

## Quick Start

1. **Install Dependencies**: 
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**: 
   ```bash
   # Method 1: Direct command
   streamlit run app.py
   
   # Method 2: Using Python module
   python -m streamlit run app.py
   
   # Method 3: Windows batch file (Windows only)
   run_app.bat
   ```

3. **Access the Dashboard**: Open your browser to `http://localhost:8501`

4. **Run Analysis**: Use the sidebar to configure parameters and execute backtests

## Project Structure

```
Statistical-Arbitrage-Simulator/
├── app.py                         # Main Streamlit application
├── main.py                        # Command-line interface
├── config.py                      # Configuration settings
├── requirements.txt               # Python dependencies
├── run_app.bat                    # Windows launcher script
│
├── backtest/                      # Backtesting framework
│   ├── backtester.py              # Basic backtesting engine
│   └── enhanced_backtester.py     # Enhanced backtester with costs & risk
│
├── trading/                       # Transaction cost modeling
│   └── transaction_costs.py       # Cost simulation components
│
├── risk/                          # Risk management
│   └── risk_management.py         # Portfolio risk controls
│
├── optimization/                  # Parameter optimization
│   └── parameter_optimization.py  # Walk-forward analysis tools
│
├── data/                          # Data management
│   ├── download_data.py           # Price data downloading
│   └── signals/                   # Generated trading signals
│
├── analysis/                      # Performance analysis
│   └── metrics.py                 # Performance calculations
│
├── pairs/                         # Pair identification
│   └── find_pairs.py              # Cointegration testing
│
└── signals/                       # Signal generation
    └── zscore_signal.py           # Z-score signal calculation
```

## Usage Examples

### Basic Backtesting
```python
from backtest.enhanced_backtester import EnhancedPairTradingBacktester

# Run basic backtest
backtester = EnhancedPairTradingBacktester(
    path_to_signals="data/signals/AAPL_MSFT_signals.csv",
    entry_z=1.5,
    exit_z=0.5,
    initial_capital=100000
)

results = backtester.backtest()
```

### Enhanced Features
```python
# Enable transaction costs and risk management
backtester = EnhancedPairTradingBacktester(
    path_to_signals="data/signals/AAPL_MSFT_signals.csv",
    entry_z=1.5,
    exit_z=0.5,
    initial_capital=100000,
    tickers=('AAPL', 'MSFT'),
    enable_costs=True,
    enable_risk_management=True
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
