# Data settings
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JPM', 'V', 'UNH', 'NVDA', 'HD']
START_DATE = "2022-01-01"
END_DATE = "2024-01-01"

# Cointegration
P_VALUE_THRESHOLD = 0.05

# Z-score calculation
ZSCORE_WINDOW = 30

# Trading strategy
ENTRY_Z = 1.0
EXIT_Z = 0.5
INITIAL_CAPITAL = 10000

# File paths
PRICE_DATA_PATH = "data/price_data.csv"
ZSCORE_OUTPUT_PATH = "data/zscore_signals.csv"
