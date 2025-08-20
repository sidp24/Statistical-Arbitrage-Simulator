try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    print("Warning: yfinance not available. Please install with: pip install yfinance")

import os
import pandas as pd

def download_prices(tickers, start="2022-01-01", end="2024-01-01", save_path="data/price_data.csv"):
    if not YF_AVAILABLE:
        raise ImportError("yfinance is required for downloading data. Please install with: pip install yfinance")
    
    all_data = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=True)
    
    # Convert to a panel-like DataFrame: Date x Ticker columns
    price_df = pd.DataFrame(index=all_data.index)
    for ticker in tickers:
        price_df[ticker] = all_data[ticker]['Close']
    
    price_df.dropna(axis=1, how='any', inplace=True)  # drop assets with missing data
    price_df.to_csv(save_path)
    print(f"Saved {len(price_df.columns)} assets to {save_path}")

def create_sample_data(tickers, start="2022-01-01", end="2024-01-01", save_path="data/price_data.csv"):
    """Create sample data for demonstration when yfinance is not available"""
    import numpy as np
    
    date_range = pd.date_range(start=start, end=end, freq='D')
    np.random.seed(42)  # For reproducible results
    
    # Create sample price data with random walk
    price_data = {}
    for ticker in tickers:
        # Start with a base price around 100
        base_price = 100
        returns = np.random.normal(0.001, 0.02, len(date_range))  # Daily returns
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        price_data[ticker] = prices
    
    price_df = pd.DataFrame(price_data, index=date_range)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    price_df.to_csv(save_path)
    print(f"Created sample data for {len(price_df.columns)} assets and saved to {save_path}")
    return price_df

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JPM', 'V', 'UNH', 'NVDA', 'HD']  # Sample set
    os.makedirs("data", exist_ok=True)
    download_prices(tickers)
