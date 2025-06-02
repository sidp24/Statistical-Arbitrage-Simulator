import yfinance as yf
import os
import pandas as pd

def download_prices(tickers, start="2022-01-01", end="2024-01-01", save_path="data/price_data.csv"):
    all_data = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=True)
    
    # Convert to a panel-like DataFrame: Date x Ticker columns
    price_df = pd.DataFrame(index=all_data.index)
    for ticker in tickers:
        price_df[ticker] = all_data[ticker]['Close']
    
    price_df.dropna(axis=1, how='any', inplace=True)  # drop assets with missing data
    price_df.to_csv(save_path)
    print(f"Saved {len(price_df.columns)} assets to {save_path}")

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JPM', 'V', 'UNH', 'NVDA', 'HD']  # Sample set
    os.makedirs("data", exist_ok=True)
    download_prices(tickers)
