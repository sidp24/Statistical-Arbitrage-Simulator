import pandas as pd
class PairTradingBacktester:
    def __init__(self, path_to_signals, entry_z, exit_z, capital, tickers=None):
        self.path = path_to_signals
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.capital = capital
        self.tickers = tickers
        self.trade_log = pd.DataFrame()  # Initialize to avoid attribute errors

    def backtest(self):
        # Load signal data
        df = pd.read_csv(self.path, parse_dates=True, index_col=0)
        df["Position"] = 0
        position = 0

        trades = []

        for i in range(1, len(df)):
            z = df["zscore"].iloc[i]

            if position == 0:
                if z > self.entry_z:
                    position = -1
                    entry_idx = i
                    entry_spread = df["spread"].iloc[i]
                elif z < -self.entry_z:
                    position = 1
                    entry_idx = i
                    entry_spread = df["spread"].iloc[i]

            elif position != 0:
                if (position == -1 and z < self.exit_z) or (position == 1 and z > -self.exit_z):
                    exit_idx = i
                    exit_spread = df["spread"].iloc[i]

                    pnl = (entry_spread - exit_spread) * position
                    trades.append({
                        "Entry Date": df.index[entry_idx],
                        "Exit Date": df.index[exit_idx],
                        "PnL": pnl
                    })

                    position = 0

        # Convert trades to DataFrame
        trade_df = pd.DataFrame(trades)

        if not trade_df.empty:
            trade_df["Cumulative PnL"] = trade_df["PnL"].cumsum()

        self.trade_log = trade_df

    def summary(self):
        if not hasattr(self, "trade_log") or self.trade_log.empty:
            return pd.DataFrame(columns=["Entry Date", "Exit Date", "PnL", "Cumulative PnL"])

        return self.trade_log
