import pandas as pd

class PairTradingBacktester:
    def __init__(self, csv_path, entry_z=1.0, exit_z=0.5, capital=10000, tickers=None):
        self.data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.capital = capital
        self.positions = []
        self.trades = []
        self.t1, self.t2 = tickers if tickers else ("AAPL", "MSFT")  # Default fallback

    def backtest(self):
        in_position = False
        entry_date = None
        entry_zscore = None

        for i in range(len(self.data)):
            z = self.data["Z-Score"].iloc[i]
            spread = self.data["Spread"].iloc[i]
            date = self.data.index[i]
            t1_price = self.data[self.t1].iloc[i]
            t2_price = self.data[self.t2].iloc[i]

            if not in_position:
                if z > self.entry_z:
                    # SHORT t1, LONG t2
                    self.positions.append(("SHORT", date, t1_price, t2_price, z))
                    in_position = True
                    entry_date = date
                    entry_zscore = z

                elif z < -self.entry_z:
                    # LONG t1, SHORT t2
                    self.positions.append(("LONG", date, t1_price, t2_price, z))
                    in_position = True
                    entry_date = date
                    entry_zscore = z

            elif in_position:
                if abs(z) < self.exit_z:
                    exit_date = date
                    exit_t1 = t1_price
                    exit_t2 = t2_price
                    direction, entry_date, entry_t1, entry_t2, _ = self.positions.pop()
                    pnl = self._calculate_pnl(direction, entry_t1, entry_t2, exit_t1, exit_t2)
                    self.trades.append({
                        "Direction": direction,
                        "Entry Date": entry_date,
                        "Exit Date": exit_date,
                        "PnL": pnl,
                        "Entry Z": entry_zscore,
                        "Exit Z": z
                    })
                    in_position = False

    def _calculate_pnl(self, direction, entry_t1, entry_t2, exit_t1, exit_t2):
        if direction == "LONG":
            return (exit_t1 - entry_t1) - (exit_t2 - entry_t2)
        elif direction == "SHORT":
            return (entry_t1 - exit_t1) - (entry_t2 - exit_t2)

    def summary(self):
        df = pd.DataFrame(self.trades)
        df["Cumulative PnL"] = df["PnL"].cumsum()
        print(df[["Entry Date", "Exit Date", "Direction", "PnL", "Cumulative PnL"]])
        print("\nTotal Return:", df["PnL"].sum())
        return df
