import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from itertools import combinations

def find_cointegrated_pairs(price_df, significance=0.05):
    n = price_df.shape[1]
    pairs = []
    pvals = np.ones((n, n))
    tickers = price_df.columns

    for i, j in combinations(range(n), 2):
        series1 = np.log(price_df[tickers[i]])
        series2 = np.log(price_df[tickers[j]])
        score, pval, _ = coint(series1, series2)

        pvals[i, j] = pval
        if pval < significance:
            pairs.append((tickers[i], tickers[j], pval))

    return pairs, pvals

if __name__ == "__main__":
    df = pd.read_csv("data/price_data.csv", index_col=0, parse_dates=True)
    pairs, _ = find_cointegrated_pairs(df)

    print("\nCointegrated pairs (p < 0.05):")
    for pair in sorted(pairs, key=lambda x: x[2]):
        print(f"{pair[0]} & {pair[1]} — p = {pair[2]:.4f}")
