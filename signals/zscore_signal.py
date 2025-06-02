import pandas as pd
import numpy as np
import statsmodels.api as sm

def compute_zscore_spread(series1, series2, window=30, ticker1="Asset1", ticker2="Asset2"):
    # OLS regression to find hedge ratio
    X = sm.add_constant(series2)
    model = sm.OLS(series1, X).fit()
    hedge_ratio = model.params.iloc[1]  # avoids the FutureWarning

    spread = series1 - hedge_ratio * series2
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    zscore = (spread - rolling_mean) / rolling_std

    df = pd.DataFrame({
        ticker1: series1,
        ticker2: series2,
        "Spread": spread,
        "Z-Score": zscore
    })

    return zscore, spread, hedge_ratio, df
