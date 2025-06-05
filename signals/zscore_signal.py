import pandas as pd
import numpy as np
import statsmodels.api as sm

def compute_zscore_spread(s1, s2, window=30, ticker1=None, ticker2=None):
    hedge = np.polyfit(s2, s1, 1)[0]
    spread = s1 - hedge * s2
    z = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()

    df = pd.DataFrame({
        "spread": spread,
        "zscore": z
    }, index=s1.index)

    return z, spread, hedge, df
