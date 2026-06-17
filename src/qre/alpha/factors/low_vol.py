import pandas as pd

from qre.alpha.base import Alpha


class LowVol(Alpha):
    """Low-volatility anomaly factor (Ang, Hodrick, Xing & Zhang, 2006).

    Captures the low-volatility anomaly: stocks with low trailing return
    volatility have historically earned higher risk-adjusted returns than
    high-volatility names. (The related "betting against beta" effect of
    Frazzini & Pedersen, 2014, is framed on market beta rather than the
    total return volatility used here.)

    Formula: -rolling_std(daily_returns, window)
    Signal: positive = low recent volatility (buy), negative = high (sell).
    Timing: uses prices up to close of date t; no lookahead.
    NaN warmup: first `window + 1` rows are NaN (1 for returns, window for std).
    """

    def __init__(self, window: int = 21) -> None:
        super().__init__(name=f"low_volatility_{window}", lookback=window + 1)
        self.window = window

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change()
        return -returns.rolling(self.window).std()