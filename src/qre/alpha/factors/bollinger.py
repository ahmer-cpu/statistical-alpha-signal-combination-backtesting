import pandas as pd

from qre.alpha.base import Alpha


class BollingerZScore(Alpha):
    """Bollinger band mean-reversion signal (Bollinger, 2001).

    Computes the z-score of price relative to its rolling mean and
    standard deviation, then negates so the signal follows the base
    class convention: below the band = buy.

    Formula: -(price - SMA) / rolling_std
    Signal: positive = below band (buy), negative = above band (sell).
    Timing: uses prices up to close of date t; no lookahead.
    NaN warmup: first `window` rows are NaN.
    """

    def __init__(self, window: int = 20) -> None:
        super().__init__(name=f"bollinger_{window}", lookback=window)
        self.window = window

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        mean = prices.rolling(self.window).mean()
        std = prices.rolling(self.window).std()
        z = (prices - mean) / std
        return -z  # positive = below band (buy), negative = above band (sell)