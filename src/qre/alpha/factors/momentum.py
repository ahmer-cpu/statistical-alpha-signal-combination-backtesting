import pandas as pd

from qre.alpha.base import Alpha


class Momentum(Alpha):
    """Time-series momentum factor (Jegadeesh & Titman, 1993).

    Captures investor under-reaction: stocks that outperformed over the
    past 12 months (skipping the most recent month) tend to continue
    outperforming. The 1-month skip avoids short-term reversal effects.

    Formula: P_{t-skip} / P_{t-window} - 1
    Default (252, 21) gives 12-month momentum with 1-month skip.
    Signal: positive = price rose over the lookback (go long).
    Timing: uses prices up to close of date t; no lookahead.
    NaN warmup: first `window` rows are NaN.
    """

    def __init__(self, window: int = 252, skip: int = 21) -> None:
        super().__init__(name=f"momentum_{window}_{skip}", lookback=window)
        self.window = window
        self.skip = skip

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        return prices.shift(self.skip) / prices.shift(self.window) - 1