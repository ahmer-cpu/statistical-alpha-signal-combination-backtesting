import pandas as pd

from qre.alpha.base import Alpha


class ShortTermReversal(Alpha):
    """Short-term reversal factor (Lehmann, 1990).

    Captures microstructure-driven mean reversion at the weekly horizon:
    stocks that fell sharply over the past few days tend to bounce as
    order-flow imbalances normalise. (Jegadeesh, 1990, documents the
    analogous effect at the one-month horizon.)

    Formula: -(P_t / P_{t-window} - 1)
    Signal: positive = recent loser (buy), negative = recent winner (sell).
    Timing: uses prices up to close of date t; no lookahead.
    NaN warmup: first `window` rows are NaN.
    """

    def __init__(self, window: int = 5) -> None:
        super().__init__(name=f"short_term_reversal_{window}", lookback=window)
        self.window = window

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        return -(prices / prices.shift(self.window) - 1)
