import pandas as pd

from qre.alpha.base import Alpha


class RSI(Alpha):
    """Relative Strength Index mean-reversion signal (Wilder, 1978).

    Computes standard RSI, then centres at zero so the signal follows
    the base class convention: positive = buy (oversold), negative =
    sell (overbought).

    Formula: 50 - RSI, where RSI = 100 - 100 / (1 + RS).
    Signal: positive = oversold (buy), negative = overbought (sell).
    Timing: uses prices up to close of date t; no lookahead.
    NaN warmup: first `window` rows are NaN.
    """

    def __init__(self, window: int = 14) -> None:
        super().__init__(name=f"rsi_{window}", lookback=window)
        self.window = window

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        diff = prices.diff()
        gains = diff.clip(lower=0)      # same shape, zeros on negative moves
        losses = -diff.clip(upper=0)    # zeros on positive moves, take absolute value
        rs = gains.rolling(self.window).mean() / losses.rolling(self.window).mean()

        rsi = 100 - (100 / (1 + rs))
        # positive = oversold (buy), negative = overbought (sell)
        return 50 - rsi