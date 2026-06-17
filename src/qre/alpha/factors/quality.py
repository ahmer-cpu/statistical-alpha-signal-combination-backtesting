import pandas as pd

from qre.alpha.base import Alpha


class QualityProxy(Alpha):
    """Quality proxy factor (inspired by Asness, Frazzini & Pedersen, 2019).

    Approximates the "Quality Minus Junk" factor using price data only.
    Stocks with stable, consistent return patterns score high. This is
    the inverse coefficient of variation of absolute returns — a proxy
    for earnings stability when fundamental data is unavailable.

    Formula: -rolling_std(r, window) / rolling_mean(|r|, window)
    Signal: positive = stable returns (buy), negative = erratic (sell).
    Timing: uses prices up to close of date t; no lookahead.
    NaN warmup: first `window + 1` rows are NaN (1 for returns, window for rolling).
    """

    def __init__(self, window: int = 252) -> None:
        super().__init__(name=f"quality_proxy_{window}", lookback=window + 1)
        self.window = window

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change()
        vol = returns.rolling(self.window).std()
        mean_abs = returns.abs().rolling(self.window).mean()

        # Guard against division by near-zero mean absolute return
        mean_abs = mean_abs.where(mean_abs > 1e-10)

        return -(vol / mean_abs)
