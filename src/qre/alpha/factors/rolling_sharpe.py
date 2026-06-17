import pandas as pd

from qre.alpha.base import Alpha


class RollingSharpe(Alpha):
    """Risk-adjusted momentum factor (Moskowitz, Ooi & Pedersen, 2012).

    Computes the rolling Sharpe ratio for each stock — mean return
    divided by return volatility. This penalises volatile winners that
    raw momentum would overweight, and rewards consistent compounders.

    Formula: rolling_mean(r, window) / rolling_std(r, window)
    Signal: positive = high risk-adjusted return (buy),
            negative = poor risk-adjusted return (sell).
    Timing: uses prices up to close of date t; no lookahead.
    NaN warmup: first `window + 1` rows are NaN (1 for returns, window for rolling).
    """

    def __init__(self, window: int = 252) -> None:
        super().__init__(name=f"rolling_sharpe_{window}", lookback=window + 1)
        self.window = window

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change()
        mean = returns.rolling(self.window).mean()
        std = returns.rolling(self.window).std()

        # Guard against division by near-zero volatility
        std = std.where(std > 1e-10)

        return mean / std
