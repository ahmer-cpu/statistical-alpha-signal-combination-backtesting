import pandas as pd

from qre.alpha.base import Alpha


class RollingMaxDrawdown(Alpha):
    """Rolling maximum drawdown factor (Magdon-Ismail & Atiya, 2004).

    Computes the worst peak-to-trough decline within a rolling window
    for each stock (a negative number). This is path-dependent — it captures
    risk that no point statistic (volatility, skewness, VaR) can measure.
    Stocks with shallower (less negative) drawdowns are considered more
    stable: they rank higher and are bought; deep-drawdown stocks are sold.

    Formula: max_drawdown(r, window)  — worst peak-to-trough decline (negative)
    Signal: higher (shallow drawdown) = long, lower (deep drawdown) = short.
        Drawdown is negative for every stock; the cross-sectional ranking
        (after the backtester's demeaning) places positions — same low-risk =
        long convention as LowVol.
    Timing: uses prices up to close of date t; no lookahead.
    NaN warmup: first `window + 1` rows are NaN (1 for returns, window for rolling).
    """

    def __init__(self, window: int = 252) -> None:
        super().__init__(
            name=f"rolling_max_drawdown_{window}", lookback=window + 1,
        )
        self.window = window

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change()

        def _max_drawdown(r: pd.Series) -> float:
            """Maximum drawdown within a return window."""
            cumulative = (1.0 + r).cumprod()
            running_max = cumulative.cummax()
            drawdowns = cumulative / running_max - 1.0
            return float(drawdowns.min())

        # Max drawdown is negative for all stocks; a shallower (less negative)
        # decline ranks higher, giving the low-risk = long convention.
        return returns.rolling(self.window).apply(_max_drawdown, raw=False)
