import pandas as pd

from qre.alpha.base import Alpha


class CrossSectionalMomentum(Alpha):
    """Cross-sectional momentum factor (Jegadeesh & Titman, 1993).

    Ranks stocks by relative time-series momentum each day. Unlike
    raw Momentum which produces absolute returns, this outputs
    cross-sectional percentile ranks — making it scale-invariant
    and directly comparable across regimes.

    Formula: rank(P_{t-skip} / P_{t-window} - 1, axis=stocks, pct=True)
    Signal: higher rank = stronger relative momentum (go long).
    Timing: uses prices up to close of date t; no lookahead.
    NaN warmup: first `window` rows are NaN.
    """

    def __init__(self, window: int = 252, skip: int = 21) -> None:
        super().__init__(
            name=f"cross_sectional_momentum_{window}_{skip}",
            lookback=window,
        )
        self.window = window
        self.skip = skip

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        time_series_momentum = prices.shift(self.skip) / prices.shift(self.window) - 1
        return time_series_momentum.rank(axis=1, pct=True)