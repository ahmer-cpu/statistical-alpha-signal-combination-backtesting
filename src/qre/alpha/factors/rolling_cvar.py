import pandas as pd

from qre.alpha.base import Alpha


class RollingCVaR(Alpha):
    """Conditional Value at Risk factor (Artzner et al., 1999).

    Computes the rolling CVaR (Expected Shortfall) at the 5th percentile
    for each stock — the average return on the worst 5% of days, a negative
    number. Stocks with a milder (less negative) left tail are considered
    higher quality: they rank higher and are bought; severe-tail stocks are
    sold. Distinct from LowVol: volatility measures overall dispersion, CVaR
    measures left-tail severity specifically.

    Formula: rolling_mean(r where r <= 5th percentile, window)
    Signal: higher (mild left tail) = long, lower (severe tail) = short.
        Expected shortfall is negative for every stock; the cross-sectional
        ranking (after the backtester's demeaning) is what places positions —
        same low-risk = long convention as LowVol.
    Timing: uses prices up to close of date t; no lookahead.
    NaN warmup: first `window + 1` rows are NaN (1 for returns, window for rolling).
    Note: uses order statistics, not higher moments — very stable.
    """

    def __init__(self, window: int = 252, confidence: float = 0.05) -> None:
        super().__init__(
            name=f"rolling_cvar_{window}_{int(confidence * 100)}",
            lookback=window + 1,
        )
        self.window = window
        self.confidence = confidence

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change()

        def _cvar(x: pd.Series) -> float:
            """Mean of returns below the confidence-th percentile."""
            threshold = x.quantile(self.confidence)
            tail = x[x <= threshold]
            if tail.empty:
                return float("nan")
            return tail.mean()

        # Expected shortfall is negative for all stocks; a milder (less
        # negative) tail ranks higher, giving the low-risk = long convention.
        return returns.rolling(self.window).apply(_cvar, raw=False)
