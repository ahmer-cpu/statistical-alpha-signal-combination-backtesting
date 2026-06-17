"""Cross-factor IC correlation analysis.

Measures how correlated the *timing* of alpha is across factors.
Two factors may have uncorrelated raw signals yet predict the same
return variation on the same days — this matrix reveals that overlap.
"""

import pandas as pd

from qre.analytics.ic import compute_ic


def ic_correlation_matrix(
    signals: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    horizon: int = 1,
    min_periods: int = 5,
) -> pd.DataFrame:
    """K x K correlation matrix of daily IC series across factors.

    For each factor, computes the daily cross-sectional rank IC against
    forward returns, then returns the Pearson correlation matrix of those
    IC time series.

    High correlation between two factors means they predict the same
    return variation on the same days — combining them adds little
    diversification benefit.

    Args:
        signals: Mapping of factor name to signal DataFrame (date x ticker).
        prices: Price DataFrame used to compute forward returns.
        horizon: Forward return horizon in days.
        min_periods: Minimum valid tickers per date for a valid IC.

    Returns:
        K x K DataFrame of pairwise IC correlations, where K is the
        number of factors.
    """
    forward_returns = prices.pct_change(horizon).shift(-horizon)

    ic_series = {
        name: compute_ic(signal, forward_returns, min_periods=min_periods)
        for name, signal in signals.items()
    }

    ic_panel = pd.DataFrame(ic_series)
    return ic_panel.corr()
