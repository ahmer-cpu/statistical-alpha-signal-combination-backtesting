"""Pre-backtest signal health checks.

These diagnostics characterize a signal's statistical properties before
committing to a full backtest. They answer: Is the signal differentiated?
How fast does it change? Is it predictive or merely reactive?
"""

import numpy as np
import pandas as pd


def signal_autocorrelation(
    signal: pd.DataFrame,
    max_lag: int = 20,
) -> pd.DataFrame:
    """Cross-sectional average autocorrelation at each lag.

    Computes per-ticker autocorrelation from lag 1 to max_lag, then
    averages across tickers. A lag-1 value above 0.95 indicates a
    quasi-static signal (e.g., momentum) — daily rebalancing would
    churn through near-identical positions.

    Args:
        signal: DataFrame of signal values (date x ticker).
        max_lag: Maximum lag to compute (inclusive).

    Returns:
        DataFrame with columns ['lag', 'autocorr'] where autocorr is
        the cross-sectional mean autocorrelation at each lag.
    """
    lags = range(1, max_lag + 1)
    mean_autocorrs = []

    for lag in lags:
        # Per-ticker autocorrelation at this lag, then average across tickers.
        # A constant column (e.g. a single-name sector's demeaned signal) gives
        # 0/0 in the underlying corrcoef -> a legitimate NaN; suppress the noisy
        # floating-point warning since .mean() already skips those NaNs.
        with np.errstate(invalid="ignore", divide="ignore"):
            per_ticker = signal.apply(lambda col: col.autocorr(lag=lag))
        mean_autocorrs.append(per_ticker.mean())

    return pd.DataFrame({"lag": list(lags), "autocorr": mean_autocorrs})


def signal_turnover(
    signal: pd.DataFrame,
    normalize: bool = True,
) -> pd.Series:
    """Daily turnover implied by the signal before backtesting.

    Measures how much the signal's cross-sectional weights change
    day-to-day. Low turnover (< 0.05) suggests daily rebalancing is
    wasteful — the signal barely moves between days.

    Args:
        signal: DataFrame of signal values (date x ticker).
        normalize: If True, divide by gross exposure (sum of absolute
            weights) so turnover is comparable across different universe
            sizes. If False, return raw absolute change.

    Returns:
        Series of daily turnover values, indexed by date.
        First date is NaN (no prior day to diff against).
    """
    raw_change = signal.diff().abs().sum(axis=1)

    if normalize:
        gross = signal.abs().sum(axis=1).replace(0.0, np.nan)
        return raw_change / gross

    return raw_change


def cross_sectional_dispersion(signal: pd.DataFrame) -> pd.Series:
    """Daily cross-sectional standard deviation of the signal.

    If dispersion is near zero, the signal assigns similar values to
    all tickers — it cannot differentiate stocks and will produce
    near-zero positions after dollar-neutral normalization.

    Args:
        signal: DataFrame of signal values (date x ticker).

    Returns:
        Series of daily cross-sectional std, indexed by date.
    """
    return signal.std(axis=1)


def signal_return_lead_lag(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    max_lead: int = 10,
    max_lag: int = 10,
) -> pd.Series:
    """Cross-correlation between signal and returns at various offsets.

    For each offset k, computes the average cross-sectional Spearman
    correlation between signal(t) and return(t+k).

    - Positive k (leads): signal predicts future returns — genuine alpha.
    - Negative k (lags): signal correlates with past returns — reactive.
      A signal with strong negative-k correlation is just echoing past
      price moves (e.g., momentum is literally past returns).

    Args:
        signal: DataFrame of signal values (date x ticker).
        prices: DataFrame of prices (date x ticker), used to compute
            daily returns.
        max_lead: Maximum forward offset (positive k).
        max_lag: Maximum backward offset (negative k).

    Returns:
        Series indexed by offset k (from -max_lag to +max_lead),
        with mean cross-sectional Spearman correlation at each offset.
    """
    prices, signal = prices.align(signal, join="inner", axis=0)
    signal = signal.reindex(columns=prices.columns)

    daily_returns = prices.pct_change()

    offsets = range(-max_lag, max_lead + 1)
    correlations = {}

    for k in offsets:
        if k == 0:
            # Contemporaneous: signal vs same-day return (not useful, but complete)
            shifted_returns = daily_returns
        else:
            # shift(-k): at offset k, we want return(t+k) aligned with signal(t)
            shifted_returns = daily_returns.shift(-k)

        # Cross-sectional Spearman correlation each day, then average.
        # Mask to the pairwise-valid set before ranking (see compute_ic): a
        # name missing in only one frame would otherwise distort the other
        # names' ranks.
        valid = signal.notna() & shifted_returns.notna()
        ranked_sig = signal.where(valid).rank(axis=1)
        ranked_ret = shifted_returns.where(valid).rank(axis=1)
        daily_corr = ranked_sig.corrwith(ranked_ret, axis=1)
        correlations[k] = daily_corr.mean()

    result = pd.Series(correlations, name="lead_lag_ic")
    result.index.name = "offset"
    return result
