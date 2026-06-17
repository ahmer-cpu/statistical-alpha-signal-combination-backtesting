"""Information Coefficient (IC) analysis for evaluating alpha signal quality."""

from collections.abc import Callable

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acovf  # type: ignore[import-untyped]


def newey_west_t_stat(
    ic_series: pd.Series,
    bandwidth: int | None = None,
) -> dict[str, float]:
    """Newey-West HAC-corrected t-statistic for testing if mean IC is zero.

    Standard t-stats assume independent observations, but daily IC values
    computed against overlapping forward returns are heavily autocorrelated.
    The Newey-West estimator inflates the standard error to account for this,
    producing honest significance tests.

    Args:
        ic_series: Daily IC values from compute_ic.
        bandwidth: Number of lags for the Bartlett kernel. If None, uses
            n^(1/3) automatic selection. For overlapping forward returns
            at horizon h, pass h - 1.
    """
    clean = ic_series.dropna()
    x = clean.to_numpy()
    n = len(x)

    nan_result = {
        "t_stat_naive": float("nan"),
        "t_stat_nw": float("nan"),
        "se_naive": float("nan"),
        "se_nw": float("nan"),
        "bandwidth": 0,
    }

    if n < 2:
        return nan_result

    mean = float(x.mean())
    std = float(x.std(ddof=1))
    se_naive = std / n**0.5
    t_naive = mean / se_naive if se_naive > 0 else float("nan")

    m = bandwidth if bandwidth is not None else int(np.ceil(n ** (1 / 3)))
    m = min(m, n - 1)  # cannot have more lags than observations

    gamma = acovf(x, demean=True, fft=True, nlag=m)

    weights = 1 - np.arange(1, m + 1) / (m + 1)
    v_nw = gamma[0] + 2 * np.sum(weights * gamma[1:])

    se_nw = float(np.sqrt(v_nw / n))
    t_nw = mean / se_nw if se_nw > 0 else float("nan")

    return {
        "t_stat_naive": t_naive,
        "t_stat_nw": t_nw,
        "se_naive": se_naive,
        "se_nw": se_nw,
        "bandwidth": m,
    }


def non_overlapping_ic(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    horizon: int = 21,
    min_periods: int = 5,
) -> pd.Series:
    """IC computed every h-th day to eliminate overlapping forward returns.

    Standard daily IC at a multi-day horizon reuses most of the same return
    data on consecutive days (e.g., 20/21 days shared at the 21d horizon).
    This function samples every `horizon`-th date so that return windows
    are completely disjoint, producing genuinely independent IC observations.

    The trade-off is sample size: ~1,300 daily ICs become ~65 non-overlapping
    ones at horizon 21. But those 65 are honest — no autocorrelation inflation.

    Args:
        signal: DataFrame of signal values (date × ticker).
        forward_returns: DataFrame of forward returns at the given horizon.
        horizon: Spacing between sampled dates (matches the forward return
            horizon to ensure no overlap).
        min_periods: Minimum valid tickers per date for a valid IC.
    """
    # Align and find dates where both signal and returns exist
    sig_aligned, fwd_aligned = signal.align(forward_returns, join="inner", axis=0)

    # Sample every h-th date — return windows become disjoint
    sampled_dates = sig_aligned.index[::horizon]

    # Reuse compute_ic on just the sampled dates
    return compute_ic(
        sig_aligned.loc[sampled_dates],
        fwd_aligned.loc[sampled_dates],
        min_periods=min_periods,
    )


def block_bootstrap_ci(
    ic_series: pd.Series,
    stat_fn: Callable[[np.ndarray], float] | None = None,
    block_length: int | None = None,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> dict[str, float]:
    """Block bootstrap confidence interval for an IC statistic.

    Standard bootstrap resamples individual observations, destroying the
    temporal dependence structure. Block bootstrap resamples contiguous
    blocks, preserving the autocorrelation within each block.

    If a block would extend past the end of the series, it is truncated
    to the remaining observations (no circular wrapping).

    Args:
        ic_series: Daily IC values from compute_ic.
        stat_fn: Statistic to compute on each resample. Defaults to np.mean.
            Pass a custom callable (e.g., Sharpe function) for other statistics.
        block_length: Length of each contiguous block. If None, uses n^(1/3).
            For IC with known horizon overlap, use the horizon value.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level for the interval (e.g., 0.95 for 95% CI).
        seed: Random seed for reproducibility.
    """
    clean = ic_series.dropna()
    x = clean.to_numpy()
    n = len(x)

    if stat_fn is None:
        stat_fn = np.mean

    nan_result = {
        "point_estimate": float("nan"),
        "ci_lower": float("nan"),
        "ci_upper": float("nan"),
        "se_bootstrap": float("nan"),
    }

    # n_bootstrap < 1 means "skip the bootstrap" (matches ic_summary's
    # `if n_bootstrap > 0` convention): there are no resamples to take a
    # percentile over, so report no CI rather than crashing on np.percentile([]).
    if n < 2 or n_bootstrap < 1:
        return nan_result

    b = block_length if block_length is not None else int(np.ceil(n ** (1 / 3)))
    b = max(1, min(b, n))  # clamp to [1, n]

    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        # Draw random block starts and concatenate until we have n values
        sample_pieces: list[np.ndarray] = []
        total = 0
        while total < n:
            start = int(rng.integers(0, n))
            end = min(start + b, n)  # truncate at array boundary
            sample_pieces.append(x[start:end])
            total += end - start

        # Trim to exactly n values
        sample = np.concatenate(sample_pieces)[:n]
        boot_stats[i] = stat_fn(sample)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return {
        "point_estimate": float(stat_fn(x)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se_bootstrap": float(np.std(boot_stats, ddof=1)),
    }


def compute_ic(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    min_periods: int = 5,
) -> pd.Series:
    """Daily Spearman rank correlation between signal and forward returns.

    For each date, ranks both signal and returns across tickers,
    then computes Pearson correlation on the ranks (= Spearman).

    Args:
        signal: DataFrame of signal values (date × ticker).
        forward_returns: DataFrame of forward returns (date × ticker).
        min_periods: Minimum number of valid ticker observations per date.
            Days with fewer valid pairs produce NaN.
    """
    # Align so indices and columns match exactly
    signal, forward_returns = signal.align(forward_returns, join="inner", axis=0)
    forward_returns = forward_returns.reindex(columns=signal.columns)

    # Count valid (non-NaN in both) observations per date
    valid = signal.notna() & forward_returns.notna()
    count = valid.sum(axis=1)

    # Rank over the pairwise-valid set only. Ranking the full row first and
    # letting corrwith drop NaN pairs corrupts the surviving ranks: a name
    # present in one frame but missing in the other still inflates the other
    # names' ranks. Masking to the common-valid set before ranking keeps the
    # rank correlation honest when signal/return NaN-masks differ in a row.
    sig_valid = signal.where(valid)
    ret_valid = forward_returns.where(valid)
    ranked_signal = sig_valid.rank(axis=1)
    ranked_returns = ret_valid.rank(axis=1)
    ic = ranked_signal.corrwith(ranked_returns, axis=1)

    # Mask dates with too few observations
    ic[count < min_periods] = float("nan")

    return ic


def ic_summary(
    ic_series: pd.Series,
    signal: pd.DataFrame | None = None,
    forward_returns: pd.DataFrame | None = None,
    horizon: int | None = None,
    n_bootstrap: int = 0,
    bootstrap_seed: int | None = 42,
) -> dict[str, float]:
    """Summary statistics for an IC series.

    Args:
        ic_series: Daily IC values from compute_ic.
        signal: Signal DataFrame (for computing avg cross-sectional coverage).
        forward_returns: Forward returns DataFrame (for computing avg coverage).
        horizon: Forward return horizon in days. When provided, adds
            Newey-West HAC-corrected t-stat (bandwidth = horizon - 1).
        n_bootstrap: Number of block bootstrap resamples. When > 0, adds
            95% confidence interval for the mean IC (block_length = horizon
            or n^(1/3) if horizon is None).
        bootstrap_seed: Random seed for bootstrap reproducibility.
    """
    clean = ic_series.dropna()

    if len(clean) == 0:
        result: dict[str, float] = {
            "mean": float("nan"),
            "std": float("nan"),
            "icir": float("nan"),
            "hit_rate": float("nan"),
            "t_stat": float("nan"),
            "n_dates": 0,
            "avg_coverage": float("nan"),
        }
        if horizon is not None:
            result["t_stat_nw"] = float("nan")
            result["se_nw"] = float("nan")
            result["bandwidth"] = 0
        if n_bootstrap > 0:
            result["ci_lower"] = float("nan")
            result["ci_upper"] = float("nan")
            result["se_bootstrap"] = float("nan")
        return result

    mean = clean.mean()
    std = clean.std()
    n = len(clean)

    avg_cov = float("nan")
    if signal is not None and forward_returns is not None:
        sig, fwd = signal.align(forward_returns, join="inner", axis=0)
        fwd = fwd.reindex(columns=sig.columns)
        valid = sig.notna() & fwd.notna()
        avg_cov = float(valid.sum(axis=1).mean())

    result = {
        "mean": float(mean),
        "std": float(std),
        "icir": float(mean / std) if std != 0 else float("nan"),
        "hit_rate": float((clean > 0).mean()),
        "t_stat": float(mean / std * n**0.5) if std != 0 else float("nan"),
        "n_dates": n,
        "avg_coverage": avg_cov,
    }

    # Newey-West corrected t-stat when horizon is known
    if horizon is not None:
        nw = newey_west_t_stat(ic_series, bandwidth=horizon - 1)
        result["t_stat_nw"] = nw["t_stat_nw"]
        result["se_nw"] = nw["se_nw"]
        result["bandwidth"] = nw["bandwidth"]

    # Block bootstrap CI for the mean IC
    if n_bootstrap > 0:
        boot = block_bootstrap_ci(
            ic_series,
            block_length=horizon,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
        )
        result["ci_lower"] = boot["ci_lower"]
        result["ci_upper"] = boot["ci_upper"]
        result["se_bootstrap"] = boot["se_bootstrap"]

    return result


def ic_decay(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    max_lag: int = 20,
    min_periods: int = 5,
) -> pd.Series:
    """Mean IC at each forward horizon from 1 to max_lag days.

    Shows how quickly the signal's predictive power fades,
    revealing the natural holding period.
    """
    mean_ic_by_lag = {
        lag: compute_ic(
            signal, prices.pct_change(lag).shift(-lag), min_periods=min_periods
        ).mean()
        for lag in range(1, max_lag + 1)
    }
    return pd.Series(mean_ic_by_lag, name="ic_decay")
