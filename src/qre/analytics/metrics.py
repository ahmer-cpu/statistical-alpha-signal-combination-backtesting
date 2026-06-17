"""Performance metrics for strategy evaluation.

All functions accept a pandas Series of simple returns (not log returns)
indexed by date/timestamp. This is the standard input format — convert
from prices using: returns = prices.pct_change().dropna()

Annualization uses an explicit periods_per_year parameter (default 252 for
daily US equity data). Common values:
    252      — daily bars
    252 * 78 — 5-minute bars (78 bars per 6.5h trading day)
    252 * 390 — 1-minute bars
    12       — monthly returns
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio.

    Sharpe = (mean excess return / std of returns) * sqrt(periods_per_year)

    Args:
        returns: Series of simple period returns.
        risk_free_rate: Annualized risk-free rate (default 0).
        periods_per_year: Number of return observations per year.

    Returns:
        Annualized Sharpe ratio. Returns inf if mean > 0 with zero std,
        -inf if mean < 0 with zero std, 0.0 if insufficient data.
    """
    if len(returns) < 2:
        return 0.0

    # Compound conversion: annual rate → per-period rate
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    excess = returns - rf_per_period
    std = excess.std(ddof=1)

    if std == 0:
        mean = excess.mean()
        if mean > 0:
            return float("inf")
        elif mean < 0:
            return float("-inf")
        return 0.0

    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sortino ratio.

    Like Sharpe but only penalizes downside volatility. The downside deviation
    is computed as sqrt(mean(negative_excess_returns²)) over ONLY the negative
    observations. This measures the average magnitude of losses, not their
    frequency. The frequency effect is captured in the numerator (mean of ALL
    returns).

    Returns:
        Annualized Sortino ratio. Returns inf if mean > 0 with no downside,
        0.0 if insufficient data.
    """
    if len(returns) < 2:
        return 0.0

    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    excess = returns - rf_per_period
    downside = excess[excess < 0]

    if len(downside) == 0:
        # No negative returns — ratio is infinite if mean > 0
        mean = excess.mean()
        return float("inf") if mean > 0 else 0.0

    downside_std = np.sqrt((downside**2).mean())

    if downside_std == 0:
        return 0.0

    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown as a negative decimal (e.g., -0.25 = 25% drawdown).

    Drawdown measures the peak-to-trough decline in cumulative returns.
    Equity starts at 1.0 before the first return, so a negative first
    return is correctly captured as a drawdown from initial capital.

    Returns:
        Maximum drawdown (negative number, or 0.0 if no drawdown occurred).
    """
    if len(returns) == 0:
        return 0.0

    clean = returns.dropna()
    if len(clean) == 0:
        return 0.0

    # Prepend 1.0 so the running max starts at initial capital,
    # not at the first post-return equity value.
    equity = np.r_[1.0, (1.0 + clean).cumprod().to_numpy()]
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0

    return float(drawdown.min())


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calmar ratio: annualized return / absolute max drawdown.

    Measures return per unit of tail risk. Higher is better.

    Returns:
        Calmar ratio. Returns 0.0 if no drawdown occurred.
    """
    if len(returns) < 2:
        return 0.0

    ann_ret = annualized_return(returns, periods_per_year)
    mdd = max_drawdown(returns)

    if mdd == 0:
        return 0.0

    return float(ann_ret / abs(mdd))


def annualized_return(
    returns: pd.Series,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """Compound annualized growth rate (CAGR).

    Uses the number of periods and periods_per_year to determine how many
    years the returns span, then compounds to an annual figure. This is
    consistent with how volatility and Sharpe are annualized.
    """
    if len(returns) == 0:
        return 0.0

    total_return = float(np.prod(1.0 + returns)) - 1.0
    years = len(returns) / periods_per_year

    if years == 0:
        return 0.0

    return float((1 + total_return) ** (1 / years) - 1)


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized standard deviation of returns."""
    if len(returns) < 2:
        return 0.0

    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def win_rate(returns: pd.Series) -> float:
    """Fraction of periods with positive returns.

    Returns:
        Value between 0 and 1.
    """
    if len(returns) == 0:
        return 0.0

    return float((returns > 0).sum() / len(returns))


def profit_factor(returns: pd.Series) -> float:
    """Gross profits / gross losses on period returns.

    Assumes equal capital exposure each period. For variable-size positions,
    use dollar PnL per trade instead of percentage returns.

    Returns:
        Profit factor. Returns inf if no losses, 0.0 if no gains.
    """
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())

    if losses == 0:
        return float("inf") if gains > 0 else 0.0

    return float(gains / losses)

def beta(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Beta of strategy relative to benchmark (OLS slope).

    β = cov(strategy, benchmark) / var(benchmark)

    Args:
        strategy_returns: Series of simple period returns.
        benchmark_returns: Series of simple period returns for the benchmark.

    Returns:
        Beta coefficient. Returns 0.0 if insufficient data or zero benchmark variance.
    """
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()

    if len(aligned) < 2:
        return 0.0

    strat = aligned.iloc[:, 0]
    bench = aligned.iloc[:, 1]

    bench_var = bench.var(ddof=1)
    if bench_var == 0:
        return 0.0

    return float(strat.cov(bench) / bench_var)

def alpha(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """Jensen's alpha: annualized excess return unexplained by beta.

    α = (mean(strategy - rf) - β * mean(benchmark - rf)) * periods_per_year

    Args:
        strategy_returns: Series of simple period returns.
        benchmark_returns: Series of simple period returns for the benchmark.
        risk_free_rate: Annualized risk-free rate (default 0).
        periods_per_year: Number of return observations per year.

    Returns:
        Annualized alpha. Returns 0.0 if insufficient data.
    """
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()

    if len(aligned) < 2:
        return 0.0

    strat = aligned.iloc[:, 0]
    bench = aligned.iloc[:, 1]

    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    beta_coeff = beta(strat, bench)
    ann_alpha = (
        (strat.mean() - rf_per_period) - beta_coeff * (bench.mean() - rf_per_period)
    ) * periods_per_year

    return float(ann_alpha)


def tracking_error(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized tracking error (volatility of active returns).

    TE = std(strategy - benchmark) * sqrt(periods_per_year)

    Measures how much a strategy's returns deviate from the benchmark
    on a day-to-day basis. A pure index fund has TE ≈ 0. A market-neutral
    strategy will have TE close to its own volatility.

    Returns:
        Annualized tracking error. Returns 0.0 if insufficient data.
    """
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()

    if len(aligned) < 2:
        return 0.0

    active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]

    return float(active_returns.std(ddof=1) * np.sqrt(periods_per_year))


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """Information ratio: annualized active return / tracking error.

    IR = mean(strategy - benchmark) * periods_per_year / TE

    This is the Sharpe ratio of your *active bets*. It answers:
    "Per unit of deviation from the benchmark, how much extra return
    did I earn?"

    Benchmarks: IR > 0.5 is good, > 1.0 is exceptional.

    Returns:
        Information ratio. Returns 0.0 if insufficient data or zero
        tracking error.
    """
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()

    if len(aligned) < 2:
        return 0.0

    active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = float(active_returns.std(ddof=1) * np.sqrt(periods_per_year))

    if te == 0:
        return 0.0

    ann_active_return = float(active_returns.mean() * periods_per_year)

    return float(ann_active_return / te)


def up_capture(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Up-capture ratio: strategy's mean return on benchmark up-days.

    up_capture = mean(strategy | benchmark > 0) / mean(benchmark | benchmark > 0)

    A value of 1.1 means "when the market is up, I capture 110% of the move."

    Returns:
        Up-capture ratio. Returns 0.0 if no benchmark up-days.
    """
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()

    if len(aligned) < 2:
        return 0.0

    strat = aligned.iloc[:, 0]
    bench = aligned.iloc[:, 1]

    up_mask = bench > 0
    if up_mask.sum() == 0:
        return 0.0

    return float(strat[up_mask].mean() / bench[up_mask].mean())


def down_capture(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Down-capture ratio: strategy's mean return on benchmark down-days.

    down_capture = mean(strategy | benchmark < 0) / mean(benchmark | benchmark < 0)

    A value of 0.8 means "when the market falls, I only lose 80% of the drop."
    Lower is better. Values < 1 indicate downside protection.

    Returns:
        Down-capture ratio. Returns 0.0 if no benchmark down-days.
    """
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()

    if len(aligned) < 2:
        return 0.0

    strat = aligned.iloc[:, 0]
    bench = aligned.iloc[:, 1]

    down_mask = bench < 0
    if down_mask.sum() == 0:
        return 0.0

    return float(strat[down_mask].mean() / bench[down_mask].mean())


# --------------------------------------------------------------------------
# Distribution metrics
# --------------------------------------------------------------------------


def skewness(returns: pd.Series) -> float:
    """Sample skewness of returns.

    Negative skew means the left tail (losses) is fatter than the right.
    Most equity strategies have negative skew — big down days are more
    common than equivalently big up days.

    Returns:
        Skewness. Returns 0.0 if fewer than 3 observations.
    """
    if len(returns) < 3:
        return 0.0

    return float(returns.skew())  # type: ignore[arg-type]


def excess_kurtosis(returns: pd.Series) -> float:
    """Excess kurtosis of returns (Fisher's definition, normal = 0).

    High kurtosis means extreme events (both tails) happen more often
    than a normal distribution predicts. Daily equity returns typically
    have excess kurtosis of 3-10+.

    Returns:
        Excess kurtosis. Returns 0.0 if fewer than 4 observations.
    """
    if len(returns) < 4:
        return 0.0

    return float(returns.kurtosis())  # type: ignore[arg-type]


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value at Risk (VaR) at a given confidence level.

    VaR is the loss threshold that is only exceeded (1 - confidence)%
    of the time. Returned as a negative number (it's a loss).

    Example: VaR(95%) = -0.02 means "on 95% of days, you lose at most 2%."

    Returns:
        VaR as a negative decimal. Returns 0.0 if insufficient data.
    """
    if len(returns) < 2:
        return 0.0

    return float(returns.quantile(1 - confidence))


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional Value at Risk (CVaR) / Expected Shortfall.

    The average loss on days that exceed the VaR threshold. Always worse
    (more negative) than VaR because it's the mean of the tail, not just
    the boundary.

    Example: CVaR(95%) = -0.035 means "when you DO have a bad day beyond
    the 5th percentile, you lose 3.5% on average."

    Returns:
        CVaR as a negative decimal. Returns 0.0 if insufficient data.
    """
    if len(returns) < 2:
        return 0.0

    var_threshold = returns.quantile(1 - confidence)
    tail = returns[returns <= var_threshold]

    if len(tail) == 0:
        return 0.0

    return float(tail.mean())


def tail_ratio(returns: pd.Series, percentile: float = 0.05) -> float:
    """Tail ratio: right tail magnitude / left tail magnitude.

    tail_ratio = abs(upper percentile) / abs(lower percentile)

    Values > 1 mean your winning extremes are larger than your losing
    extremes. Values < 1 mean your worst days are bigger than your best.

    Args:
        returns: Series of simple period returns.
        percentile: Percentile for tails (default 0.05 = 5th/95th).

    Returns:
        Tail ratio. Returns 0.0 if left tail is zero.
    """
    if len(returns) < 2:
        return 0.0

    upper = returns.quantile(1 - percentile)
    lower = returns.quantile(percentile)

    if lower == 0:
        return 0.0

    return float(abs(upper) / abs(lower))


def expected_tail_ratio(returns: pd.Series, percentile: float = 0.05) -> float:
    """Expected tail ratio: mean of right tail / abs(mean of left tail).

    Unlike tail_ratio which compares single quantile boundaries, this
    compares the average magnitude of gains beyond the upper percentile
    to the average magnitude of losses beyond the lower percentile.

    Values > 1 mean your average extreme win is larger than your average
    extreme loss.

    Args:
        returns: Series of simple period returns.
        percentile: Percentile for tails (default 0.05 = 5th/95th).

    Returns:
        Expected tail ratio. Returns 0.0 if either tail is empty or
        left tail mean is zero.
    """
    if len(returns) < 2:
        return 0.0

    upper_threshold = returns.quantile(1 - percentile)
    lower_threshold = returns.quantile(percentile)

    right_tail = returns[returns >= upper_threshold]
    left_tail = returns[returns <= lower_threshold]

    if len(left_tail) == 0 or len(right_tail) == 0:
        return 0.0

    left_mean = left_tail.mean()
    if left_mean == 0:
        return 0.0

    return float(abs(right_tail.mean()) / abs(left_mean))


def best_day(returns: pd.Series) -> float:
    """Largest single-period return."""
    if len(returns) == 0:
        return 0.0
    return float(returns.max())


def worst_day(returns: pd.Series) -> float:
    """Smallest (most negative) single-period return."""
    if len(returns) == 0:
        return 0.0
    return float(returns.min())


def summary(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Compute standalone metrics (no benchmark needed).

    Convenient for displaying results or comparing strategies.
    """
    return {
        "annualized_return": annualized_return(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns, periods_per_year),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
    }


def full_summary(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Compute all metrics including benchmark-relative and distribution.

    This is the complete tear-sheet summary — standalone performance,
    benchmark-relative analytics, and distributional characteristics.

    Args:
        strategy_returns: Series of simple period returns.
        benchmark_returns: Series of simple period returns for the benchmark.
        risk_free_rate: Annualized risk-free rate (default 0).
        periods_per_year: Number of return observations per year.

    Returns:
        Dictionary of all metric names to float values.
    """
    return {
        # Standalone performance
        "annualized_return": annualized_return(strategy_returns, periods_per_year),
        "annualized_volatility": annualized_volatility(
            strategy_returns, periods_per_year
        ),
        "sharpe_ratio": sharpe_ratio(
            strategy_returns, risk_free_rate, periods_per_year
        ),
        "sortino_ratio": sortino_ratio(
            strategy_returns, risk_free_rate, periods_per_year
        ),
        "max_drawdown": max_drawdown(strategy_returns),
        "calmar_ratio": calmar_ratio(strategy_returns, periods_per_year),
        "win_rate": win_rate(strategy_returns),
        "profit_factor": profit_factor(strategy_returns),
        # Benchmark-relative
        "beta": beta(strategy_returns, benchmark_returns),
        "alpha": alpha(
            strategy_returns, benchmark_returns, risk_free_rate, periods_per_year
        ),
        "tracking_error": tracking_error(
            strategy_returns, benchmark_returns, periods_per_year
        ),
        "information_ratio": information_ratio(
            strategy_returns, benchmark_returns, periods_per_year
        ),
        "up_capture": up_capture(strategy_returns, benchmark_returns),
        "down_capture": down_capture(strategy_returns, benchmark_returns),
        # Distribution
        "skewness": skewness(strategy_returns),
        "excess_kurtosis": excess_kurtosis(strategy_returns),
        "value_at_risk_95": value_at_risk(strategy_returns, confidence=0.95),
        "cvar_95": cvar(strategy_returns, confidence=0.95),
        "tail_ratio": tail_ratio(strategy_returns),
        "expected_tail_ratio": expected_tail_ratio(strategy_returns),
        "best_day": best_day(strategy_returns),
        "worst_day": worst_day(strategy_returns),
    }
