"""Unit tests for qre.analytics.metrics."""

import numpy as np
import pandas as pd
import pytest

from qre.analytics import metrics


def _make_returns(values: list[float], start: str = "2024-01-02") -> pd.Series:
    """Helper to create a returns Series with a proper DatetimeIndex."""
    index = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=index)


# --- Sharpe Ratio ---


class TestSharpeRatio:
    def test_positive_returns(self):
        returns = _make_returns([0.01, 0.005, 0.02, 0.008, 0.015] * 20)
        assert metrics.sharpe_ratio(returns) > 0

    def test_zero_volatility_positive_mean_returns_inf(self):
        returns = _make_returns([0.01] * 50)
        assert metrics.sharpe_ratio(returns) == float("inf")

    def test_zero_volatility_negative_mean_returns_neg_inf(self):
        returns = _make_returns([-0.01] * 50)
        assert metrics.sharpe_ratio(returns) == float("-inf")

    def test_zero_volatility_zero_mean_returns_zero(self):
        returns = _make_returns([0.0] * 50)
        assert metrics.sharpe_ratio(returns) == 0.0

    def test_negative_returns(self):
        returns = _make_returns([-0.01, 0.005, -0.01, 0.005, -0.01] * 20)
        assert metrics.sharpe_ratio(returns) < 0

    def test_known_value(self):
        # Mean ≈ 0.001, std ≈ 0.01, Sharpe ≈ (0.001/0.01) * sqrt(252) ≈ 1.587
        rng = np.random.default_rng(42)
        values = rng.normal(0.001, 0.01, 252).tolist()
        returns = _make_returns(values)
        result = metrics.sharpe_ratio(returns)
        assert 0.5 < result < 3.0

    def test_with_risk_free_rate(self):
        rng = np.random.default_rng(42)
        values = rng.normal(0.001, 0.01, 252).tolist()
        returns = _make_returns(values)
        sharpe_no_rf = metrics.sharpe_ratio(returns, risk_free_rate=0.0)
        sharpe_with_rf = metrics.sharpe_ratio(returns, risk_free_rate=0.05)
        assert sharpe_with_rf < sharpe_no_rf

    def test_risk_free_compounding(self):
        # 5% annual rate with 252 periods should give a per-period rate of
        # (1.05)^(1/252) - 1 ≈ 0.0001938, NOT 0.05/252 ≈ 0.000198
        rf_per_period = (1.05) ** (1 / 252) - 1
        linear_approx = 0.05 / 252
        assert rf_per_period != pytest.approx(linear_approx, abs=1e-7)

    def test_too_few_returns(self):
        returns = _make_returns([0.01])
        assert metrics.sharpe_ratio(returns) == 0.0

    def test_custom_periods_per_year(self):
        rng = np.random.default_rng(42)
        values = rng.normal(0.001, 0.01, 100).tolist()
        returns = _make_returns(values)
        sharpe_daily = metrics.sharpe_ratio(returns, periods_per_year=252)
        sharpe_monthly = metrics.sharpe_ratio(returns, periods_per_year=12)
        # Lower periods_per_year → lower annualized Sharpe (sqrt effect)
        assert sharpe_monthly < sharpe_daily


# --- Sortino Ratio ---


class TestSortinoRatio:
    def test_higher_than_sharpe_when_skewed_up(self):
        # Big up days, small down days → Sortino > Sharpe
        values = [0.02, 0.03, -0.005, 0.025, -0.003] * 20
        returns = _make_returns(values)
        assert metrics.sortino_ratio(returns) > metrics.sharpe_ratio(returns)

    def test_all_positive_returns_inf(self):
        # No downside returns → infinite Sortino
        returns = _make_returns([0.01, 0.02, 0.015, 0.005] * 10)
        assert metrics.sortino_ratio(returns) == float("inf")

    def test_too_few_returns(self):
        returns = _make_returns([0.01])
        assert metrics.sortino_ratio(returns) == 0.0


# --- Max Drawdown ---


class TestMaxDrawdown:
    def test_no_drawdown(self):
        returns = _make_returns([0.01, 0.02, 0.01, 0.03, 0.01])
        assert metrics.max_drawdown(returns) == 0.0

    def test_known_drawdown(self):
        # Equity: 1.0 → 1.10 → 0.88 → 0.95
        # Peak = 1.10, Trough = 0.88, Drawdown = (0.88 - 1.10) / 1.10 = -0.2
        returns = _make_returns([0.10, -0.20, 0.0795])
        assert metrics.max_drawdown(returns) == pytest.approx(-0.20, abs=0.001)

    def test_negative_first_return(self):
        # Equity: 1.0 → 0.90 → 0.95
        # Peak = 1.0, Trough = 0.90, Drawdown = (0.90 - 1.0) / 1.0 = -0.10
        returns = _make_returns([-0.10, 0.0556])
        assert metrics.max_drawdown(returns) == pytest.approx(-0.10, abs=0.001)

    def test_total_loss(self):
        # Equity: 1.0 → 0.5 → 0.25 → 0.125
        # Peak = 1.0 (initial), Trough = 0.125
        # Drawdown = (0.125 - 1.0) / 1.0 = -0.875
        returns = _make_returns([-0.50, -0.50, -0.50])
        assert metrics.max_drawdown(returns) == pytest.approx(-0.875)

    def test_empty_returns(self):
        returns = _make_returns([])
        assert metrics.max_drawdown(returns) == 0.0

    def test_flat_curve(self):
        returns = _make_returns([0.0, 0.0, 0.0, 0.0])
        assert metrics.max_drawdown(returns) == 0.0


# --- Calmar Ratio ---


class TestCalmarRatio:
    def test_no_drawdown_returns_zero(self):
        returns = _make_returns([0.01, 0.02, 0.01, 0.03])
        assert metrics.calmar_ratio(returns) == 0.0

    def test_positive(self):
        values = [0.01, -0.02, 0.015, 0.01, -0.005] * 50
        returns = _make_returns(values)
        assert metrics.calmar_ratio(returns) > 0


# --- Annualized Return ---


class TestAnnualizedReturn:
    def test_one_year_of_daily_returns(self):
        # 252 days of 0.04% daily ≈ 10% annual
        returns = _make_returns([0.0004] * 252)
        result = metrics.annualized_return(returns)
        assert 0.09 < result < 0.12

    def test_negative_total(self):
        returns = _make_returns([-0.005] * 100)
        assert metrics.annualized_return(returns) < 0

    def test_empty(self):
        returns = _make_returns([])
        assert metrics.annualized_return(returns) == 0.0

    def test_consistent_with_periods_per_year(self):
        # Same data, different periods_per_year should give different results
        returns = _make_returns([0.001] * 100)
        r_daily = metrics.annualized_return(returns, periods_per_year=252)
        r_monthly = metrics.annualized_return(returns, periods_per_year=12)
        assert r_daily != r_monthly


# --- Annualized Volatility ---


class TestAnnualizedVolatility:
    def test_constant_returns_zero_vol(self):
        returns = _make_returns([0.01] * 50)
        assert metrics.annualized_volatility(returns) == 0.0

    def test_known_magnitude(self):
        # daily std ≈ 0.01, annual ≈ 0.01 * sqrt(252) ≈ 0.159
        rng = np.random.default_rng(99)
        values = rng.normal(0.0, 0.01, 252).tolist()
        returns = _make_returns(values)
        result = metrics.annualized_volatility(returns)
        assert 0.10 < result < 0.22


# --- Win Rate ---


class TestWinRate:
    def test_all_wins(self):
        returns = _make_returns([0.01, 0.02, 0.03])
        assert metrics.win_rate(returns) == 1.0

    def test_all_losses(self):
        returns = _make_returns([-0.01, -0.02, -0.03])
        assert metrics.win_rate(returns) == 0.0

    def test_mixed(self):
        returns = _make_returns([0.01, -0.01, 0.01, -0.01])
        assert metrics.win_rate(returns) == 0.5

    def test_zero_is_not_a_win(self):
        returns = _make_returns([0.0, 0.0, 0.01])
        assert metrics.win_rate(returns) == pytest.approx(1 / 3)

    def test_empty(self):
        returns = _make_returns([])
        assert metrics.win_rate(returns) == 0.0


# --- Profit Factor ---


class TestProfitFactor:
    def test_equal_gains_and_losses(self):
        returns = _make_returns([0.01, -0.01, 0.01, -0.01])
        assert metrics.profit_factor(returns) == pytest.approx(1.0)

    def test_more_gains(self):
        returns = _make_returns([0.02, -0.01, 0.02, -0.01])
        assert metrics.profit_factor(returns) == pytest.approx(2.0)

    def test_no_losses(self):
        returns = _make_returns([0.01, 0.02, 0.03])
        assert metrics.profit_factor(returns) == float("inf")

    def test_no_gains(self):
        returns = _make_returns([-0.01, -0.02])
        assert metrics.profit_factor(returns) == 0.0


# --- Summary ---


class TestSummary:
    def test_returns_all_keys(self):
        returns = _make_returns([0.01, -0.005, 0.008, -0.003] * 50)
        result = metrics.summary(returns)
        expected_keys = {
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "win_rate",
            "profit_factor",
        }
        assert set(result.keys()) == expected_keys

    def test_all_values_are_floats(self):
        returns = _make_returns([0.01, -0.005, 0.008, -0.003] * 50)
        result = metrics.summary(returns)
        for value in result.values():
            assert isinstance(value, float)

    def test_passes_periods_per_year_through(self):
        returns = _make_returns([0.01, -0.005, 0.008, -0.003] * 50)
        result_daily = metrics.summary(returns, periods_per_year=252)
        result_hourly = metrics.summary(returns, periods_per_year=252 * 7)
        assert result_daily["sharpe_ratio"] != result_hourly["sharpe_ratio"]


# --- Beta ---


class TestBeta:
    def test_perfect_correlation(self):
        # Strategy = 2x benchmark → beta = 2
        bench = _make_returns([0.01, -0.005, 0.008, -0.003, 0.01] * 20)
        strat = bench * 2
        assert metrics.beta(strat, bench) == pytest.approx(2.0, abs=0.01)

    def test_zero_correlation(self):
        # Uncorrelated returns → beta ≈ 0
        rng = np.random.default_rng(42)
        bench = _make_returns(rng.normal(0, 0.01, 200).tolist())
        strat = _make_returns(rng.normal(0, 0.01, 200).tolist())
        assert abs(metrics.beta(strat, bench)) < 0.3

    def test_negative_beta(self):
        # Strategy = -1x benchmark
        bench = _make_returns([0.01, -0.005, 0.008, -0.003, 0.01] * 20)
        strat = bench * -1
        assert metrics.beta(strat, bench) == pytest.approx(-1.0, abs=0.01)

    def test_zero_variance_benchmark(self):
        bench = _make_returns([0.01] * 50)
        strat = _make_returns([0.02, -0.01] * 25)
        assert metrics.beta(strat, bench) == 0.0

    def test_insufficient_data(self):
        bench = _make_returns([0.01])
        strat = _make_returns([0.02])
        assert metrics.beta(strat, bench) == 0.0


# --- Alpha ---


class TestAlpha:
    def test_pure_beta_no_alpha(self):
        # Strategy = 1x benchmark → alpha ≈ 0
        bench = _make_returns([0.01, -0.005, 0.008, -0.003] * 60)
        strat = bench * 1.0
        assert metrics.alpha(strat, bench) == pytest.approx(0.0, abs=0.01)

    def test_positive_alpha(self):
        # Strategy = benchmark + constant daily alpha
        bench = _make_returns([0.01, -0.005, 0.008, -0.003] * 60)
        strat = bench + 0.001  # ~25% annualized alpha
        result = metrics.alpha(strat, bench)
        assert result > 0.15

    def test_insufficient_data(self):
        bench = _make_returns([0.01])
        strat = _make_returns([0.02])
        assert metrics.alpha(strat, bench) == 0.0


# --- Tracking Error ---


class TestTrackingError:
    def test_identical_returns_zero_te(self):
        returns = _make_returns([0.01, -0.005, 0.008] * 50)
        assert metrics.tracking_error(returns, returns) == 0.0

    def test_different_returns_positive_te(self):
        bench = _make_returns([0.01, -0.005, 0.008, -0.003] * 50)
        strat = _make_returns([0.008, -0.003, 0.012, -0.001] * 50)
        assert metrics.tracking_error(strat, bench) > 0

    def test_insufficient_data(self):
        bench = _make_returns([0.01])
        strat = _make_returns([0.02])
        assert metrics.tracking_error(strat, bench) == 0.0


# --- Information Ratio ---


class TestInformationRatio:
    def test_positive_active_return(self):
        bench = _make_returns([0.01, -0.005, 0.008, -0.003] * 50)
        strat = bench + 0.001  # constant daily outperformance
        assert metrics.information_ratio(strat, bench) > 0

    def test_identical_returns_zero_ir(self):
        returns = _make_returns([0.01, -0.005, 0.008] * 50)
        assert metrics.information_ratio(returns, returns) == 0.0

    def test_insufficient_data(self):
        bench = _make_returns([0.01])
        strat = _make_returns([0.02])
        assert metrics.information_ratio(strat, bench) == 0.0


# --- Up/Down Capture ---


class TestUpDownCapture:
    def test_identical_returns_capture_one(self):
        returns = _make_returns([0.01, -0.005, 0.008, -0.003] * 50)
        assert metrics.up_capture(returns, returns) == pytest.approx(1.0)
        assert metrics.down_capture(returns, returns) == pytest.approx(1.0)

    def test_double_leverage(self):
        bench = _make_returns([0.01, -0.005, 0.008, -0.003] * 50)
        strat = bench * 2
        assert metrics.up_capture(strat, bench) == pytest.approx(2.0, abs=0.01)
        assert metrics.down_capture(strat, bench) == pytest.approx(2.0, abs=0.01)

    def test_no_up_days(self):
        bench = _make_returns([-0.01, -0.02, -0.005] * 10)
        strat = _make_returns([-0.005, -0.01, -0.002] * 10)
        assert metrics.up_capture(strat, bench) == 0.0

    def test_no_down_days(self):
        bench = _make_returns([0.01, 0.02, 0.005] * 10)
        strat = _make_returns([0.005, 0.01, 0.002] * 10)
        assert metrics.down_capture(strat, bench) == 0.0


# --- Skewness ---


class TestSkewness:
    def test_symmetric_near_zero(self):
        rng = np.random.default_rng(42)
        returns = _make_returns(rng.normal(0, 0.01, 1000).tolist())
        assert abs(metrics.skewness(returns)) < 0.3

    def test_negative_skew(self):
        # Large losses, small gains
        values = [0.005] * 95 + [-0.10] * 5
        returns = _make_returns(values)
        assert metrics.skewness(returns) < 0

    def test_insufficient_data(self):
        returns = _make_returns([0.01, 0.02])
        assert metrics.skewness(returns) == 0.0


# --- Excess Kurtosis ---


class TestExcessKurtosis:
    def test_normal_near_zero(self):
        rng = np.random.default_rng(42)
        returns = _make_returns(rng.normal(0, 0.01, 5000).tolist())
        assert abs(metrics.excess_kurtosis(returns)) < 0.5

    def test_fat_tails_positive(self):
        # t-distribution has fat tails
        rng = np.random.default_rng(42)
        returns = _make_returns((rng.standard_t(df=3, size=1000) * 0.01).tolist())
        assert metrics.excess_kurtosis(returns) > 1.0

    def test_insufficient_data(self):
        returns = _make_returns([0.01, 0.02, 0.03])
        assert metrics.excess_kurtosis(returns) == 0.0


# --- Value at Risk ---


class TestValueAtRisk:
    def test_known_quantile(self):
        # Sorted: [-0.05, -0.02, 0.01, 0.01, ...] with 100 obs
        # 5th percentile should be around the worst values
        rng = np.random.default_rng(42)
        returns = _make_returns(rng.normal(0, 0.01, 1000).tolist())
        var = metrics.value_at_risk(returns, confidence=0.95)
        assert var < 0  # It's a loss
        assert var > -0.05  # Shouldn't be extreme for std=0.01

    def test_higher_confidence_worse_var(self):
        rng = np.random.default_rng(42)
        returns = _make_returns(rng.normal(0, 0.01, 500).tolist())
        var_95 = metrics.value_at_risk(returns, confidence=0.95)
        var_99 = metrics.value_at_risk(returns, confidence=0.99)
        assert var_99 < var_95  # 99% VaR is more negative

    def test_insufficient_data(self):
        returns = _make_returns([0.01])
        assert metrics.value_at_risk(returns) == 0.0


# --- CVaR ---


class TestCVaR:
    def test_worse_than_var(self):
        rng = np.random.default_rng(42)
        returns = _make_returns(rng.normal(0, 0.01, 500).tolist())
        var = metrics.value_at_risk(returns, confidence=0.95)
        cvar_val = metrics.cvar(returns, confidence=0.95)
        assert cvar_val <= var  # CVaR is always worse (more negative)

    def test_negative(self):
        rng = np.random.default_rng(42)
        returns = _make_returns(rng.normal(0, 0.01, 500).tolist())
        assert metrics.cvar(returns) < 0

    def test_insufficient_data(self):
        returns = _make_returns([0.01])
        assert metrics.cvar(returns) == 0.0


# --- Tail Ratio ---


class TestTailRatio:
    def test_symmetric_near_one(self):
        rng = np.random.default_rng(42)
        returns = _make_returns(rng.normal(0, 0.01, 5000).tolist())
        assert metrics.tail_ratio(returns) == pytest.approx(1.0, abs=0.3)

    def test_positive_skew_above_one(self):
        # Big positive outliers, small negative ones
        values = [-0.005] * 90 + [0.10] * 10
        returns = _make_returns(values)
        assert metrics.tail_ratio(returns) > 1.0

    def test_insufficient_data(self):
        returns = _make_returns([0.01])
        assert metrics.tail_ratio(returns) == 0.0


# --- Expected Tail Ratio ---


class TestExpectedTailRatio:
    def test_symmetric_near_one(self):
        rng = np.random.default_rng(42)
        returns = _make_returns(rng.normal(0, 0.01, 5000).tolist())
        assert metrics.expected_tail_ratio(returns) == pytest.approx(1.0, abs=0.3)

    def test_positive_skew_above_one(self):
        values = [-0.005] * 90 + [0.10] * 10
        returns = _make_returns(values)
        assert metrics.expected_tail_ratio(returns) > 1.0

    def test_insufficient_data(self):
        returns = _make_returns([0.01])
        assert metrics.expected_tail_ratio(returns) == 0.0


# --- Best/Worst Day ---


class TestBestWorstDay:
    def test_best_day(self):
        returns = _make_returns([0.01, -0.02, 0.05, -0.01])
        assert metrics.best_day(returns) == pytest.approx(0.05)

    def test_worst_day(self):
        returns = _make_returns([0.01, -0.02, 0.05, -0.01])
        assert metrics.worst_day(returns) == pytest.approx(-0.02)

    def test_empty(self):
        returns = _make_returns([])
        assert metrics.best_day(returns) == 0.0
        assert metrics.worst_day(returns) == 0.0


# --- Full Summary ---


class TestFullSummary:
    def test_contains_all_keys(self):
        bench = _make_returns([0.005, -0.003, 0.004, -0.002] * 50)
        strat = _make_returns([0.01, -0.005, 0.008, -0.003] * 50)
        result = metrics.full_summary(strat, bench)
        expected_keys = {
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "win_rate",
            "profit_factor",
            "beta",
            "alpha",
            "tracking_error",
            "information_ratio",
            "up_capture",
            "down_capture",
            "skewness",
            "excess_kurtosis",
            "value_at_risk_95",
            "cvar_95",
            "tail_ratio",
            "expected_tail_ratio",
            "best_day",
            "worst_day",
        }
        assert set(result.keys()) == expected_keys

    def test_all_values_are_floats(self):
        bench = _make_returns([0.005, -0.003, 0.004, -0.002] * 50)
        strat = _make_returns([0.01, -0.005, 0.008, -0.003] * 50)
        result = metrics.full_summary(strat, bench)
        for value in result.values():
            assert isinstance(value, float)
