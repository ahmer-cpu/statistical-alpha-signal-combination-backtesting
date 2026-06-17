"""Tests for IC analysis module."""

import math

import numpy as np
import pandas as pd
import pytest

from qre.analytics.ic import (
    block_bootstrap_ci,
    compute_ic,
    ic_decay,
    ic_summary,
    newey_west_t_stat,
    non_overlapping_ic,
)


def _make_panel(n_days: int = 100, n_tickers: int = 10, seed: int = 42):
    """Generate signal and forward returns with known correlation structure."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]

    # Signal: random
    signal = pd.DataFrame(
        rng.standard_normal((n_days, n_tickers)),
        index=dates, columns=tickers,
    )
    # Forward returns: signal + noise (so IC should be positive)
    noise = rng.standard_normal((n_days, n_tickers)) * 3
    fwd_returns = pd.DataFrame(
        signal.values + noise,
        index=dates, columns=tickers,
    )
    return signal, fwd_returns


class TestComputeIC:
    """Tests for daily Spearman IC computation."""

    def test_perfect_positive_signal(self):
        """If signal == forward returns, IC should be 1.0 every day."""
        dates = pd.bdate_range("2023-01-01", periods=20)
        tickers = [f"T{i}" for i in range(10)]
        data = np.random.default_rng(0).standard_normal((20, 10))
        signal = pd.DataFrame(data, index=dates, columns=tickers)
        fwd = signal.copy()

        ic = compute_ic(signal, fwd)
        assert (ic.dropna() == 1.0).all()

    def test_perfect_negative_signal(self):
        """If signal == -forward returns, IC should be -1.0 every day."""
        dates = pd.bdate_range("2023-01-01", periods=20)
        tickers = [f"T{i}" for i in range(10)]
        data = np.random.default_rng(0).standard_normal((20, 10))
        signal = pd.DataFrame(data, index=dates, columns=tickers)
        fwd = -signal

        ic = compute_ic(signal, fwd)
        assert (ic.dropna() == -1.0).all()

    def test_positive_ic_with_correlated_data(self):
        signal, fwd = _make_panel()
        ic = compute_ic(signal, fwd)
        # Mean IC should be positive since fwd = signal + noise
        assert ic.mean() > 0

    def test_output_length_matches_input(self):
        signal, fwd = _make_panel(n_days=50)
        ic = compute_ic(signal, fwd)
        assert len(ic) == 50

    def test_min_periods_masks_sparse_dates(self):
        """Dates with fewer than min_periods valid values should be NaN."""
        dates = pd.bdate_range("2023-01-01", periods=10)
        tickers = [f"T{i}" for i in range(6)]
        signal = pd.DataFrame(
            np.random.default_rng(0).standard_normal((10, 6)),
            index=dates, columns=tickers,
        )
        fwd = pd.DataFrame(
            np.random.default_rng(1).standard_normal((10, 6)),
            index=dates, columns=tickers,
        )
        # Set most tickers to NaN on first date → fewer than 5 valid
        signal.iloc[0, :4] = np.nan

        ic = compute_ic(signal, fwd, min_periods=5)
        assert pd.isna(ic.iloc[0])
        # Other dates should be valid
        assert ic.iloc[1:].notna().all()

    def test_one_sided_nan_does_not_corrupt_ranks(self):
        """A name missing in only one frame must not distort other ranks.

        Ranks must be computed over the pairwise-valid set. Here the three
        valid pairs (A, B, D) are perfectly monotonic, so IC must be exactly
        1.0 — even though C has a signal but no forward return. Ranking the
        full row first (and letting corrwith drop C) would inflate D's rank
        and return ~0.982 instead.
        """
        dates = pd.bdate_range("2023-01-01", periods=1)
        signal = pd.DataFrame(
            {"A": [1.0], "B": [2.0], "C": [3.0], "D": [4.0]}, index=dates
        )
        fwd = pd.DataFrame(
            {"A": [0.1], "B": [0.2], "C": [np.nan], "D": [0.4]}, index=dates
        )
        ic = compute_ic(signal, fwd, min_periods=3)
        assert ic.iloc[0] == pytest.approx(1.0)


class TestAlignment:
    """IC should handle misaligned inputs correctly."""

    def test_different_column_order(self):
        """Shuffled columns should produce the same IC as the original order."""
        dates = pd.bdate_range("2023-01-01", periods=20)
        tickers = [f"T{i}" for i in range(8)]
        rng = np.random.default_rng(0)

        signal = pd.DataFrame(
            rng.standard_normal((20, 8)), index=dates, columns=tickers,
        )
        fwd = pd.DataFrame(
            rng.standard_normal((20, 8)), index=dates, columns=tickers,
        )

        ic_ordered = compute_ic(signal, fwd)
        # Shuffle fwd columns — alignment should produce identical IC
        fwd_shuffled = fwd[list(reversed(tickers))]
        ic_shuffled = compute_ic(signal, fwd_shuffled)

        pd.testing.assert_series_equal(ic_ordered, ic_shuffled)

    def test_partial_date_overlap(self):
        dates_a = pd.bdate_range("2023-01-01", periods=30)
        dates_b = pd.bdate_range("2023-01-15", periods=30)
        tickers = [f"T{i}" for i in range(8)]
        rng = np.random.default_rng(0)

        signal = pd.DataFrame(
            rng.standard_normal((30, 8)), index=dates_a, columns=tickers
        )
        fwd = pd.DataFrame(
            rng.standard_normal((30, 8)), index=dates_b, columns=tickers
        )

        ic = compute_ic(signal, fwd)
        # Should only have values for the overlapping dates
        assert len(ic) < 30


class TestICSummary:
    """Tests for ic_summary statistics."""

    def test_keys_present(self):
        signal, fwd = _make_panel()
        ic = compute_ic(signal, fwd)
        stats = ic_summary(ic, signal, fwd)
        expected_keys = {
            "mean", "std", "icir", "hit_rate",
            "t_stat", "n_dates", "avg_coverage",
        }
        assert set(stats.keys()) == expected_keys

    def test_icir_equals_mean_over_std(self):
        signal, fwd = _make_panel()
        ic = compute_ic(signal, fwd)
        stats = ic_summary(ic)
        expected_icir = stats["mean"] / stats["std"]
        assert pytest.approx(stats["icir"]) == expected_icir

    def test_t_stat_formula(self):
        signal, fwd = _make_panel()
        ic = compute_ic(signal, fwd)
        stats = ic_summary(ic)
        expected_t = stats["mean"] / stats["std"] * stats["n_dates"] ** 0.5
        assert pytest.approx(stats["t_stat"]) == expected_t

    def test_empty_series(self):
        ic = pd.Series(dtype=float)
        stats = ic_summary(ic)
        assert math.isnan(stats["mean"])
        assert math.isnan(stats["icir"])
        assert stats["n_dates"] == 0

    def test_all_nan_series(self):
        ic = pd.Series([np.nan, np.nan, np.nan])
        stats = ic_summary(ic)
        assert math.isnan(stats["mean"])
        assert stats["n_dates"] == 0

    def test_avg_coverage_with_data(self):
        signal, fwd = _make_panel()
        ic = compute_ic(signal, fwd)
        stats = ic_summary(ic, signal, fwd)
        assert stats["avg_coverage"] == 10.0  # no NaNs in synthetic data

    def test_avg_coverage_without_data_is_nan(self):
        signal, fwd = _make_panel()
        ic = compute_ic(signal, fwd)
        stats = ic_summary(ic)
        assert math.isnan(stats["avg_coverage"])


class TestICSummaryExtended:
    """Tests for ic_summary with horizon and bootstrap parameters."""

    def test_horizon_adds_nw_keys(self):
        signal, fwd = _make_panel()
        ic = compute_ic(signal, fwd)
        stats = ic_summary(ic, horizon=5)
        assert "t_stat_nw" in stats
        assert "se_nw" in stats
        assert "bandwidth" in stats

    def test_horizon_none_omits_nw_keys(self):
        signal, fwd = _make_panel()
        ic = compute_ic(signal, fwd)
        stats = ic_summary(ic)
        assert "t_stat_nw" not in stats

    def test_bootstrap_adds_ci_keys(self):
        signal, fwd = _make_panel()
        ic = compute_ic(signal, fwd)
        stats = ic_summary(ic, n_bootstrap=500, bootstrap_seed=42)
        assert "ci_lower" in stats
        assert "ci_upper" in stats
        assert "se_bootstrap" in stats

    def test_bootstrap_zero_omits_ci_keys(self):
        signal, fwd = _make_panel()
        ic = compute_ic(signal, fwd)
        stats = ic_summary(ic, n_bootstrap=0)
        assert "ci_lower" not in stats

    def test_nw_t_stat_smaller_than_naive_on_autocorrelated(self):
        """Newey-West should shrink t-stat when IC is strongly autocorrelated."""
        # Create explicitly autocorrelated IC (AR(1) with rho=0.9)
        rng = np.random.default_rng(42)
        n = 200
        ic_vals = np.empty(n)
        ic_vals[0] = 0.05 + rng.normal() * 0.01
        for i in range(1, n):
            ic_vals[i] = 0.9 * ic_vals[i - 1] + rng.normal() * 0.01
        ic = pd.Series(ic_vals)
        stats = ic_summary(ic, horizon=21)
        # NW should produce smaller t-stat on strongly autocorrelated data
        assert abs(stats["t_stat_nw"]) < abs(stats["t_stat"])

    def test_backward_compat_keys_unchanged(self):
        """Original 7 keys should always be present regardless of new params."""
        signal, fwd = _make_panel()
        ic = compute_ic(signal, fwd)
        base_keys = {
            "mean", "std", "icir", "hit_rate", "t_stat", "n_dates", "avg_coverage",
        }
        stats = ic_summary(ic, signal, fwd, horizon=5, n_bootstrap=500)
        assert base_keys.issubset(stats.keys())

    def test_empty_series_with_horizon_and_bootstrap(self):
        ic = pd.Series(dtype=float)
        stats = ic_summary(ic, horizon=5, n_bootstrap=100)
        assert math.isnan(stats["t_stat_nw"])
        assert math.isnan(stats["ci_lower"])
        assert stats["n_dates"] == 0


class TestNeweyWestTStat:
    """Tests for newey_west_t_stat."""

    def test_returns_expected_keys(self):
        signal, fwd = _make_panel()
        ic = compute_ic(signal, fwd)
        result = newey_west_t_stat(ic)
        expected = {"t_stat_naive", "t_stat_nw", "se_naive", "se_nw", "bandwidth"}
        assert set(result.keys()) == expected

    def test_nw_se_larger_on_autocorrelated_data(self):
        """HAC SE should be >= naive SE when data is positively autocorrelated."""
        # Create strongly autocorrelated IC series (AR(1) with rho=0.9)
        rng = np.random.default_rng(42)
        n = 200
        ic_vals = np.empty(n)
        ic_vals[0] = rng.normal()
        for i in range(1, n):
            ic_vals[i] = 0.9 * ic_vals[i - 1] + rng.normal() * 0.1
        ic = pd.Series(ic_vals)

        result = newey_west_t_stat(ic)
        assert result["se_nw"] > result["se_naive"]

    def test_nw_matches_naive_on_iid(self):
        """On truly iid data, NW SE should be close to naive SE."""
        rng = np.random.default_rng(123)
        ic = pd.Series(rng.standard_normal(500))
        result = newey_west_t_stat(ic)
        # Should be within 30% of each other on iid data
        ratio = result["se_nw"] / result["se_naive"]
        assert 0.7 < ratio < 1.3

    def test_explicit_bandwidth(self):
        signal, fwd = _make_panel()
        ic = compute_ic(signal, fwd)
        result = newey_west_t_stat(ic, bandwidth=10)
        assert result["bandwidth"] == 10

    def test_single_observation_returns_nan(self):
        ic = pd.Series([0.5])
        result = newey_west_t_stat(ic)
        assert math.isnan(result["t_stat_nw"])

    def test_empty_series_returns_nan(self):
        ic = pd.Series(dtype=float)
        result = newey_west_t_stat(ic)
        assert math.isnan(result["t_stat_nw"])

    def test_all_nan_returns_nan(self):
        ic = pd.Series([np.nan, np.nan, np.nan])
        result = newey_west_t_stat(ic)
        assert math.isnan(result["t_stat_nw"])


class TestNonOverlappingIC:
    """Tests for non_overlapping_ic."""

    def test_perfect_signal_gives_ic_one(self):
        """If signal == forward returns, non-overlapping IC should also be 1.0."""
        dates = pd.bdate_range("2023-01-01", periods=100)
        tickers = [f"T{i}" for i in range(10)]
        data = np.random.default_rng(0).standard_normal((100, 10))
        signal = pd.DataFrame(data, index=dates, columns=tickers)
        fwd = signal.copy()

        ic = non_overlapping_ic(signal, fwd, horizon=5)
        assert (ic.dropna() == 1.0).all()

    def test_fewer_observations_than_daily(self):
        """Non-overlapping IC should have roughly n_days / horizon observations."""
        signal, fwd = _make_panel(n_days=100)
        ic = non_overlapping_ic(signal, fwd, horizon=10)
        # 100 days / 10 = 10 sampled dates
        assert len(ic) == 10

    def test_horizon_1_matches_compute_ic(self):
        """horizon=1 should produce same result as compute_ic (every day sampled)."""
        signal, fwd = _make_panel()
        ic_non_overlap = non_overlapping_ic(signal, fwd, horizon=1)
        ic_daily = compute_ic(signal, fwd)
        pd.testing.assert_series_equal(ic_non_overlap, ic_daily)

    def test_respects_min_periods(self):
        """Dates with too few valid tickers should be NaN."""
        dates = pd.bdate_range("2023-01-01", periods=20)
        tickers = [f"T{i}" for i in range(6)]
        rng = np.random.default_rng(0)
        signal = pd.DataFrame(
            rng.standard_normal((20, 6)), index=dates, columns=tickers
        )
        fwd = pd.DataFrame(rng.standard_normal((20, 6)), index=dates, columns=tickers)
        # Kill most tickers on the first date
        signal.iloc[0, :5] = np.nan
        ic = non_overlapping_ic(signal, fwd, horizon=5, min_periods=5)
        assert pd.isna(ic.iloc[0])


class TestBlockBootstrapCI:
    """Tests for block_bootstrap_ci."""

    def test_returns_expected_keys(self):
        rng = np.random.default_rng(42)
        ic = pd.Series(rng.standard_normal(100))
        result = block_bootstrap_ci(ic, n_bootstrap=200, seed=0)
        expected = {"point_estimate", "ci_lower", "ci_upper", "se_bootstrap"}
        assert set(result.keys()) == expected

    def test_ci_contains_point_estimate(self):
        rng = np.random.default_rng(42)
        ic = pd.Series(rng.standard_normal(100))
        result = block_bootstrap_ci(ic, n_bootstrap=5000, seed=0)
        assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]

    def test_ci_contains_true_mean_of_iid(self):
        """For a known-mean iid series, the 95% CI should contain the true mean."""
        rng = np.random.default_rng(42)
        ic = pd.Series(rng.normal(loc=0.05, scale=0.1, size=500))
        result = block_bootstrap_ci(ic, n_bootstrap=5000, seed=0)
        assert result["ci_lower"] < 0.05 < result["ci_upper"]

    def test_wider_ci_on_autocorrelated_data(self):
        """Block bootstrap CI should be wider than naive CI on autocorrelated data."""
        rng = np.random.default_rng(42)
        n = 200
        ic_vals = np.empty(n)
        ic_vals[0] = rng.normal()
        for i in range(1, n):
            ic_vals[i] = 0.9 * ic_vals[i - 1] + rng.normal() * 0.1
        ic = pd.Series(ic_vals)

        boot = block_bootstrap_ci(ic, n_bootstrap=5000, seed=0)
        boot_width = boot["ci_upper"] - boot["ci_lower"]
        # Naive CI: mean ± 1.96 * std / sqrt(n)
        naive_width = 2 * 1.96 * ic.std() / len(ic) ** 0.5
        assert boot_width > naive_width

    def test_reproducible_with_seed(self):
        ic = pd.Series(np.random.default_rng(0).standard_normal(100))
        r1 = block_bootstrap_ci(ic, n_bootstrap=500, seed=42)
        r2 = block_bootstrap_ci(ic, n_bootstrap=500, seed=42)
        assert r1["ci_lower"] == r2["ci_lower"]
        assert r1["ci_upper"] == r2["ci_upper"]

    def test_custom_stat_fn(self):
        ic = pd.Series(np.random.default_rng(0).standard_normal(100))
        result = block_bootstrap_ci(ic, stat_fn=np.median, n_bootstrap=500, seed=0)
        assert pytest.approx(result["point_estimate"], abs=0.01) == float(
            np.median(ic.values)
        )

    def test_single_observation_returns_nan(self):
        ic = pd.Series([0.5])
        result = block_bootstrap_ci(ic, n_bootstrap=100, seed=0)
        assert math.isnan(result["ci_lower"])

    def test_empty_series_returns_nan(self):
        ic = pd.Series(dtype=float)
        result = block_bootstrap_ci(ic, n_bootstrap=100, seed=0)
        assert math.isnan(result["point_estimate"])

    def test_zero_bootstrap_returns_nan_not_crash(self):
        """n_bootstrap=0 (skip-bootstrap convention) must return the NaN result
        on a live series, not crash on np.percentile of an empty array."""
        ic = pd.Series(np.random.default_rng(0).standard_normal(100))
        result = block_bootstrap_ci(ic, n_bootstrap=0, seed=0)
        assert math.isnan(result["ci_lower"])
        assert math.isnan(result["ci_upper"])


class TestICDecay:
    """Tests for IC decay curve."""

    def test_output_length(self):
        signal, _ = _make_panel(n_days=100, n_tickers=10)
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(
                np.random.default_rng(0).normal(0, 0.02, (100, 10)), axis=0
            )),
            index=signal.index, columns=signal.columns,
        )
        decay = ic_decay(signal, prices, max_lag=10)
        assert len(decay) == 10
        assert list(decay.index) == list(range(1, 11))

    def test_decay_values_are_finite(self):
        signal, _ = _make_panel(n_days=100, n_tickers=10)
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(
                np.random.default_rng(0).normal(0, 0.02, (100, 10)), axis=0
            )),
            index=signal.index, columns=signal.columns,
        )
        decay = ic_decay(signal, prices, max_lag=5)
        assert decay.notna().all()
