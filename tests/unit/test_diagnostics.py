"""Tests for signal diagnostics module."""

import numpy as np
import pandas as pd
import pytest

from qre.analytics.validation.diagnostics import (
    cross_sectional_dispersion,
    signal_autocorrelation,
    signal_return_lead_lag,
    signal_turnover,
)


def _make_signal(n_days: int = 100, n_tickers: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]
    return pd.DataFrame(
        rng.standard_normal((n_days, n_tickers)),
        index=dates,
        columns=tickers,
    )


def _make_prices(n_days: int = 100, n_tickers: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]
    returns = rng.normal(0.0005, 0.02, (n_days, n_tickers))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


class TestSignalAutocorrelation:
    """Tests for signal_autocorrelation."""

    def test_output_shape(self):
        signal = _make_signal()
        result = signal_autocorrelation(signal, max_lag=10)
        assert result.shape == (10, 2)
        assert list(result.columns) == ["lag", "autocorr"]

    def test_lags_are_sequential(self):
        signal = _make_signal()
        result = signal_autocorrelation(signal, max_lag=5)
        assert list(result["lag"]) == [1, 2, 3, 4, 5]

    def test_constant_signal_has_autocorr_nan(self):
        """A constant signal has zero variance, so autocorr is undefined (NaN)."""
        dates = pd.bdate_range("2023-01-01", periods=50)
        tickers = [f"T{i}" for i in range(5)]
        signal = pd.DataFrame(1.0, index=dates, columns=tickers)
        result = signal_autocorrelation(signal, max_lag=3)
        assert result["autocorr"].isna().all()

    def test_highly_persistent_signal(self):
        """A random-walk signal should have very high lag-1 autocorrelation."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2023-01-01", periods=500)
        tickers = [f"T{i}" for i in range(10)]
        # Cumulative sum = random walk, very persistent
        signal = pd.DataFrame(
            np.cumsum(rng.standard_normal((500, 10)), axis=0),
            index=dates, columns=tickers,
        )
        result = signal_autocorrelation(signal, max_lag=5)
        assert result.loc[0, "autocorr"] > 0.95

    def test_iid_signal_low_autocorr(self):
        """An iid signal should have near-zero autocorrelation."""
        signal = _make_signal(n_days=500)
        result = signal_autocorrelation(signal, max_lag=5)
        assert abs(result["autocorr"].iloc[0]) < 0.15


class TestSignalTurnover:
    """Tests for signal_turnover."""

    def test_output_length(self):
        signal = _make_signal()
        result = signal_turnover(signal)
        assert len(result) == len(signal)

    def test_first_value_is_zero(self):
        """First row diff is NaN, so turnover is 0 (0 / gross)."""
        signal = _make_signal()
        result = signal_turnover(signal)
        assert result.iloc[0] == pytest.approx(0.0, abs=1e-10)

    def test_constant_signal_zero_turnover(self):
        """If signal doesn't change, turnover should be zero."""
        dates = pd.bdate_range("2023-01-01", periods=50)
        tickers = [f"T{i}" for i in range(5)]
        signal = pd.DataFrame(1.0, index=dates, columns=tickers)
        result = signal_turnover(signal)
        assert (result.dropna() == 0.0).all()

    def test_normalized_vs_raw(self):
        """Normalized turnover should be <= raw turnover (divided by gross)."""
        signal = _make_signal()
        norm = signal_turnover(signal, normalize=True)
        raw = signal_turnover(signal, normalize=False)
        # Raw is absolute change sum, normalized divides by gross exposure
        # Both should have same NaN pattern
        valid = norm.dropna().index.intersection(raw.dropna().index)
        assert (norm.loc[valid] <= raw.loc[valid] + 1e-10).all()

    def test_high_turnover_on_sign_flipping_signal(self):
        """A signal that flips sign every day should have high turnover."""
        dates = pd.bdate_range("2023-01-01", periods=50)
        tickers = [f"T{i}" for i in range(5)]
        vals = np.ones((50, 5))
        vals[1::2] = -1  # flip sign on odd rows
        signal = pd.DataFrame(vals, index=dates, columns=tickers)
        result = signal_turnover(signal, normalize=True)
        # Skip first row (diff is NaN → 0); remaining days should be 2.0
        assert result.iloc[1:].mean() == pytest.approx(2.0)


class TestCrossSectionalDispersion:
    """Tests for cross_sectional_dispersion."""

    def test_output_length(self):
        signal = _make_signal()
        result = cross_sectional_dispersion(signal)
        assert len(result) == len(signal)

    def test_uniform_signal_zero_dispersion(self):
        """If all tickers have the same value, dispersion is zero."""
        dates = pd.bdate_range("2023-01-01", periods=20)
        tickers = [f"T{i}" for i in range(5)]
        signal = pd.DataFrame(3.0, index=dates, columns=tickers)
        result = cross_sectional_dispersion(signal)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-10)

    def test_positive_dispersion_on_varied_signal(self):
        signal = _make_signal()
        result = cross_sectional_dispersion(signal)
        assert (result > 0).all()

    def test_higher_spread_means_higher_dispersion(self):
        """Scaling the signal should scale dispersion proportionally."""
        signal = _make_signal()
        disp_1x = cross_sectional_dispersion(signal)
        disp_3x = cross_sectional_dispersion(signal * 3)
        np.testing.assert_allclose(disp_3x.values, disp_1x.values * 3, atol=1e-10)


class TestSignalReturnLeadLag:
    """Tests for signal_return_lead_lag."""

    def test_output_length(self):
        signal = _make_signal()
        prices = _make_prices()
        result = signal_return_lead_lag(signal, prices, max_lead=5, max_lag=5)
        # -5, -4, ..., 0, ..., 4, 5 = 11 offsets
        assert len(result) == 11

    def test_offset_range(self):
        signal = _make_signal()
        prices = _make_prices()
        result = signal_return_lead_lag(signal, prices, max_lead=3, max_lag=2)
        assert list(result.index) == [-2, -1, 0, 1, 2, 3]

    def test_values_are_bounded(self):
        """Correlations should be between -1 and 1."""
        signal = _make_signal()
        prices = _make_prices()
        result = signal_return_lead_lag(signal, prices)
        assert (result.dropna() >= -1.0).all()
        assert (result.dropna() <= 1.0).all()

    def test_predictive_signal_has_positive_lead(self):
        """If signal == next-day return, offset +1 should have high correlation."""
        prices = _make_prices(n_days=200)
        daily_returns = prices.pct_change()
        # Signal = tomorrow's return (perfect foresight)
        signal = daily_returns.shift(-1)
        result = signal_return_lead_lag(signal, prices, max_lead=3, max_lag=3)
        # Offset +1 should be strongly positive
        assert result.loc[1] > 0.5

    def test_reactive_signal_has_negative_lag(self):
        """If signal == yesterday's return, offset -1 should have high correlation."""
        prices = _make_prices(n_days=200)
        daily_returns = prices.pct_change()
        # Signal = yesterday's return (pure momentum / reactive)
        signal = daily_returns.shift(1)
        result = signal_return_lead_lag(signal, prices, max_lead=3, max_lag=3)
        # Offset -1 should be strongly positive (signal correlates with past return)
        assert result.loc[-1] > 0.5
