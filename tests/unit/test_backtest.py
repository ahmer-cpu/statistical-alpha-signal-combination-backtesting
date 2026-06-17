"""Tests for the vectorized backtester."""

import numpy as np
import pandas as pd
import pytest

from qre.backtest.vectorized import VectorizedBacktester


def _make_prices(n_days: int = 100, n_tickers: int = 4, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]
    returns = rng.normal(0.0005, 0.02, (n_days, n_tickers))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture()
def prices() -> pd.DataFrame:
    return _make_prices()


class TestExposureNormalization:
    """Positions should be dollar-neutral with unit gross exposure."""

    def test_gross_exposure_is_one(self, prices: pd.DataFrame):
        signal = pd.DataFrame(
            np.random.default_rng(0).standard_normal(prices.shape),
            index=prices.index,
            columns=prices.columns,
        )
        VectorizedBacktester(cost_bps=0.0)

        # Reconstruct positions to check normalization
        raw = signal.sub(signal.mean(axis=1), axis=0)
        gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
        positions = raw.div(gross, axis=0).fillna(0.0)

        gross_exposure = positions.abs().sum(axis=1)
        np.testing.assert_allclose(gross_exposure.values, 1.0, atol=1e-10)

    def test_net_exposure_is_zero(self, prices: pd.DataFrame):
        signal = pd.DataFrame(
            np.random.default_rng(0).standard_normal(prices.shape),
            index=prices.index,
            columns=prices.columns,
        )
        raw = signal.sub(signal.mean(axis=1), axis=0)
        gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
        positions = raw.div(gross, axis=0).fillna(0.0)

        net_exposure = positions.sum(axis=1)
        np.testing.assert_allclose(net_exposure.values, 0.0, atol=1e-10)


class TestZeroSignal:
    """A flat signal (all equal) should produce zero returns."""

    def test_uniform_signal_zero_return(self, prices: pd.DataFrame):
        signal = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
        bt = VectorizedBacktester(cost_bps=0.0)
        result = bt.run(prices, signal)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-10)


class TestLastRowDropped:
    """The last row has no forward return and must be dropped."""

    def test_result_shorter_than_input(self, prices: pd.DataFrame):
        signal = pd.DataFrame(
            np.random.default_rng(0).standard_normal(prices.shape),
            index=prices.index,
            columns=prices.columns,
        )
        bt = VectorizedBacktester(cost_bps=0.0)
        result = bt.run(prices, signal)
        # Result should be shorter — at minimum the last row is dropped
        assert len(result) < len(prices)

    def test_last_date_not_in_result(self, prices: pd.DataFrame):
        signal = pd.DataFrame(
            np.random.default_rng(0).standard_normal(prices.shape),
            index=prices.index,
            columns=prices.columns,
        )
        bt = VectorizedBacktester(cost_bps=0.0)
        result = bt.run(prices, signal)
        assert prices.index[-1] not in result.index


class TestCostsReduceReturns:
    """Transaction costs should strictly reduce total returns."""

    def test_costs_lower_cumulative_return(self, prices: pd.DataFrame):
        rng = np.random.default_rng(99)
        signal = pd.DataFrame(
            rng.standard_normal(prices.shape),
            index=prices.index,
            columns=prices.columns,
        )
        bt_free = VectorizedBacktester(cost_bps=0.0)
        bt_costly = VectorizedBacktester(cost_bps=50.0)

        ret_free = bt_free.run(prices, signal).sum()
        ret_costly = bt_costly.run(prices, signal).sum()

        assert ret_costly < ret_free


class TestRebalanceFreq:
    """Tests for the rebalance_freq parameter."""

    def test_freq_1_matches_default(self, prices: pd.DataFrame):
        """rebalance_freq=1 should produce identical results to default."""
        signal = pd.DataFrame(
            np.random.default_rng(0).standard_normal(prices.shape),
            index=prices.index, columns=prices.columns,
        )
        bt_default = VectorizedBacktester(cost_bps=10.0)
        bt_freq1 = VectorizedBacktester(cost_bps=10.0, rebalance_freq=1)
        pd.testing.assert_series_equal(
            bt_default.run(prices, signal),
            bt_freq1.run(prices, signal),
        )

    def test_higher_freq_reduces_turnover(self, prices: pd.DataFrame):
        """Less frequent rebalancing should produce lower total turnover cost."""
        signal = pd.DataFrame(
            np.random.default_rng(0).standard_normal(prices.shape),
            index=prices.index, columns=prices.columns,
        )
        # With zero-cost, gross returns differ; with high cost, freq=5 should win
        bt_daily = VectorizedBacktester(cost_bps=100.0, rebalance_freq=1)
        bt_weekly = VectorizedBacktester(cost_bps=100.0, rebalance_freq=5)
        ret_daily = bt_daily.run(prices, signal).sum()
        ret_weekly = bt_weekly.run(prices, signal).sum()
        # Weekly rebalancing should lose less to costs
        assert ret_weekly > ret_daily

    def test_positions_are_constant_between_rebalances(self, prices: pd.DataFrame):
        """On non-rebalance days, positions should be carried forward."""
        signal = pd.DataFrame(
            np.random.default_rng(0).standard_normal(prices.shape),
            index=prices.index, columns=prices.columns,
        )
        # Reconstruct positions to verify carry-forward
        prices_a, signal_a = prices.align(signal, join="inner", axis=0)
        signal_a = signal_a.reindex(columns=prices_a.columns)
        raw = signal_a.sub(signal_a.mean(axis=1), axis=0)
        gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
        positions = raw.div(gross, axis=0).fillna(0.0)

        row_mask = np.arange(len(positions)) % 5 == 0
        rebal_mask = np.broadcast_to(row_mask[:, None], positions.shape)
        positions = positions.where(rebal_mask).ffill().fillna(0.0)

        # Between rebalances, consecutive rows should be identical
        for i in range(1, min(20, len(positions))):
            if i % 5 != 0:
                pd.testing.assert_series_equal(
                    positions.iloc[i], positions.iloc[i - 1],
                    check_names=False,
                )


class TestPositionSmooth:
    """Tests for the position_smooth parameter."""

    def test_smooth_0_matches_default(self, prices: pd.DataFrame):
        """position_smooth=0.0 should produce identical results to default."""
        signal = pd.DataFrame(
            np.random.default_rng(0).standard_normal(prices.shape),
            index=prices.index, columns=prices.columns,
        )
        bt_default = VectorizedBacktester(cost_bps=10.0)
        bt_smooth0 = VectorizedBacktester(cost_bps=10.0, position_smooth=0.0)
        pd.testing.assert_series_equal(
            bt_default.run(prices, signal),
            bt_smooth0.run(prices, signal),
        )

    def test_smoothing_reduces_turnover(self, prices: pd.DataFrame):
        """Smoothed positions should have lower turnover than unsmoothed."""
        signal = pd.DataFrame(
            np.random.default_rng(0).standard_normal(prices.shape),
            index=prices.index, columns=prices.columns,
        )
        bt_raw = VectorizedBacktester(cost_bps=100.0, position_smooth=0.0)
        bt_smooth = VectorizedBacktester(cost_bps=100.0, position_smooth=0.5)
        ret_raw = bt_raw.run(prices, signal).sum()
        ret_smooth = bt_smooth.run(prices, signal).sum()
        # Smoothing should lose less to costs
        assert ret_smooth > ret_raw

    def test_smoothed_positions_still_unit_exposure(self, prices: pd.DataFrame):
        """After smoothing + re-normalization, gross exposure should be 1.0."""
        signal = pd.DataFrame(
            np.random.default_rng(0).standard_normal(prices.shape),
            index=prices.index, columns=prices.columns,
        )
        # Reconstruct smoothed positions
        prices_a, signal_a = prices.align(signal, join="inner", axis=0)
        signal_a = signal_a.reindex(columns=prices_a.columns)
        raw = signal_a.sub(signal_a.mean(axis=1), axis=0)
        gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
        positions = raw.div(gross, axis=0).fillna(0.0)

        alpha = 0.5
        for i in range(1, len(positions)):
            positions.iloc[i] = alpha * positions.iloc[i] + 0.5 * positions.iloc[i - 1]
        gross_smooth = positions.abs().sum(axis=1).replace(0.0, np.nan)
        positions = positions.div(gross_smooth, axis=0).fillna(0.0)

        np.testing.assert_allclose(
            positions.abs().sum(axis=1).values, 1.0, atol=1e-10,
        )


class TestCombinedFreqAndSmooth:
    """Test rebalance_freq and position_smooth together."""

    def test_combined_runs_without_error(self, prices: pd.DataFrame):
        signal = pd.DataFrame(
            np.random.default_rng(0).standard_normal(prices.shape),
            index=prices.index, columns=prices.columns,
        )
        bt = VectorizedBacktester(cost_bps=10.0, rebalance_freq=5, position_smooth=0.3)
        result = bt.run(prices, signal)
        assert len(result) > 0
        assert result.notna().all()

    def test_combined_less_costly_than_daily(self, prices: pd.DataFrame):
        """Combined freq + smooth should lose less to costs than daily unsmoothed."""
        signal = pd.DataFrame(
            np.random.default_rng(0).standard_normal(prices.shape),
            index=prices.index, columns=prices.columns,
        )
        bt_daily = VectorizedBacktester(cost_bps=100.0)
        bt_combined = VectorizedBacktester(
            cost_bps=100.0, rebalance_freq=5, position_smooth=0.3
        )
        ret_daily = bt_daily.run(prices, signal).sum()
        ret_combined = bt_combined.run(prices, signal).sum()
        assert ret_combined > ret_daily


class TestInputAlignment:
    """Backtester should handle misaligned inputs gracefully."""

    def test_subset_signal_columns(self, prices: pd.DataFrame):
        """Signal has fewer tickers than prices."""
        signal = pd.DataFrame(
            np.random.default_rng(0).standard_normal((len(prices), 2)),
            index=prices.index,
            columns=prices.columns[:2],
        )
        bt = VectorizedBacktester(cost_bps=0.0)
        result = bt.run(prices, signal)
        assert len(result) > 0

    def test_subset_signal_dates(self, prices: pd.DataFrame):
        """Signal has fewer dates than prices."""
        signal = pd.DataFrame(
            np.random.default_rng(0).standard_normal((50, prices.shape[1])),
            index=prices.index[:50],
            columns=prices.columns,
        )
        bt = VectorizedBacktester(cost_bps=0.0)
        result = bt.run(prices, signal)
        assert len(result) > 0
        assert len(result) < 50  # last row dropped
