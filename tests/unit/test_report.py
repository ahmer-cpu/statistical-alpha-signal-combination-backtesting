"""Tests for SignalReport orchestrator."""

import numpy as np
import pandas as pd
import pytest

from qre.analytics.validation.report import (
    SignalReport,
    SignalReportConfig,
    _natural_freq,
)


def _make_data(n_days: int = 120, n_tickers: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, (n_days, n_tickers)), axis=0)),
        index=dates,
        columns=tickers,
    )
    signal = pd.DataFrame(
        rng.standard_normal((n_days, n_tickers)),
        index=dates,
        columns=tickers,
    )
    return signal, prices


class TestSignalReportRun:
    """Tests that run() completes and populates results."""

    def test_run_completes(self):
        signal, prices = _make_data()
        cfg = SignalReportConfig(
            horizons=[1, 5],
            ic_decay_max_lag=5,
            n_bootstrap=100,
            seed=42,
        )
        report = SignalReport(signal, prices, name="Test", config=cfg).run()
        assert report._results is not None

    def test_all_nan_signal_does_not_crash(self):
        """A degenerate (all-NaN) signal gives an all-NaN decay; natural_freq
        must fall back to 1 instead of crashing on int(NaN)."""
        _, prices = _make_data()
        dead = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
        cfg = SignalReportConfig(
            horizons=[1, 5], ic_decay_max_lag=5, n_bootstrap=0,
        )
        report = SignalReport(dead, prices, name="dead", config=cfg).run()
        assert report.results["natural_freq"] == 1

    def test_results_has_expected_keys(self):
        signal, prices = _make_data()
        cfg = SignalReportConfig(
            horizons=[1, 5],
            ic_decay_max_lag=5,
            n_bootstrap=100,
            seed=42,
        )
        report = SignalReport(signal, prices, config=cfg).run()
        expected = {
            "autocorrelation", "turnover", "dispersion", "lead_lag",
            "ic_by_horizon", "ic_decay", "ic_decay_ci",
            "natural_freq", "backtest_daily", "backtest_natural",
            "sharpe_daily", "sharpe_natural",
        }
        assert expected == set(report.results.keys())

    def test_ic_by_horizon_has_correct_rows(self):
        signal, prices = _make_data()
        cfg = SignalReportConfig(
            horizons=[1, 5, 10], n_bootstrap=100, ic_decay_max_lag=5
        )
        report = SignalReport(signal, prices, config=cfg).run()
        assert len(report.results["ic_by_horizon"]) == 3

    def test_natural_freq_is_positive_int(self):
        signal, prices = _make_data()
        cfg = SignalReportConfig(horizons=[1, 5], n_bootstrap=100, ic_decay_max_lag=5)
        report = SignalReport(signal, prices, config=cfg).run()
        freq = report.results["natural_freq"]
        assert isinstance(freq, int)
        assert freq >= 1

    def test_results_raises_before_run(self):
        signal, prices = _make_data()
        report = SignalReport(signal, prices)
        with pytest.raises(RuntimeError, match="Call .run"):
            _ = report.results

    def test_run_returns_self(self):
        signal, prices = _make_data()
        cfg = SignalReportConfig(horizons=[1], n_bootstrap=100, ic_decay_max_lag=3)
        report = SignalReport(signal, prices, config=cfg)
        returned = report.run()
        assert returned is report


class TestNaturalFreq:
    """Tests for the reliability-weighted natural-frequency picker."""

    def test_prefers_reliable_lag_over_raw_ic_peak(self):
        """Raw IC rises monotonically to lag 63, but its se explodes there;
        lag 21 has the best |t|, so it — not the raw-IC boundary peak — wins."""
        lags = [1, 5, 10, 21, 42, 63]
        decay = pd.Series([0.01, 0.02, 0.03, 0.05, 0.06, 0.07], index=lags)
        se = pd.Series([0.02, 0.02, 0.02, 0.02, 0.05, 0.09], index=lags)
        decay_ci = pd.DataFrame({
            "point_estimate": decay, "se_bootstrap": se,
            "ci_lower": decay - 2 * se, "ci_upper": decay + 2 * se,
        })
        # |t|: 0.5, 1.0, 1.5, 2.5, 1.2, 0.78 -> peak at 21; raw-IC peak is 63.
        assert _natural_freq(decay, decay_ci) == 21

    def test_falls_back_to_raw_ic_when_no_bootstrap(self):
        """se all NaN (n_bootstrap=0): fall back to |IC| argmax."""
        lags = [1, 5, 10]
        decay = pd.Series([0.01, 0.03, 0.02], index=lags)
        decay_ci = pd.DataFrame(
            {"point_estimate": [np.nan] * 3, "se_bootstrap": [np.nan] * 3},
            index=lags,
        )
        assert _natural_freq(decay, decay_ci) == 5

    def test_negative_signal_picks_strongest_magnitude(self):
        """A robustly-negative IC signal: argmax of |t|, not signed idxmax."""
        lags = [1, 5, 10]
        decay = pd.Series([-0.01, -0.05, -0.02], index=lags)
        se = pd.Series([0.02, 0.02, 0.02], index=lags)
        decay_ci = pd.DataFrame({"point_estimate": decay, "se_bootstrap": se})
        assert _natural_freq(decay, decay_ci) == 5

    def test_all_nan_returns_one(self):
        lags = [1, 5]
        decay = pd.Series([np.nan, np.nan], index=lags)
        decay_ci = pd.DataFrame(
            {"point_estimate": [np.nan] * 2, "se_bootstrap": [np.nan] * 2},
            index=lags,
        )
        assert _natural_freq(decay, decay_ci) == 1


class TestSignalReportOutput:
    """Tests for print_summary and plot."""

    def test_print_summary_runs(self, capsys):
        signal, prices = _make_data()
        cfg = SignalReportConfig(horizons=[1, 5], n_bootstrap=100, ic_decay_max_lag=5)
        report = SignalReport(signal, prices, name="TestSig", config=cfg).run()
        report.print_summary()
        captured = capsys.readouterr()
        assert "TestSig" in captured.out
        assert "Sharpe" in captured.out

    def test_plot_returns_figure(self):
        signal, prices = _make_data()
        cfg = SignalReportConfig(horizons=[1, 5], n_bootstrap=100, ic_decay_max_lag=5)
        report = SignalReport(signal, prices, config=cfg).run()
        import matplotlib.pyplot as plt
        fig = report.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
