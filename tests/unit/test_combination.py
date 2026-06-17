"""Tests for alpha signal combination methods."""

import numpy as np
import pandas as pd
import pytest

from qre.alpha.combination import (
    EqualWeightCombiner,
    ICWeightedCombiner,
    LassoCombiner,
    _cross_sectional_zscore,
)
from qre.analytics.ic import compute_ic


def _make_prices(n_days: int = 400, n_tickers: int = 6, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]
    returns = rng.normal(0.0005, 0.02, (n_days, n_tickers))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def _make_signals(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "momentum": prices.pct_change(60),
        "reversal": -prices.pct_change(5),
        "low_vol": -prices.pct_change().rolling(21).std(),
    }


@pytest.fixture()
def prices() -> pd.DataFrame:
    return _make_prices()


@pytest.fixture()
def signals(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return _make_signals(prices)


class TestCrossSectionalZscore:
    """The z-score helper must produce zero-mean, unit-std rows."""

    def test_row_mean_is_zero(self):
        df = pd.DataFrame(
            {"A": [1.0, 2.0], "B": [3.0, 6.0], "C": [5.0, 10.0]}
        )
        z = _cross_sectional_zscore(df)
        np.testing.assert_allclose(z.mean(axis=1), 0, atol=1e-12)

    def test_row_std_is_one(self):
        df = pd.DataFrame(
            {"A": [1.0, 2.0], "B": [3.0, 6.0], "C": [5.0, 10.0]}
        )
        z = _cross_sectional_zscore(df)
        # pandas std uses ddof=1
        np.testing.assert_allclose(z.std(axis=1, ddof=1), 1, atol=1e-12)

    def test_preserves_shape_and_labels(self, signals: dict[str, pd.DataFrame]):
        df = signals["momentum"]
        z = _cross_sectional_zscore(df)
        assert z.shape == df.shape
        assert list(z.columns) == list(df.columns)
        assert list(z.index) == list(df.index)


class TestEqualWeightCombiner:
    def test_output_shape(self, signals: dict[str, pd.DataFrame]):
        out = EqualWeightCombiner().combine(signals)
        ref = signals["momentum"]
        assert out.shape == ref.shape
        assert list(out.columns) == list(ref.columns)
        assert list(out.index) == list(ref.index)

    def test_single_factor_equals_its_zscore(self, signals: dict[str, pd.DataFrame]):
        """Combining one factor should just return its z-score."""
        one = {"momentum": signals["momentum"]}
        out = EqualWeightCombiner().combine(one)
        expected = _cross_sectional_zscore(signals["momentum"])
        pd.testing.assert_frame_equal(out, expected)

    def test_is_mean_of_zscores(self, signals: dict[str, pd.DataFrame]):
        """Output equals the nanmean of the z-scored inputs."""
        import warnings

        out = EqualWeightCombiner().combine(signals)
        zs = [_cross_sectional_zscore(s) for s in signals.values()]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            expected = np.nanmean(
                np.stack([z.values for z in zs], axis=0), axis=0
            )
        np.testing.assert_allclose(out.values, expected, equal_nan=True)

    def test_no_runtime_warning_on_warmup(self, signals: dict[str, pd.DataFrame]):
        """All-NaN warmup rows must not emit a RuntimeWarning."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", category=RuntimeWarning)
            EqualWeightCombiner().combine(signals)


class TestICWeightedCombiner:
    def test_output_shape(
        self, signals: dict[str, pd.DataFrame], prices: pd.DataFrame
    ):
        out = ICWeightedCombiner(lookback=120).combine(signals, prices)
        ref = signals["momentum"]
        assert out.shape == ref.shape
        assert list(out.columns) == list(ref.columns)

    def test_warmup_rows_are_nan(
        self, signals: dict[str, pd.DataFrame], prices: pd.DataFrame
    ):
        """All-NaN warmup rows come out NaN, not a spurious 0.0.

        nansum treats NaN as 0, so before the mask an all-NaN warmup row
        (every factor's signal *or* trailing-IC weight still missing)
        collapsed to a fake 0.0 that read as 'present'. Row 0 has no valid
        signal and no estimable weight, so the combined output must be NaN.
        """
        out = ICWeightedCombiner(lookback=120, shrinkage=0.0).combine(
            signals, prices
        )
        assert out.iloc[0].isna().all()
        assert out.notna().any().any()

    def test_full_shrinkage_matches_equal_weight(
        self, signals: dict[str, pd.DataFrame], prices: pd.DataFrame
    ):
        """shrinkage=1 → pure equal-weight once all IC-weights are warm.

        A factor's IC-weight needs its own signal warmup PLUS the rolling
        lookback before it is estimable. Until every factor's weight exists,
        ICWeighted re-normalizes over the available factors while
        EqualWeight (nanmean) includes any factor whose z-score is valid —
        so the two only coincide after the longest combined warmup.
        """
        icw = ICWeightedCombiner(lookback=120, shrinkage=1.0).combine(
            signals, prices
        )
        ew = EqualWeightCombiner().combine(signals)
        # momentum: 60d signal warmup + 120d rolling + 1 shift ≈ 181; use 200.
        np.testing.assert_allclose(
            icw.iloc[200:].values,
            ew.iloc[200:].values,
            atol=1e-10,
            equal_nan=True,
        )

    def test_ewm_option_runs_and_differs(
        self, signals: dict[str, pd.DataFrame], prices: pd.DataFrame
    ):
        """EWM weighting should produce a different result than flat rolling."""
        flat = ICWeightedCombiner(
            lookback=120, shrinkage=0.0, use_ewm=False
        ).combine(signals, prices)
        ewm = ICWeightedCombiner(
            halflife=60, shrinkage=0.0, use_ewm=True
        ).combine(signals, prices)
        # Compare on a region where both have values
        idx = flat.dropna(how="all").index[200:]
        diff = (flat.loc[idx] - ewm.loc[idx]).abs().to_numpy()
        assert np.nanmax(diff) > 1e-6

    def test_no_lookahead(
        self, signals: dict[str, pd.DataFrame], prices: pd.DataFrame
    ):
        """Truncating future data must not change past combined signals."""
        combiner = ICWeightedCombiner(lookback=120)
        full = combiner.combine(signals, prices)

        trunc_prices = prices.iloc[:-30]
        trunc_signals = {n: s.iloc[:-30] for n, s in signals.items()}
        trunc = combiner.combine(trunc_signals, trunc_prices)

        # Overlap (minus the last row of trunc, whose forward return is NaN)
        overlap = trunc.index[:-1]
        np.testing.assert_allclose(
            full.loc[overlap].values,
            trunc.loc[overlap].values,
            atol=1e-10,
            equal_nan=True,
        )

    def test_negative_ic_factor_is_flipped(self, prices: pd.DataFrame):
        """A factor with negative trailing IC is shorted, not applied long.

        Two factors carry identical information with opposite sign: `good`
        equals the forward return (IC = +1), `bad` is its negation (IC = -1).
        A sign-aware combiner flips `bad` so the two reinforce instead of
        cancelling, leaving a combined signal that predicts strongly positively.
        Before the fix (absolute-value weights) they cancelled to ~zero.
        """
        fwd = prices.pct_change().shift(-1)
        signals = {"good": fwd, "bad": -fwd}
        out = ICWeightedCombiner(lookback=20, shrinkage=0.0).combine(signals, prices)

        ic = compute_ic(out, fwd).dropna()
        assert ic.mean() > 0.5


class TestLassoCombiner:
    def test_output_shape(
        self, signals: dict[str, pd.DataFrame], prices: pd.DataFrame
    ):
        out = LassoCombiner(lookback=120, alpha=0.01).combine(signals, prices)
        ref = signals["momentum"]
        assert out.shape == ref.shape
        assert list(out.columns) == list(ref.columns)

    def test_warmup_rows_are_nan(
        self, signals: dict[str, pd.DataFrame], prices: pd.DataFrame
    ):
        """First `lookback` rows have no training window → all NaN."""
        out = LassoCombiner(lookback=120, alpha=0.01).combine(signals, prices)
        assert out.iloc[:120].isna().all().all()
        assert out.iloc[120:].notna().any().any()

    def test_high_alpha_zeros_signal(
        self, signals: dict[str, pd.DataFrame], prices: pd.DataFrame
    ):
        """A very large alpha drives all coefficients to zero → zero signal."""
        out = LassoCombiner(lookback=120, alpha=10.0).combine(signals, prices)
        valid = out.iloc[120:].to_numpy()
        valid = valid[np.isfinite(valid)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-12)

    def test_no_lookahead(
        self, signals: dict[str, pd.DataFrame], prices: pd.DataFrame
    ):
        """Truncating future data must not change past combined signals."""
        combiner = LassoCombiner(lookback=120, alpha=0.01)
        full = combiner.combine(signals, prices)

        trunc_prices = prices.iloc[:-30]
        trunc_signals = {n: s.iloc[:-30] for n, s in signals.items()}
        trunc = combiner.combine(trunc_signals, trunc_prices)

        # Last trunc row has NaN forward returns in training → compare the rest
        overlap = trunc.dropna(how="all").index[:-1]
        np.testing.assert_allclose(
            full.loc[overlap].values,
            trunc.loc[overlap].values,
            atol=1e-8,
            equal_nan=True,
        )
