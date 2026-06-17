"""Tests for alpha factor implementations."""

import numpy as np
import pandas as pd
import pytest

from qre.alpha.factors.bollinger import BollingerZScore
from qre.alpha.factors.cross_sectional_momentum import CrossSectionalMomentum
from qre.alpha.factors.momentum import Momentum
from qre.alpha.factors.rsi import RSI


def _make_prices(n_days: int = 300, n_tickers: int = 5, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic price panel with realistic structure."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]
    returns = rng.normal(0.0005, 0.02, (n_days, n_tickers))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture()
def prices() -> pd.DataFrame:
    return _make_prices()


class TestOutputShape:
    """Every factor must return a DataFrame with the same shape as the input."""

    @pytest.mark.parametrize(
        "factor",
        [
            Momentum(window=60, skip=5),
            CrossSectionalMomentum(window=60, skip=5),
            RSI(window=14),
            BollingerZScore(window=20),
        ],
        ids=["momentum", "cs_momentum", "rsi", "bollinger"],
    )
    def test_output_shape_matches_input(self, prices: pd.DataFrame, factor):
        signal = factor.compute(prices)
        assert signal.shape == prices.shape
        assert list(signal.columns) == list(prices.columns)
        assert list(signal.index) == list(prices.index)


class TestNaNWarmup:
    """The first `lookback` rows should be NaN (insufficient history)."""

    def test_momentum_warmup(self, prices: pd.DataFrame):
        m = Momentum(window=60, skip=5)
        signal = m.compute(prices)
        # First `window` rows should be all NaN
        assert signal.iloc[:60].isna().all().all()
        # After warmup, should have valid values
        assert signal.iloc[60:].notna().any().any()

    def test_rsi_warmup(self, prices: pd.DataFrame):
        r = RSI(window=14)
        signal = r.compute(prices)
        # diff() produces 1 NaN, rolling(14) needs 14 → first 14 NaN
        assert signal.iloc[:14].isna().all().all()
        assert signal.iloc[14:].notna().any().any()

    def test_bollinger_warmup(self, prices: pd.DataFrame):
        b = BollingerZScore(window=20)
        signal = b.compute(prices)
        assert signal.iloc[:19].isna().all().all()
        assert signal.iloc[20:].notna().any().any()

    def test_cs_momentum_warmup(self, prices: pd.DataFrame):
        cs = CrossSectionalMomentum(window=60, skip=5)
        signal = cs.compute(prices)
        assert signal.iloc[:60].isna().all().all()
        assert signal.iloc[60:].notna().any().any()


class TestNoLookahead:
    """Signal at date t must not change when future data changes."""

    @pytest.mark.parametrize(
        "factor",
        [
            Momentum(window=60, skip=5),
            CrossSectionalMomentum(window=60, skip=5),
            RSI(window=14),
            BollingerZScore(window=20),
        ],
        ids=["momentum", "cs_momentum", "rsi", "bollinger"],
    )
    def test_future_data_does_not_affect_past_signal(
        self, prices: pd.DataFrame, factor
    ):
        signal_full = factor.compute(prices)

        # Truncate last 30 days and recompute
        truncated = prices.iloc[:-30].copy()
        signal_trunc = factor.compute(truncated)

        # Signals for the overlapping period must be identical
        overlap = signal_full.loc[signal_trunc.index]
        pd.testing.assert_frame_equal(overlap, signal_trunc)


class TestSignConvention:
    """Positive signal = long, negative = short."""

    def test_momentum_rising_prices_positive(self):
        """Steadily rising prices should produce positive momentum signal."""
        dates = pd.bdate_range("2023-01-01", periods=300)
        prices = pd.DataFrame(
            {"A": np.linspace(100, 200, 300)},
            index=dates,
        )
        m = Momentum(window=60, skip=5)
        signal = m.compute(prices)
        valid = signal.dropna()
        assert (valid > 0).all().all()

    def test_rsi_oversold_is_positive(self):
        """After a sustained decline, RSI signal should be positive (buy)."""
        dates = pd.bdate_range("2023-01-01", periods=100)
        prices = pd.DataFrame(
            {"A": np.linspace(200, 100, 100)},
            index=dates,
        )
        r = RSI(window=14)
        signal = r.compute(prices)
        valid = signal.dropna()
        # Sustained decline → oversold → RSI < 50 → (50 - RSI) > 0
        assert (valid.iloc[-10:] > 0).all().all()

    def test_bollinger_below_band_is_positive(self):
        """Price below the lower band should produce positive (buy) signal."""
        dates = pd.bdate_range("2023-01-01", periods=100)
        # Flat then sudden drop
        flat = np.full(80, 100.0)
        drop = np.linspace(100, 80, 20)
        prices = pd.DataFrame(
            {"A": np.concatenate([flat, drop])},
            index=dates,
        )
        b = BollingerZScore(window=20)
        signal = b.compute(prices)
        # After the drop, price is below the band → positive signal
        assert (signal.iloc[-5:] > 0).all().all()
