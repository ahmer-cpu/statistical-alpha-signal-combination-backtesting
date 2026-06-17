"""Tests for the 9 new alpha factors added in Phase 3."""

import numpy as np
import pandas as pd
import pytest

from qre.alpha.factors.low_vol import LowVol
from qre.alpha.factors.quality import QualityProxy
from qre.alpha.factors.residual_momentum import ResidualMomentum
from qre.alpha.factors.rolling_cvar import RollingCVaR
from qre.alpha.factors.rolling_max_drawdown import RollingMaxDrawdown
from qre.alpha.factors.rolling_sharpe import RollingSharpe
from qre.alpha.factors.rolling_skewness import RollingSkewness
from qre.alpha.factors.sector_neutral_momentum import SectorNeutralMomentum
from qre.alpha.factors.short_term_reversal import ShortTermReversal


def _make_prices(n_days: int = 300, n_tickers: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]
    returns = rng.normal(0.0005, 0.02, (n_days, n_tickers))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def _make_prices_with_spy(
    n_days: int = 300, n_tickers: int = 5, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)] + ["SPY"]
    returns = rng.normal(0.0005, 0.02, (n_days, n_tickers + 1))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture()
def prices() -> pd.DataFrame:
    return _make_prices()


@pytest.fixture()
def prices_spy() -> pd.DataFrame:
    return _make_prices_with_spy()


# Use small windows so tests are fast and fit within 300 rows
SIMPLE_FACTORS = [
    ShortTermReversal(window=5),
    LowVol(window=21),
    QualityProxy(window=60),
    RollingSharpe(window=60),
    RollingCVaR(window=60),
    RollingMaxDrawdown(window=60),
    RollingSkewness(window=60),
]

SIMPLE_IDS = [
    "short_term_reversal",
    "low_vol",
    "quality_proxy",
    "rolling_sharpe",
    "rolling_cvar",
    "rolling_max_drawdown",
    "rolling_skewness",
]


class TestOutputShape:
    """Every factor must return a DataFrame with the same shape as the input."""

    @pytest.mark.parametrize("factor", SIMPLE_FACTORS, ids=SIMPLE_IDS)
    def test_output_shape_matches_input(self, prices: pd.DataFrame, factor):
        signal = factor.compute(prices)
        assert signal.shape == prices.shape
        assert list(signal.columns) == list(prices.columns)
        assert list(signal.index) == list(prices.index)

    def test_residual_momentum_shape(self, prices_spy: pd.DataFrame):
        f = ResidualMomentum(beta_window=30, momentum_window=60, skip=5)
        signal = f.compute(prices_spy)
        # Market proxy (SPY) is consumed as benchmark and dropped from output.
        expected_cols = [c for c in prices_spy.columns if c != "SPY"]
        assert signal.shape == (prices_spy.shape[0], len(expected_cols))
        assert list(signal.columns) == expected_cols

    def test_sector_neutral_momentum_shape(self, prices: pd.DataFrame):
        sector_map = {c: "Sector_A" for c in prices.columns}
        f = SectorNeutralMomentum(sector_map=sector_map, window=60, skip=5)
        signal = f.compute(prices)
        assert signal.shape == prices.shape
        assert list(signal.columns) == list(prices.columns)


class TestNaNWarmup:
    """The first `lookback` rows should be NaN (insufficient history)."""

    def test_short_term_reversal_warmup(self, prices: pd.DataFrame):
        f = ShortTermReversal(window=5)
        signal = f.compute(prices)
        assert signal.iloc[:5].isna().all().all()
        assert signal.iloc[5:].notna().any().any()

    def test_low_vol_warmup(self, prices: pd.DataFrame):
        f = LowVol(window=21)
        signal = f.compute(prices)
        # 1 NaN from pct_change + 20 NaN from rolling(21).std() min_periods
        assert signal.iloc[:21].isna().all().all()
        assert signal.iloc[22:].notna().any().any()

    def test_quality_proxy_warmup(self, prices: pd.DataFrame):
        f = QualityProxy(window=60)
        signal = f.compute(prices)
        assert signal.iloc[:60].isna().all().all()
        assert signal.iloc[61:].notna().any().any()

    def test_rolling_sharpe_warmup(self, prices: pd.DataFrame):
        f = RollingSharpe(window=60)
        signal = f.compute(prices)
        assert signal.iloc[:60].isna().all().all()
        assert signal.iloc[61:].notna().any().any()

    def test_rolling_cvar_warmup(self, prices: pd.DataFrame):
        f = RollingCVaR(window=60)
        signal = f.compute(prices)
        assert signal.iloc[:60].isna().all().all()
        assert signal.iloc[61:].notna().any().any()

    def test_rolling_max_drawdown_warmup(self, prices: pd.DataFrame):
        f = RollingMaxDrawdown(window=60)
        signal = f.compute(prices)
        assert signal.iloc[:60].isna().all().all()
        assert signal.iloc[61:].notna().any().any()

    def test_rolling_skewness_warmup(self, prices: pd.DataFrame):
        f = RollingSkewness(window=60)
        signal = f.compute(prices)
        assert signal.iloc[:60].isna().all().all()
        assert signal.iloc[61:].notna().any().any()

    def test_residual_momentum_warmup(self, prices_spy: pd.DataFrame):
        f = ResidualMomentum(beta_window=30, momentum_window=60, skip=5)
        signal = f.compute(prices_spy)
        # warmup: 1 (pct_change) + 29 (rolling cov) + 5 (shift) + 54 (rolling sum) = 89
        assert signal.iloc[:89].isna().all().all()
        assert signal.iloc[90:].notna().any().any()

    def test_sector_neutral_momentum_warmup(self, prices: pd.DataFrame):
        sector_map = {c: "Sector_A" for c in prices.columns}
        f = SectorNeutralMomentum(sector_map=sector_map, window=60, skip=5)
        signal = f.compute(prices)
        assert signal.iloc[:60].isna().all().all()
        assert signal.iloc[60:].notna().any().any()


class TestNoLookahead:
    """Signal at date t must not change when future data changes."""

    @pytest.mark.parametrize("factor", SIMPLE_FACTORS, ids=SIMPLE_IDS)
    def test_future_data_does_not_affect_past_signal(
        self, prices: pd.DataFrame, factor
    ):
        signal_full = factor.compute(prices)
        truncated = prices.iloc[:-30].copy()
        signal_trunc = factor.compute(truncated)
        overlap = signal_full.loc[signal_trunc.index]
        pd.testing.assert_frame_equal(overlap, signal_trunc)

    def test_residual_momentum_no_lookahead(self, prices_spy: pd.DataFrame):
        f = ResidualMomentum(beta_window=30, momentum_window=60, skip=5)
        signal_full = f.compute(prices_spy)
        truncated = prices_spy.iloc[:-30].copy()
        signal_trunc = f.compute(truncated)
        overlap = signal_full.loc[signal_trunc.index]
        pd.testing.assert_frame_equal(overlap, signal_trunc)

    def test_sector_neutral_momentum_no_lookahead(self, prices: pd.DataFrame):
        sector_map = {c: "Sector_A" for c in prices.columns}
        f = SectorNeutralMomentum(sector_map=sector_map, window=60, skip=5)
        signal_full = f.compute(prices)
        truncated = prices.iloc[:-30].copy()
        signal_trunc = f.compute(truncated)
        overlap = signal_full.loc[signal_trunc.index]
        pd.testing.assert_frame_equal(overlap, signal_trunc)


class TestSignConvention:
    """Positive signal = long, negative = short."""

    def test_reversal_falling_prices_positive(self):
        """Recent losers should get a positive (buy) signal."""
        dates = pd.bdate_range("2023-01-01", periods=50)
        prices = pd.DataFrame({"A": np.linspace(200, 100, 50)}, index=dates)
        signal = ShortTermReversal(window=5).compute(prices)
        valid = signal.dropna()
        assert (valid > 0).all().all()

    def test_low_vol_stable_stock_positive(self):
        """A stock with lower volatility should score higher than a volatile one."""
        dates = pd.bdate_range("2023-01-01", periods=100)
        rng = np.random.default_rng(42)
        stable = 100 + np.cumsum(rng.normal(0, 0.001, 100))
        volatile = 100 + np.cumsum(rng.normal(0, 0.05, 100))
        prices = pd.DataFrame({"stable": stable, "volatile": volatile}, index=dates)
        signal = LowVol(window=21).compute(prices)
        valid = signal.dropna()
        # Stable stock should have higher (less negative) signal
        assert (valid["stable"] > valid["volatile"]).all()

    def test_rolling_cvar_mild_tail_higher(self):
        """Milder left tail should score higher than a severe one (low risk = long)."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2023-01-01", periods=200)
        mild = rng.normal(0.0005, 0.005, 200)
        severe = rng.normal(0.0005, 0.005, 200).copy()
        severe[::20] = -0.10  # periodic crashes -> severe left tail
        prices = pd.DataFrame(
            {
                "mild": 100 * np.exp(np.cumsum(mild)),
                "severe": 100 * np.exp(np.cumsum(severe)),
            },
            index=dates,
        )
        signal = RollingCVaR(window=60).compute(prices)
        valid = signal.dropna()
        assert (valid["mild"] > valid["severe"]).all()

    def test_rolling_max_drawdown_shallow_higher(self):
        """Shallower drawdown should score higher than a deep one (low risk = long)."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2023-01-01", periods=200)
        shallow = rng.normal(0.0005, 0.005, 200)
        deep = rng.normal(0.0005, 0.005, 200).copy()
        deep[::20] = -0.12  # periodic crashes -> recurring deep drawdowns
        prices = pd.DataFrame(
            {
                "shallow": 100 * np.exp(np.cumsum(shallow)),
                "deep": 100 * np.exp(np.cumsum(deep)),
            },
            index=dates,
        )
        signal = RollingMaxDrawdown(window=60).compute(prices)
        valid = signal.dropna()
        assert (valid["shallow"] > valid["deep"]).all()

    def test_rolling_sharpe_rising_positive(self):
        """Steadily rising prices should produce positive rolling Sharpe."""
        dates = pd.bdate_range("2023-01-01", periods=200)
        prices = pd.DataFrame(
            {"A": np.linspace(100, 200, 200)}, index=dates
        )
        signal = RollingSharpe(window=60).compute(prices)
        valid = signal.dropna()
        assert (valid > 0).all().all()

    def test_rolling_skewness_negative_skew_positive(self):
        """Negatively skewed returns should produce positive (buy) signal."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2023-01-01", periods=200)
        # Simulate negative skewness: mostly small gains, occasional large losses
        returns = np.abs(rng.normal(0.001, 0.005, 200))
        returns[::20] = -0.05  # periodic large losses
        prices = pd.DataFrame(
            {"A": 100 * np.exp(np.cumsum(returns))}, index=dates
        )
        signal = RollingSkewness(window=60).compute(prices)
        valid = signal.iloc[-30:]
        # Negative skew negated → should be positive
        assert (valid > 0).all().all()


class TestResidualMomentumSpecific:
    """ResidualMomentum-specific edge cases."""

    def test_market_column_dropped(self, prices_spy: pd.DataFrame):
        """The market proxy (SPY) is consumed as benchmark and dropped."""
        f = ResidualMomentum(beta_window=30, momentum_window=60, skip=5)
        signal = f.compute(prices_spy)
        assert "SPY" not in signal.columns
        assert signal.shape[1] == prices_spy.shape[1] - 1

    def test_raises_without_market(self, prices: pd.DataFrame):
        """Should raise ValueError when the market proxy column is missing."""
        f = ResidualMomentum(beta_window=30, momentum_window=60, skip=5)
        with pytest.raises(ValueError):
            f.compute(prices)


class TestSectorNeutralSpecific:
    """SectorNeutralMomentum-specific edge cases."""

    def test_unmapped_tickers_are_nan(self):
        """Tickers not in sector_map should have NaN signal."""
        prices = _make_prices(n_days=200, n_tickers=5)
        sector_map = {"T0": "A", "T1": "A", "T2": "B"}  # T3, T4 unmapped
        f = SectorNeutralMomentum(sector_map=sector_map, window=60, skip=5)
        signal = f.compute(prices)
        assert signal["T3"].isna().all()
        assert signal["T4"].isna().all()
        # Mapped tickers should have some valid values
        assert signal["T0"].notna().any()

    def test_sector_demeaning_sums_to_zero(self):
        """Within each sector, the demeaned signals should sum to ~0."""
        prices = _make_prices(n_days=200, n_tickers=4)
        sector_map = {"T0": "A", "T1": "A", "T2": "B", "T3": "B"}
        f = SectorNeutralMomentum(sector_map=sector_map, window=60, skip=5)
        signal = f.compute(prices)
        valid = signal.dropna()
        sector_a_sum = valid[["T0", "T1"]].sum(axis=1)
        sector_b_sum = valid[["T2", "T3"]].sum(axis=1)
        np.testing.assert_allclose(sector_a_sum, 0, atol=1e-12)
        np.testing.assert_allclose(sector_b_sum, 0, atol=1e-12)

    def test_single_name_sector_is_nan(self):
        """A lone stock in its sector has no peers, so its signal is NaN, not 0."""
        prices = _make_prices(n_days=200, n_tickers=4)
        sector_map = {"T0": "A", "T1": "A", "T2": "A", "T3": "Solo"}
        f = SectorNeutralMomentum(sector_map=sector_map, window=60, skip=5)
        signal = f.compute(prices)
        assert signal["T3"].isna().all()   # lone member of "Solo"
        assert signal["T0"].notna().any()  # sector A has peers
