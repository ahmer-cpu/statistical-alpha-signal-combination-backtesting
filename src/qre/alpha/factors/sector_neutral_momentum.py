import pandas as pd

from qre.alpha.base import Alpha


class SectorNeutralMomentum(Alpha):
    """Sector-neutral momentum factor (Moskowitz & Grinblatt, 1999).

    Removes sector rotation bias from raw momentum by subtracting
    the sector average. The signal measures how much a stock's
    momentum exceeds (or trails) its sector peers, preventing the
    factor from degenerating into a sector bet.

    Formula:
        1. raw = P_{t-skip} / P_{t-window} - 1  (standard momentum)
        2. sector_mean = mean(raw) within each GICS sector
        3. signal = raw - sector_mean

    Signal: positive = outperforming sector peers (buy),
            negative = underperforming (sell).
    Timing: uses prices up to close of date t; no lookahead.
    NaN warmup: first `window` rows are NaN.
    Note: tickers not in sector_map are masked to NaN. Single-name sectors are
        also NaN — with no peers there is no within-sector signal to measure.
    """

    def __init__(
        self,
        sector_map: dict[str, str],
        window: int = 252,
        skip: int = 21,
    ) -> None:
        super().__init__(
            name=f"sector_neutral_momentum_{window}_{skip}",
            lookback=window,
        )
        self.sector_map = sector_map
        self.window = window
        self.skip = skip

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        raw_momentum = prices.shift(self.skip) / prices.shift(self.window) - 1

        # Build sector series aligned to columns (skip tickers without mapping)
        sectors = pd.Series({
            t: self.sector_map[t]
            for t in raw_momentum.columns
            if t in self.sector_map
        })

        # Demean: subtract sector average from each stock's momentum
        mapped = raw_momentum[sectors.index]
        sector_mean = mapped.T.groupby(sectors).transform("mean").T
        signal = mapped - sector_mean

        # A single-name sector has no peers to neutralise against, so its
        # demeaned momentum is identically zero (no information). Mask those
        # columns to NaN rather than emit a spurious neutral-zero signal.
        sector_sizes = sectors.value_counts()
        singletons = [t for t in sectors.index if sector_sizes[sectors[t]] == 1]
        if singletons:
            signal.loc[:, singletons] = float("nan")

        # Reindex to full column set; unmapped tickers become NaN
        return signal.reindex(columns=raw_momentum.columns)
