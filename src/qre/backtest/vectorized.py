import numpy as np
import pandas as pd


class VectorizedBacktester:
    """Fast signal-level backtester for alpha research screening.

    Converts a raw signal into dollar-neutral, unit-gross-exposure positions,
    then computes: positions × forward_returns - transaction_costs.
    """

    def __init__(
        self,
        cost_bps: float = 10.0,
        rebalance_freq: int = 1,
        position_smooth: float = 0.0,
    ) -> None:
        self.cost_bps = cost_bps
        self.rebalance_freq = rebalance_freq
        self.position_smooth = position_smooth

    def run(self, prices: pd.DataFrame, signal: pd.DataFrame) -> pd.Series:
        # Align inputs so indices and columns match exactly
        prices, signal = prices.align(signal, join="inner", axis=0)
        signal = signal.reindex(columns=prices.columns)

        fwd_returns = prices.pct_change().shift(-1)

        # Dollar-neutral: subtract cross-sectional mean
        raw = signal.sub(signal.mean(axis=1), axis=0)

        # Unit gross exposure: divide by sum of absolute weights each day
        gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
        positions = raw.div(gross, axis=0).fillna(0.0)

        # Rebalance mask: carry forward positions on non-rebalance days
        if self.rebalance_freq > 1:
            row_mask = np.arange(len(positions)) % self.rebalance_freq == 0
            rebal_mask = np.broadcast_to(row_mask[:, None], positions.shape)
            positions = positions.where(rebal_mask).ffill().fillna(0.0)

        # Exponential smoothing: blend toward yesterday's position
        if self.position_smooth > 0:
            alpha = 1.0 - self.position_smooth
            for i in range(1, len(positions)):
                positions.iloc[i] = (
                    alpha * positions.iloc[i]
                    + self.position_smooth * positions.iloc[i - 1]
                )
            # Re-normalize to unit gross exposure after smoothing
            gross_smooth = positions.abs().sum(axis=1).replace(0.0, np.nan)
            positions = positions.div(gross_smooth, axis=0).fillna(0.0)

        # Turnover: charge initial portfolio construction, not just changes
        turnover = positions.diff().abs().sum(axis=1)
        if len(turnover) > 0:
            turnover.iloc[0] = positions.iloc[0].abs().sum()

        # Strategy returns. min_count=1 omits NaN-return names from the day's
        # sum, which equals a zero-PnL contribution for those names (it does
        # NOT rescale the others) — the right neutral treatment for a mid-sample
        # halt/gap. The all-NaN final row (forward returns shifted off the end)
        # stays NaN and is dropped below. Masking positions by forward-return
        # availability instead would leak future info (dodging names about to
        # delist), so we deliberately do not.
        gross_returns = (positions * fwd_returns).sum(axis=1, min_count=1)
        costs = turnover * self.cost_bps / 10_000

        return (gross_returns - costs).dropna()