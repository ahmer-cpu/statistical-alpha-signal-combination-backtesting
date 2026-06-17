"""Alpha signal combination methods.

Three approaches with increasing complexity:
- EqualWeightCombiner: robust baseline, hardest to overfit
- ICWeightedCombiner: adapts to trailing signal quality
- LassoCombiner: penalized regression, handles multicollinearity
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso  # type: ignore[import-untyped]

from qre.analytics.ic import compute_ic


def _cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each row (date) across columns (tickers)."""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


class EqualWeightCombiner:
    """Average of cross-sectionally z-scored factor signals."""

    def combine(self, signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
        normalized = [
            _cross_sectional_zscore(signal) for signal in signals.values()
        ]
        ref = normalized[0]
        stacked = np.stack([s.values for s in normalized], axis=0)
        # Rows where every factor is NaN (warmup) yield an empty-slice mean;
        # the resulting NaN is correct, so silence the expected warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            combined = np.nanmean(stacked, axis=0)
        return pd.DataFrame(combined, index=ref.index, columns=ref.columns)


class ICWeightedCombiner:
    """Weight each factor by its trailing mean IC, with the sign preserved.

    Each factor's weight is its trailing IC normalised to unit gross exposure
    per date (sum of absolute weights = 1). Because the sign is kept, a factor
    whose recent IC is negative is *flipped* (shorted) — combined in its
    profitable direction rather than its as-constructed one. This lets the
    combiner exploit a robustly negative-IC factor (e.g. a volatility factor in
    a regime where high-vol outperforms) without a hand-oriented duplicate.

    Args:
        lookback: Rolling window for trailing mean IC (used when
            use_ewm=False).
        shrinkage: Blend the signed IC weight toward (positively oriented)
            equal-weight 1/k (0 = pure signed IC, 1 = equal weight). The target
            is positive, so heavy shrinkage pulls a negative-IC factor back
            toward a positive weight; use low shrinkage to keep a factor's
            flipped sign.
        use_ewm: Use exponentially weighted mean instead of simple
            rolling mean. Gives more weight to recent IC observations.
        halflife: EWM halflife in days (only used when use_ewm=True).
            After `halflife` days, an observation's weight decays to 50%.
    """

    def __init__(
        self,
        lookback: int = 252,
        shrinkage: float = 0.5,
        use_ewm: bool = False,
        halflife: int = 126,
    ):
        self.lookback = lookback
        self.shrinkage = shrinkage
        self.use_ewm = use_ewm
        self.halflife = halflife

    def combine(
        self, signals: dict[str, pd.DataFrame], prices: pd.DataFrame
    ) -> pd.DataFrame:
        forward_returns = prices.pct_change().shift(-1)
        names = list(signals.keys())
        k = len(names)

        # Step 1: compute trailing IC weight per factor
        daily_ics = {
            name: compute_ic(signals[name], forward_returns)
            for name in names
        }
        if self.use_ewm:
            ic_weight_df = pd.DataFrame({
                name: ic.ewm(halflife=self.halflife).mean().shift(1)
                for name, ic in daily_ics.items()
            })
        else:
            ic_weight_df = pd.DataFrame({
                name: ic.rolling(self.lookback).mean().shift(1)
                for name, ic in daily_ics.items()
            })

        # Step 2: shrink toward equal-weight
        equal_w = 1.0 / k
        ic_weight_df = (
            (1 - self.shrinkage) * ic_weight_df + self.shrinkage * equal_w
        )

        # Step 3: normalize to unit gross exposure per date, PRESERVING sign so
        # a factor with negative trailing IC is flipped (shorted) rather than
        # applied in its losing direction. Dividing the signed weights by the
        # sum of absolute weights keeps signs and makes sum(|w|) = 1.
        gross = ic_weight_df.abs().sum(axis=1).where(lambda s: s > 0)
        ic_weight_df = ic_weight_df.div(gross, axis=0)

        # Step 4: z-score each signal, then weighted-average
        normalized = {
            name: _cross_sectional_zscore(signal)
            for name, signal in signals.items()
        }
        stacked = np.stack(
            [normalized[n].values for n in names], axis=0
        )  # (K, T, N)
        weights = ic_weight_df[names].values.T  # (K, T)
        weights_3d = weights[:, :, np.newaxis]  # (K, T, 1) — broadcast over N

        # nansum treats NaN as 0, so an all-NaN warmup cell (every factor's
        # signal *or* trailing-IC weight still missing) would collapse to a
        # spurious 0.0 instead of NaN. Mask those cells back to NaN so the
        # warmup matches the input signals (and EqualWeightCombiner's nanmean).
        contrib = stacked * weights_3d  # (K, T, N); NaN where signal or weight missing
        combined = np.nansum(contrib, axis=0)  # (T, N)
        combined[np.isnan(contrib).all(axis=0)] = np.nan

        ref = normalized[names[0]]
        return pd.DataFrame(combined, index=ref.index, columns=ref.columns)


class LassoCombiner:
    """Rolling Lasso regression of forward returns on z-scored signals.

    Fits one Lasso per date on the trailing `lookback` window. The fitted
    coefficients become factor weights; the L1 penalty drives redundant
    factors' coefficients to exactly zero, handling multicollinearity that
    independent IC-weighting ignores.

    No lookahead: each training window ends strictly before the date it
    produces a signal for. This is O(T) model fits, each on ~lookback x N
    rows — slow, but acceptable for research.

    Args:
        lookback: Training window length in days.
        alpha: L1 regularization strength (higher = more factors zeroed).
            The target is standardized per window, so alpha is on a unit-
            variance scale. An NB04 alpha sweep showed a flat performance
            plateau across ~[0.0005, 0.005] (IC and daily Sharpe near-constant)
            with a sharp drop by 0.01 and the signal degrading by 0.05. The
            0.001 default sits mid-plateau; for a final freeze, tune with
            walk-forward CV (NB04 Research Idea D).
    """

    def __init__(self, lookback: int = 252, alpha: float = 0.001):
        self.lookback = lookback
        self.alpha = alpha

    def combine(
        self, signals: dict[str, pd.DataFrame], prices: pd.DataFrame
    ) -> pd.DataFrame:
        forward_returns = prices.pct_change().shift(-1)
        names = list(signals.keys())

        normalized = {
            name: _cross_sectional_zscore(signal)
            for name, signal in signals.items()
        }

        ref = normalized[names[0]]
        dates = ref.index
        fwd = forward_returns.reindex(columns=ref.columns)

        # Output: start all-NaN, fill in rows we can compute
        combined = pd.DataFrame(np.nan, index=dates, columns=ref.columns)

        for t in range(self.lookback, len(dates)):
            train_dates = dates[t - self.lookback : t]  # strictly before t
            apply_date = dates[t]

            # X: rows = (date, ticker) pairs, cols = factors
            x_train = np.column_stack([
                normalized[name].loc[train_dates].values.ravel()
                for name in names
            ])
            # y: matching forward returns, same ravel order as X
            y_train = fwd.loc[train_dates].values.ravel()

            # Drop rows with any NaN in features or target
            mask = np.isfinite(x_train).all(axis=1) & np.isfinite(y_train)
            if mask.sum() == 0:
                continue
            x_train, y_train = x_train[mask], y_train[mask]

            # Standardize the target so the L1 penalty sits on a unit-variance
            # scale. Raw daily forward returns have std ~1e-2 while z-scored
            # features have std ~1, so the true coefficients are ~1e-3 and any
            # usable alpha (>=1e-3) would zero them out, killing the signal.
            # This only rescales the fitted coefficients (and thus the combined
            # signal) by a global constant, leaving ranks and the backtester's
            # dollar-neutral positions unchanged.
            y_std = y_train.std()
            if y_std > 0:
                y_train = (y_train - y_train.mean()) / y_std

            model = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=5000)
            model.fit(x_train, y_train)

            # Apply coefficients to today's z-scored signals: (N, K) @ (K,)
            today = np.column_stack(
                [normalized[name].loc[apply_date].values for name in names]
            )
            combined.loc[apply_date] = today @ model.coef_

        return combined
