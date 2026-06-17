import pandas as pd

from qre.alpha.base import Alpha


class ResidualMomentum(Alpha):
    """Residual momentum factor (Blitz, Huij & Martens, 2011).

    Strips out market beta via rolling OLS, then computes momentum on
    the idiosyncratic (residual) returns. This isolates stock-specific
    performance from broad market moves, producing a signal with higher
    IC and lower crash risk than raw momentum.

    Formula:
        1. β_i = rolling_cov(r_i, r_SPY) / rolling_var(r_SPY)
        2. ε_i = r_i − β_i · r_SPY
        3. signal = Σ ε_{t-momentum_window..t-skip}  (sum of residuals)

    Signal: positive = idiosyncratic winner (buy), negative = loser (sell).
    Timing: uses prices up to close of date t; no lookahead.
    NaN warmup: first `beta_window + momentum_window + 1` rows are NaN.

    The market proxy is an explicit dependency named at construction
    (`market`, default "SPY"). Its column must be present in the input prices;
    it is consumed as the benchmark and removed from the output, so the signal
    covers the investable universe (input columns minus `market`).
    """

    def __init__(
        self,
        beta_window: int = 63,
        momentum_window: int = 252,
        skip: int = 21,
        market: str = "SPY",
    ) -> None:
        super().__init__(
            name=f"residual_momentum_{beta_window}_{momentum_window}_{skip}",
            lookback=beta_window + momentum_window + 1,
        )
        self.beta_window = beta_window
        self.momentum_window = momentum_window
        self.skip = skip
        self.market = market

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        if self.market not in prices.columns:
            raise ValueError(
                f"ResidualMomentum requires the market proxy column "
                f"'{self.market}' in prices; got {list(prices.columns)}."
            )

        returns = prices.pct_change()
        market_returns = returns[self.market]

        # Compute rolling covariance column-by-column to avoid pandas cross-join
        rolling_cov = pd.DataFrame(
            {
                col: returns[col].rolling(self.beta_window).cov(market_returns)
                for col in returns.columns
            },
            index=returns.index,
        )
        rolling_var = market_returns.rolling(self.beta_window).var()
        beta = rolling_cov.div(rolling_var, axis=0)

        residuals = returns - beta.mul(market_returns, axis=0)
        signal = residuals.shift(self.skip).rolling(
            self.momentum_window - self.skip
        ).sum()

        # The market is a covariate, not an investable name: drop it so the
        # output covers exactly the investable universe.
        return signal.drop(columns=self.market)