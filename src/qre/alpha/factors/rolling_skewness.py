import pandas as pd

from qre.alpha.base import Alpha


class RollingSkewness(Alpha):
    """Rolling skewness factor (Boyer, Mitton & Vorkink, 2010).

    Captures the "lottery demand" effect: stocks with positive skewness
    (lottery-like payoffs) are overpriced by retail investors seeking
    large upside, and subsequently underperform. Stocks with negative
    skewness are underpriced and outperform. Boyer, Mitton & Vorkink
    (2010) price *expected idiosyncratic* skewness; here we use *realized*
    trailing skewness as a tractable proxy. (Bali, Cakici & Whitelaw,
    2011, document the related lottery effect via MAX, the maximum daily
    return.)

    Formula: -rolling_skew(r, window)
    Signal: positive = negative skew (buy), negative = positive skew (sell).
    Timing: uses prices up to close of date t; no lookahead.
    NaN warmup: first `window + 1` rows are NaN (1 for returns, window for rolling).
    Note: 3rd moment — noisier than 1st/2nd moment signals but finite
    for most plausible return distributions (Student-t with df > 3).
    """

    def __init__(self, window: int = 252) -> None:
        super().__init__(name=f"rolling_skewness_{window}", lookback=window + 1)
        self.window = window

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change()
        return -returns.rolling(self.window).skew()
