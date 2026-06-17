from abc import ABC, abstractmethod

import pandas as pd


class Alpha(ABC):
    """Abstract base class for alpha factors.

    Subclasses implement compute() to transform a close-price DataFrame
    into a signal DataFrame of the same shape (dates x tickers).

    Contract:
        Input:  prices DataFrame indexed by date, columns are tickers.
        Output: signal DataFrame with the same index and columns.

    Signal convention:
        positive = long, negative = short.
        Mean-reversion factors must negate their output so this holds.

    Timing convention:
        Signal at date t uses only information available by close of date t.
        If backtested against close-to-close returns from t to t+1, the
        signal is computed after the close and traded at the next open/close.
    """

    def __init__(self, name: str, lookback: int) -> None:
        self.name = name                # e.g, "momentum_12_1"
        self.lookback = lookback        # trading days needed

    @abstractmethod
    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Take a close-price DataFrame, and return a signal DataFrame."""
        ...
