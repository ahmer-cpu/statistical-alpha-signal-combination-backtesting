"""Thin wrapper around Alpaca's SDK for market data access."""

import os
from datetime import datetime

import pandas as pd
from alpaca.data.enums import Adjustment
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

# Maps our string timeframe codes to Alpaca's TimeFrame objects
TIMEFRAME_MAP: dict[str, TimeFrame] = {
    "1min": TimeFrame(1, TimeFrameUnit.Minute),
    "5min": TimeFrame(5, TimeFrameUnit.Minute),
    "15min": TimeFrame(15, TimeFrameUnit.Minute),
    "1h": TimeFrame(1, TimeFrameUnit.Hour),
    "1d": TimeFrame(1, TimeFrameUnit.Day),
}


class AlpacaClient:
    """Authenticated client for fetching historical market data from Alpaca."""

    def __init__(self) -> None:
        load_dotenv()

        api_key = os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env"
            )

        self._client = StockHistoricalDataClient(api_key, secret_key)

    def get_bars(
        self,
        ticker: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        adjustment: Adjustment = Adjustment.ALL,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for a single ticker.

        Args:
            ticker: Stock symbol (e.g., "AAPL").
            timeframe: Bar size - one of "1min", "5min", "15min", "1h", "1d".
            start: Start datetime (inclusive).
            end: End datetime (exclusive).
            adjustment: Price adjustment mode (default: ALL = splits + dividends).

        Returns:
            DataFrame with columns: open, high, low, close, volume,
            indexed by timestamp.
        """
        tf = TIMEFRAME_MAP.get(timeframe)
        if tf is None:
            valid = ", ".join(TIMEFRAME_MAP.keys())
            raise ValueError(
                f"Invalid timeframe '{timeframe}'. Must be one of: {valid}"
            )

        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=tf,
            start=start,
            end=end,
            adjustment=adjustment,
        )

        bars = self._client.get_stock_bars(request)
        df: pd.DataFrame = bars.df  # type: ignore[union-attr]

        if df.empty:
            return df

        # Alpaca returns a MultiIndex (symbol, timestamp) — drop the symbol level
        df = df.droplevel("symbol")

        # Keep only the columns we care about
        df = df[["open", "high", "low", "close", "volume"]]

        return df
