"""Historical data storage — fetch from Alpaca and persist as Parquet files."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from alpaca.data.enums import Adjustment

from qre.data.alpaca_client import AlpacaClient

# Project root / data
DATA_ROOT = Path(__file__).resolve().parents[3] / "data"

# Maps user-facing adjustment names to Alpaca enum + storage directory
ADJUSTMENT_MODES = {
    "adjusted": "adjusted",    # splits + dividends (Adjustment.ALL)
    "unadjusted": "unadjusted",  # raw prices (Adjustment.RAW)
}


class HistoricalDataStore:
    """Manages fetching and storing OHLCV bars as Parquet files.

    Storage layout:
        data/{adjustment}/{TICKER}/{timeframe}.parquet
        e.g., data/adjusted/AAPL/1d.parquet

    Each Parquet file includes metadata: source, adjustment mode,
    fetch timestamp, and schema version for reproducibility.
    """

    def __init__(
        self,
        data_root: Path = DATA_ROOT,
        adjustment: str = "adjusted",
        client: AlpacaClient | None = None,
    ) -> None:
        if adjustment not in ADJUSTMENT_MODES:
            raise ValueError(
                f"Invalid adjustment '{adjustment}'. "
                f"Must be one of: {list(ADJUSTMENT_MODES.keys())}"
            )
        self._data_root = data_root
        self._adjustment = adjustment
        self._data_dir = data_root / ADJUSTMENT_MODES[adjustment]
        self._client = client

    @property
    def client(self) -> AlpacaClient:
        """Lazy-initialize AlpacaClient on first use (requires API keys)."""
        if self._client is None:
            self._client = AlpacaClient()
        return self._client

    def _parquet_path(self, ticker: str, timeframe: str) -> Path:
        """Get the file path for a ticker/timeframe combination."""
        return self._data_dir / ticker / f"{timeframe}.parquet"

    def _alpaca_adjustment(self) -> Adjustment:
        """Map our adjustment mode to the Alpaca SDK enum."""
        if self._adjustment == "adjusted":
            return Adjustment.ALL
        return Adjustment.RAW

    def _write_parquet(
        self, df: pd.DataFrame, path: Path, ticker: str, timeframe: str,
    ) -> None:
        """Write DataFrame to Parquet with embedded metadata."""
        table = pa.Table.from_pandas(df)

        metadata = {
            b"ticker": ticker.encode(),
            b"timeframe": timeframe.encode(),
            b"adjustment": self._adjustment.encode(),
            b"source": b"alpaca",
            b"start": str(df.index.min()).encode(),
            b"end": str(df.index.max()).encode(),
            b"fetch_timestamp": datetime.now(UTC).isoformat().encode(),
            b"schema_version": b"1",
        }

        # Merge with any existing Arrow metadata (e.g., pandas schema)
        existing = table.schema.metadata or {}
        merged = {**existing, **metadata}
        table = table.replace_schema_metadata(merged)

        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)  # type: ignore[no-untyped-call]

    def fetch_and_store(
        self,
        ticker: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch bars from Alpaca and save to Parquet.

        If data already exists on disk, fetches only the missing ranges
        (before the earliest stored or after the latest stored) and merges.

        Args:
            ticker: Stock symbol (e.g., "AAPL").
            timeframe: Bar size (e.g., "1d", "1h").
            start: Start datetime for the fetch.
            end: End datetime for the fetch.

        Returns:
            The complete DataFrame (existing + new data).
        """
        path = self._parquet_path(ticker, timeframe)
        existing = self.load(ticker, timeframe)
        adj = self._alpaca_adjustment()

        chunks: list[pd.DataFrame] = []

        if existing is not None and not existing.empty:
            # Strip timezone so comparisons work with naive user-supplied datetimes
            first_timestamp = existing.index.min().to_pydatetime().replace(tzinfo=None)
            last_timestamp = existing.index.max().to_pydatetime().replace(tzinfo=None)

            # Backfill: fetch data before the earliest stored timestamp
            if start < first_timestamp:
                backfill = self.client.get_bars(
                    ticker, timeframe, start, first_timestamp, adjustment=adj,
                )
                if not backfill.empty:
                    chunks.append(backfill)

            chunks.append(existing)

            # Forward-fill: fetch data after the latest stored timestamp
            if end > last_timestamp:
                forward = self.client.get_bars(
                    ticker, timeframe, last_timestamp, end, adjustment=adj,
                )
                if not forward.empty:
                    chunks.append(forward)
        else:
            new_data = self.client.get_bars(
                ticker, timeframe, start, end, adjustment=adj,
            )
            if not new_data.empty:
                chunks.append(new_data)

        if not chunks:
            return existing if existing is not None else pd.DataFrame()

        combined = pd.concat(chunks)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()

        self._write_parquet(combined, path, ticker, timeframe)

        return combined

    def load(self, ticker: str, timeframe: str) -> pd.DataFrame | None:
        """Load stored bars from Parquet.

        Returns:
            DataFrame if file exists, None otherwise.
        """
        path = self._parquet_path(ticker, timeframe)

        if not path.exists():
            return None

        return pd.read_parquet(path)

    def load_metadata(self, ticker: str, timeframe: str) -> dict[str, str] | None:
        """Read the embedded metadata from a stored Parquet file.

        Returns:
            Dict of metadata key-value pairs, or None if file doesn't exist.
        """
        path = self._parquet_path(ticker, timeframe)

        if not path.exists():
            return None

        schema = pq.read_schema(path)  # type: ignore[no-untyped-call]
        if schema.metadata is None:
            return None

        # Decode byte keys/values, skip the pandas internal metadata
        return {
            k.decode(): v.decode()
            for k, v in schema.metadata.items()
            if not k.startswith(b"pandas")
        }

    def load_panel(
        self, tickers: list[str] | None = None, timeframe: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Load multiple tickers into wide DataFrames (date x ticker).

        Args:
            tickers: List of tickers to load. If None, loads all available.
            timeframe: Bar size (default "1d").

        Returns:
            Dict with keys "close", "open", "high", "low", "volume",
            each a DataFrame indexed by date with ticker columns.
        """
        if tickers is None:
            tickers = self.list_tickers(timeframe)

        frames: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = self.load(ticker, timeframe)
            if df is not None:
                frames[ticker] = df

        if not frames:
            cols = ["open", "high", "low", "close", "volume"]
            return {col: pd.DataFrame() for col in cols}

        panel: dict[str, pd.DataFrame] = {}
        for col in ["open", "high", "low", "close", "volume"]:
            panel[col] = pd.DataFrame({
                ticker: df[col] for ticker, df in frames.items()
            })

        return panel

    def list_tickers(self, timeframe: str = "1d") -> list[str]:
        """List all tickers that have stored data for a given timeframe."""
        tickers: list[str] = []
        if not self._data_dir.exists():
            return tickers

        for ticker_dir in sorted(self._data_dir.iterdir()):
            if ticker_dir.is_dir() and (ticker_dir / f"{timeframe}.parquet").exists():
                tickers.append(ticker_dir.name)

        return tickers
