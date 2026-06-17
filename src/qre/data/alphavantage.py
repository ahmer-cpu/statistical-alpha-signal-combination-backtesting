"""Alpha Vantage fundamental data — fetch and cache company financials."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests  # type: ignore[import-untyped]
from dotenv import load_dotenv

from qre.data.historical import DATA_ROOT

load_dotenv()

logger = logging.getLogger(__name__)

_FUNDAMENTALS_DIR = DATA_ROOT / "fundamentals"
_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
_BASE_URL = "https://www.alphavantage.co/query"

# Free tier: 25 requests/day, 5 requests/minute
_RATE_LIMIT_PAUSE = 13.0  # seconds between requests (safe for 5/min)


class AlphaVantageRateLimitError(ValueError):
    """Raised when Alpha Vantage signals its minute/daily request cap.

    Subclasses ValueError so existing ``except ValueError`` handlers still
    catch it, while budget-aware callers can catch it specifically to stop a
    run cleanly instead of burning attempts against an exhausted quota.
    """


def _api_request(function: str, symbol: str) -> dict[str, object]:
    """Make a single Alpha Vantage API request.

    Raises:
        ValueError: If the API returns an error or rate limit note.
        RuntimeError: If the API key is not set.
    """
    if not _API_KEY:
        raise RuntimeError(
            "ALPHAVANTAGE_API_KEY not set in .env. "
            "Get a free key at https://www.alphavantage.co/support/"
        )

    params = {
        "function": function,
        "symbol": symbol,
        "apikey": _API_KEY,
    }
    resp = requests.get(_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data: dict[str, object] = resp.json()

    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
    # "Note" = per-minute throttle; "Information" = daily-cap message (newer API).
    if "Note" in data:
        raise AlphaVantageRateLimitError(f"Rate limit (Note): {data['Note']}")
    if "Information" in data:
        info = data["Information"]
        raise AlphaVantageRateLimitError(f"Rate limit (Information): {info}")

    return data


def fetch_overview(symbol: str) -> dict[str, object]:
    """Fetch company overview (P/E, ROE, EPS, margins, etc.)."""
    return _api_request("OVERVIEW", symbol)


def fetch_income_statement(symbol: str) -> dict[str, object]:
    """Fetch quarterly and annual income statements."""
    return _api_request("INCOME_STATEMENT", symbol)


def fetch_balance_sheet(symbol: str) -> dict[str, object]:
    """Fetch quarterly and annual balance sheets."""
    return _api_request("BALANCE_SHEET", symbol)


def fetch_earnings(symbol: str) -> dict[str, object]:
    """Fetch quarterly and annual earnings.

    The quarterly records include ``reportedDate`` — the day each quarter's
    numbers were actually announced — which is the point-in-time stamp used to
    align income/balance figures without lookahead bias.
    """
    return _api_request("EARNINGS", symbol)


def fetch_fundamentals(
    tickers: list[str],
    output_dir: Path = _FUNDAMENTALS_DIR,
    pause: float = _RATE_LIMIT_PAUSE,
) -> pd.DataFrame:
    """Fetch and cache fundamental data for a list of tickers.

    For each ticker, fetches the company overview (1 API call) which
    contains ROE, profit margin, EPS, P/E, beta, and other key metrics.
    Results are cached as individual CSVs and combined into a single
    DataFrame.

    Args:
        tickers: List of ticker symbols to fetch.
        output_dir: Directory to cache individual ticker CSVs.
        pause: Seconds between API calls (default 13s for free tier).

    Returns:
        DataFrame with one row per ticker and fundamental columns.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    fetched = 0

    for i, ticker in enumerate(tickers):
        cache_path = output_dir / f"{ticker}_overview.csv"

        # Use cache if it exists and is less than 7 days old
        if cache_path.exists():
            age_days = (time.time() - cache_path.stat().st_mtime) / 86400
            if age_days < 7:
                logger.info("[%d/%d] %s — using cache", i + 1, len(tickers), ticker)
                cached_df = pd.read_csv(cache_path, index_col=0)
                cached_series = cached_df.iloc[:, 0]
                results.append(
                    {str(k): v for k, v in cached_series.to_dict().items()}
                )
                continue

        logger.info("[%d/%d] %s — fetching from API", i + 1, len(tickers), ticker)
        try:
            data = fetch_overview(ticker)
            fetched += 1
        except ValueError as e:
            logger.warning("  Skipping %s: %s", ticker, e)
            continue

        # Save individual ticker cache
        series = pd.Series(data, name=ticker)
        series.to_csv(cache_path)
        results.append(data)

        # Rate limiting (skip pause after last ticker)
        if i < len(tickers) - 1:
            logger.info("  Waiting %.0fs (rate limit)...", pause)
            time.sleep(pause)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Select and clean key fundamental columns
    key_cols = {
        "Symbol": "ticker",
        "Name": "company",
        "Sector": "sector",
        "ReturnOnEquityTTM": "roe",
        "ProfitMargin": "profit_margin",
        "OperatingMarginTTM": "operating_margin",
        "EPS": "eps",
        "PERatio": "pe_ratio",
        "DividendYield": "dividend_yield",
        "Beta": "beta",
        "QuarterlyEarningsGrowthYOY": "earnings_growth",
        "GrossProfitTTM": "gross_profit",
        "MarketCapitalization": "market_cap",
    }

    available = {k: v for k, v in key_cols.items() if k in df.columns}
    clean = df[list(available.keys())].rename(columns=available)

    # Convert numeric columns
    numeric_cols = [
        "roe", "profit_margin", "operating_margin", "eps", "pe_ratio",
        "dividend_yield", "beta", "earnings_growth", "gross_profit",
        "market_cap",
    ]
    for col in numeric_cols:
        if col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")

    # Save combined file
    combined_path = output_dir / "fundamentals_overview.csv"
    clean.to_csv(combined_path, index=False)
    logger.info(
        "Saved %d tickers to %s (%d API calls used)",
        len(clean), combined_path, fetched,
    )

    return clean


def load_fundamentals(path: Path | None = None) -> pd.DataFrame:
    """Load cached fundamentals overview from CSV.

    Args:
        path: Path to fundamentals_overview.csv. Defaults to
              data/fundamentals/fundamentals_overview.csv.

    Returns:
        DataFrame with one row per ticker.

    Raises:
        FileNotFoundError: If CSV does not exist. Run fetch_fundamentals() first.
    """
    if path is None:
        path = _FUNDAMENTALS_DIR / "fundamentals_overview.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run fetch_fundamentals() first."
        )

    return pd.read_csv(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Fetch for the current 47-ticker universe (minus SPY which is an ETF)
    from qre.data.historical import HistoricalDataStore

    store = HistoricalDataStore()
    tickers = [t for t in store.list_tickers() if t != "SPY"]
    print(f"Fetching fundamentals for {len(tickers)} tickers...")
    print(f"Estimated API calls: {len(tickers)}")
    print("Free tier limit: 25/day\n")

    df = fetch_fundamentals(tickers)
    print(f"\nDone. Shape: {df.shape}")
    print(df.head(10).to_string())
