"""S&P 500 universe data — fetch constituents, weights, and GICS sectors."""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests  # type: ignore[import-untyped]
from bs4 import BeautifulSoup

from qre.data.historical import DATA_ROOT

logger = logging.getLogger(__name__)

_DEFAULT_CSV = DATA_ROOT / "sp500_constituents.csv"
_SP100_CSV = DATA_ROOT / "sp100_constituents.csv"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
}


def _normalize_ticker(ticker: str) -> str:
    """Normalize ticker symbols to the Alpaca convention.

    Alpaca uses a dot for class shares (e.g. BRK.B, BF.B). Constituent sources
    sometimes use a hyphen or slash; convert those to a dot so the symbol is
    fetchable from Alpaca and used consistently everywhere (the data store,
    the constituent CSVs, and the notebooks).
    """
    return ticker.strip().upper().replace("-", ".").replace("/", ".")


def _fetch_slickcharts() -> pd.DataFrame:
    """Scrape S&P 500 constituents with weights from SlickCharts.

    Returns DataFrame with columns: ticker, company, weight, rank.
    """
    url = "https://www.slickcharts.com/sp500"
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError("Could not find constituents table on SlickCharts")

    df = pd.read_html(StringIO(str(table)))[0]

    # SlickCharts columns: #, Company, Symbol, Weight, Price, Chg, % Chg
    df = df.rename(columns={
        "#": "rank",
        "Company": "company",
        "Symbol": "ticker",
        "Weight": "weight",
    })
    df["ticker"] = df["ticker"].apply(_normalize_ticker)
    df = df[["rank", "ticker", "company", "weight"]].copy()
    df["rank"] = df["rank"].astype(int)
    # Strip '%' suffix if present, then convert to float
    df["weight"] = (
        df["weight"].astype(str).str.rstrip("%").pipe(pd.to_numeric, errors="coerce")
    )

    return df


def _fetch_wikipedia() -> pd.DataFrame:
    """Scrape S&P 500 GICS sector and sub-industry from Wikipedia.

    Returns DataFrame with columns: ticker, sector, sub_industry.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text), header=0)
    df = tables[0]

    # Wikipedia columns: Symbol, Security, GICS Sector, GICS Sub-Industry, ...
    df = df.rename(columns={
        "Symbol": "ticker",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
    })
    df["ticker"] = df["ticker"].apply(_normalize_ticker)
    df = df[["ticker", "sector", "sub_industry"]].copy()

    return df


def fetch_sp500(output_path: Path = _DEFAULT_CSV) -> pd.DataFrame:
    """Fetch S&P 500 constituents from SlickCharts and Wikipedia, merge, and save.

    Performs an outer join on ticker so no constituents are lost from
    either source. Logs any tickers that appear in only one list.

    Args:
        output_path: Where to save the CSV. Defaults to data/sp500_constituents.csv.

    Returns:
        Merged DataFrame with columns: rank, ticker, company, weight,
        sector, sub_industry. Ordered by weight descending.
    """
    logger.info("Fetching S&P 500 from SlickCharts...")
    slick = _fetch_slickcharts()
    logger.info("Fetched %d tickers from SlickCharts", len(slick))

    logger.info("Fetching S&P 500 from Wikipedia...")
    wiki = _fetch_wikipedia()
    logger.info("Fetched %d tickers from Wikipedia", len(wiki))

    merged = slick.merge(wiki, on="ticker", how="outer", indicator=True)

    # Report mismatches
    left_only = merged.loc[merged["_merge"] == "left_only", "ticker"].tolist()
    right_only = merged.loc[merged["_merge"] == "right_only", "ticker"].tolist()

    if left_only:
        logger.warning(
            "Tickers in SlickCharts but not Wikipedia: %s", left_only,
        )
    if right_only:
        logger.warning(
            "Tickers in Wikipedia but not SlickCharts: %s", right_only,
        )

    merged = merged.drop(columns=["_merge"])
    merged = merged.sort_values("weight", ascending=False, na_position="last")
    merged = merged.reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    logger.info("Saved %d constituents to %s", len(merged), output_path)

    return merged


def fetch_sp100(output_path: Path = _SP100_CSV) -> pd.DataFrame:
    """Fetch S&P 100 (OEX) constituents and GICS sectors from Wikipedia, save.

    The S&P 100 is the ~100 largest, most established S&P 500 names. The
    Wikipedia components table provides ticker, company name, and sector — so
    the saved CSV is directly usable by ``load_sector_map`` (same schema).

    Args:
        output_path: Where to save the CSV. Defaults to data/sp100_constituents.csv.

    Returns:
        DataFrame with columns: ticker, company, sector.
    """
    url = "https://en.wikipedia.org/wiki/S%26P_100"
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text), header=0)

    # Find the components table — the one with a Symbol column.
    df = None
    for table in tables:
        cols = [str(c) for c in table.columns]
        if "Symbol" in cols:
            df = table
            break
    if df is None:
        raise ValueError("Could not find S&P 100 components table on Wikipedia")

    df = df.rename(columns={
        "Symbol": "ticker",
        "Name": "company",
        "Sector": "sector",
    })
    df["ticker"] = df["ticker"].apply(_normalize_ticker)
    keep = [c for c in ["ticker", "company", "sector"] if c in df.columns]
    df = df[keep].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d S&P 100 constituents to %s", len(df), output_path)

    return df


def load_sp100(path: Path = _SP100_CSV) -> pd.DataFrame:
    """Load the cached S&P 100 constituents (ticker, company, sector).

    Raises:
        FileNotFoundError: If the CSV does not exist. Run fetch_sp100() first.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run fetch_sp100() to create it."
        )
    return pd.read_csv(path)


def load_sector_map(
    path: Path = _DEFAULT_CSV,
    tickers: list[str] | None = None,
) -> dict[str, str]:
    """Load ticker → GICS sector mapping from the constituents CSV.

    Args:
        path: Path to sp500_constituents.csv.
        tickers: If provided, filter to only these tickers.

    Returns:
        Dict mapping ticker to GICS sector string.

    Raises:
        FileNotFoundError: If the CSV does not exist. Run fetch_sp500() first.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run fetch_sp500() to create it."
        )

    df = pd.read_csv(path)
    df = df.dropna(subset=["ticker", "sector"])

    if tickers is not None:
        df = df[df["ticker"].isin(tickers)]

    return dict(zip(df["ticker"], df["sector"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fetch_sp500()
