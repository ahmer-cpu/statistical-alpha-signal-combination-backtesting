"""Budget-aware daily collector for Alpha Vantage fundamental statements.

Pulls EARNINGS / INCOME_STATEMENT / BALANCE_SHEET for the universe. Each call
returns several years of quarterly history, so the binding constraint is the
free-tier cap (25 calls/day), not history depth. A JSON ledger records the last
successful fetch per (ticker, endpoint) so runs are resumable: the first ~week
backfills full history, then the loop settles into monthly maintenance to pick
up newly reported quarters.

Lookahead note: EARNINGS carries ``reportedDate`` (when the market learned each
quarter's numbers). The downstream panel builder aligns income/balance figures
to that date for point-in-time correctness. This module only *collects and
stores* the raw statements as JSON; parsing into a panel is a separate step.

Run:
    python -m qre.data.fundamentals_daily              # one budgeted cycle
    python -m qre.data.fundamentals_daily --dry-run    # show plan, no API calls
    python -m qre.data.fundamentals_daily --budget 25  # override daily budget
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path

import requests  # type: ignore[import-untyped]

from qre.data.alphavantage import (
    _RATE_LIMIT_PAUSE,
    AlphaVantageRateLimitError,
    fetch_balance_sheet,
    fetch_earnings,
    fetch_income_statement,
)
from qre.data.historical import DATA_ROOT, HistoricalDataStore

logger = logging.getLogger(__name__)

_STATEMENTS_DIR = DATA_ROOT / "fundamentals" / "statements"
_LEDGER_PATH = DATA_ROOT / "fundamentals" / "_ledger.json"

# Endpoint -> fetcher function.
_FETCHERS: dict[str, Callable[[str], dict[str, object]]] = {
    "EARNINGS": fetch_earnings,
    "INCOME_STATEMENT": fetch_income_statement,
    "BALANCE_SHEET": fetch_balance_sheet,
}
# Endpoint -> JSON key holding its quarterly history (presence = a good response).
_REPORTS_KEY: dict[str, str] = {
    "EARNINGS": "quarterlyEarnings",
    "INCOME_STATEMENT": "quarterlyReports",
    "BALANCE_SHEET": "quarterlyReports",
}
ENDPOINTS = tuple(_FETCHERS)

# ledger[ticker][endpoint] = ISO date of last successful fetch, or None.
Ledger = dict[str, dict[str, str | None]]


def _universe() -> list[str]:
    """Universe tickers minus SPY (an ETF with no company fundamentals)."""
    store = HistoricalDataStore()
    return [t for t in store.list_tickers() if t != "SPY"]


def _load_ledger() -> Ledger:
    if not _LEDGER_PATH.exists():
        return {}
    loaded: Ledger = json.loads(_LEDGER_PATH.read_text())
    return loaded


def _save_ledger(ledger: Ledger) -> None:
    _LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LEDGER_PATH.write_text(json.dumps(ledger, indent=2, sort_keys=True))


def _normalize(ledger: Ledger, universe: list[str]) -> Ledger:
    """Ensure every (ticker, endpoint) has an entry (None = never fetched)."""
    for ticker in universe:
        slot = ledger.setdefault(ticker, {})
        for endpoint in ENDPOINTS:
            slot.setdefault(endpoint, None)
    return ledger


def _is_stale(last: str | None, refresh_days: int, today: date) -> bool:
    if last is None:
        return True
    return (today - date.fromisoformat(last)).days >= refresh_days


def _build_queue(
    ledger: Ledger, universe: list[str], refresh_days: int, today: date
) -> list[tuple[str, str]]:
    """Units needing a fetch: never-fetched first, then stalest-first."""
    never: list[tuple[str, str]] = []
    stale: list[tuple[tuple[str, str], str]] = []  # (unit, last_date)
    for ticker in universe:
        for endpoint in ENDPOINTS:
            last = ledger[ticker][endpoint]
            if last is None:
                never.append((ticker, endpoint))
            elif _is_stale(last, refresh_days, today):
                stale.append(((ticker, endpoint), last))
    stale.sort(key=lambda item: item[1])  # oldest last-fetch first
    return never + [unit for unit, _ in stale]


def _save_raw(ticker: str, endpoint: str, data: dict[str, object]) -> Path:
    _STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _STATEMENTS_DIR / f"{ticker}_{endpoint}.json"
    path.write_text(json.dumps(data))
    return path


def run_cycle(
    budget: int = 20,
    refresh_days: int = 30,
    pause: float = _RATE_LIMIT_PAUSE,
    dry_run: bool = False,
) -> dict[str, int]:
    """Run one budgeted collection cycle. Returns a {fetched, errors, remaining}.

    Args:
        budget: Max API calls this run (free tier caps at 25/day).
        refresh_days: Re-fetch a statement once its last fetch is this old.
        pause: Seconds between calls (5/min limit -> ~13s).
        dry_run: Log the plan and exit without making any API calls.
    """
    today = date.today()
    universe = _universe()
    ledger = _normalize(_load_ledger(), universe)
    queue = _build_queue(ledger, universe, refresh_days, today)

    total_units = len(universe) * len(ENDPOINTS)
    logger.info(
        "Universe %d tickers x %d endpoints = %d units; %d need fetching; "
        "budget %d/run.",
        len(universe), len(ENDPOINTS), total_units, len(queue), budget,
    )

    if dry_run:
        preview = queue[: budget]
        logger.info("DRY RUN — would fetch %d unit(s) this cycle:", len(preview))
        for ticker, endpoint in preview:
            logger.info("  %s %s", ticker, endpoint)
        if queue:
            runs_left = -(-len(queue) // max(budget, 1))  # ceil division
            logger.info("~%d run(s) to clear the current backlog.", runs_left)
        _save_ledger(ledger)  # persist the normalized skeleton
        return {"fetched": 0, "errors": 0, "remaining": len(queue)}

    fetched = 0
    errors = 0
    for ticker, endpoint in queue:
        if fetched + errors >= budget:
            break
        if fetched + errors > 0:
            time.sleep(pause)  # space calls; never trails the loop

        try:
            data = _FETCHERS[endpoint](ticker)
        except AlphaVantageRateLimitError as exc:
            logger.warning("Daily cap reached — stopping cycle: %s", exc)
            break
        except (ValueError, requests.RequestException) as exc:
            logger.warning("  %s %s — error, skipping: %s", ticker, endpoint, exc)
            errors += 1
            continue

        quarters = data.get(_REPORTS_KEY[endpoint])
        n_quarters = len(quarters) if isinstance(quarters, list) else 0
        if n_quarters == 0:
            logger.warning(
                "  %s %s — no quarterly reports in response, skipping.",
                ticker, endpoint,
            )
            errors += 1
            continue

        _save_raw(ticker, endpoint, data)
        ledger[ticker][endpoint] = today.isoformat()
        _save_ledger(ledger)  # persist after each success -> resumable
        fetched += 1
        logger.info(
            "  [%d/%d] %s %s — saved %d quarters.",
            fetched + errors, budget, ticker, endpoint, n_quarters,
        )

    remaining = len(_build_queue(ledger, universe, refresh_days, today))
    logger.info(
        "Cycle done: %d fetched, %d errors, %d unit(s) still pending.",
        fetched, errors, remaining,
    )
    return {"fetched": fetched, "errors": errors, "remaining": remaining}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Alpha Vantage fundamentals within the daily cap."
    )
    parser.add_argument(
        "--budget", type=int, default=20,
        help="Max API calls this run (free tier cap is 25/day). Default 20.",
    )
    parser.add_argument(
        "--refresh-days", type=int, default=30,
        help="Re-fetch a statement once its last fetch is this old. Default 30.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show the plan without making any API calls.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )
    run_cycle(
        budget=args.budget,
        refresh_days=args.refresh_days,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
