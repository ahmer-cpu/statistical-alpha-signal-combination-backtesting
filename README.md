# Statistical Alpha Signal Combination & Robust Backtesting for US Equities

A Python research project for **building, validating, and combining
cross-sectional equity alpha signals** — with a statistically robust backtesting
and validation layer designed to isolate alpha from noise.

Every factor is gated through Newey-West-corrected
IC tests, block-bootstrap confidence intervals, IC-decay curves, and a turnover/cost-aware
backtest before it is allowed into a combined signal.

---

## Headline result

After testing a 13-factor library on the S&P 100 (2021–2026), mapping factor redundancy
with a K×K IC-correlation matrix, and running a three-combiner comparison across horizons
and market regimes, the research **delivers a single combined alpha signal**:

> **Signed IC-weighted (EWM, 126-day half-life) blend of sector-neutral momentum + low-volatility,
> rebalanced every 31 days.**
>
> - **Annualized alpha** — 13.53%
> - **Market-neutral** — β ≈ −0.02 to SPY
> - **Full-sample Sharpe ≈ 0.99** (cumulative PnL +73%, hit rate 56%) with **10 bps** rebalancing cost
> - **Positive in *both* market conditions** — Sharpe **+0.89** through the 2022 bear (a *positive* return
>   while SPY fell ~18%) and **+1.03** through the bull

The largest contribution to drawdown occurs during **regime-change** and periods of high volatility - the results do not claim to find tradeable alpha and the caveats are as follows:
- Membership/selection bias: the study used current S&P 500 members' history over the sample period, not a point-in-time membership. This biases the results upwards.
- Structural "lookahead" with statistical validation over the full sample prior to factor selection. This can be avoided by true train/test splits of the data - however, the factor selection remained unchanged.
- A much larger study over number of assets and history is needed to fully ascertain the results (e.g. the period only observed a single extended bear market).

Naive IC significance tests on **overlapping** forward returns badly overstate confidence. Applying a Newey-West HAC correction to the IC t-statistics in this project deflated apparent significance by **~3.5×** — enough to change several "significant" single factors into null results. The Lasso combiner (rolling Lasso regression on forward returns, using the coefficients as factor weights) delivered similar results.

---

## What's inside

- **Alpha factor library** — an abstract `Alpha` base class with **13 factor implementations**
  across four families:
  - *Price-momentum:* momentum (12–1), cross-sectional momentum, residual (beta-adjusted)
    momentum, sector-neutral momentum
  - *Mean-reversion:* RSI, Bollinger z-score, short-term reversal
  - *Volatility / risk:* low-vol (betting-against-beta), rolling Sharpe
  - *Tail / path:* rolling CVaR, rolling max-drawdown, rolling skewness, plus a quality proxy
- **Signal combination** — equal-weight, **signed IC-weighted** (flat or EWM-decayed rolling IC),
  and **Lasso** combiners, plus a K×K multi-factor IC-correlation matrix for redundancy analysis
- **Vectorized backtester** — dollar-neutral, unit-gross-exposure signal backtester with
  configurable rebalance frequency, position smoothing, and transaction costs (bps)
- **IC analysis** — Spearman rank IC, ICIR, **Newey-West HAC t-statistics**, non-overlapping IC,
  IC-decay curves, and **block-bootstrap 95% confidence intervals**
- **Signal validation** — autocorrelation, turnover, cross-sectional dispersion, lead-lag profiles,
  sub-period stability, leave-one-out, and sector splits, all orchestrated by a one-call `SignalReport`
- **Performance analytics** — 22 metrics (Sharpe, Sortino, Calmar, Jensen's alpha, beta, tracking
  error, information ratio, up/down capture, VaR, CVaR, tail ratio, skewness, kurtosis, …)
- **Tear sheet** — full-page visual tearsheet: metrics banner, equity curve, drawdown,
  monthly-returns heatmap
- **Data pipeline** — Alpaca client with incremental Parquet storage (adjustment-separated,
  metadata-embedded), an S&P 500 / S&P 100 constituent + GICS-sector scraper, and a budget-aware
  Alpha Vantage fundamentals collector
- **242 unit tests** covering metrics, factors, combiners, the backtester, IC analysis, and diagnostics

---

## Repository layout

```
src/qre/
├── core/         Core domain types: Bar, Signal, Order, Fill, Position
├── data/         Alpaca client, Parquet store, universe scraper, fundamentals collector
├── alpha/
│   ├── factors/  13 alpha factors (momentum, reversal, volatility, tail families)
│   └── combination.py   Equal-weight, IC-weighted, and Lasso signal combiners
├── backtest/     Vectorized backtester (rebalance frequency, smoothing, costs)
└── analytics/
    ├── metrics.py          22 performance / benchmark / distribution metrics
    ├── ic.py               IC analysis (Newey-West, bootstrap CIs, non-overlapping IC)
    ├── multi_factor_ic.py  K×K IC-correlation matrix for factor redundancy
    ├── tearsheet.py        Full-page visual tearsheet
    └── validation/         Diagnostics battery + one-call SignalReport
notebooks/        01 data exploration · 02 factor analysis · 03 signal validation · 04 alpha research
tests/unit/       242 unit tests
```

## Research notebooks

| Notebook | What it covers |
|----------|----------------|
| [01 — Data Exploration](notebooks/01_data_exploration.ipynb) | Price analysis, return distributions, cross-sector correlations, rolling volatility |
| [02 — Factor Analysis](notebooks/02_factor_analysis.ipynb) | Single-factor screening: IC analysis, factor correlations, vectorized backtests, elimination |
| [03 — Signal Validation](notebooks/03_signal_validation.ipynb) | Statistical rigour: NW-corrected IC, bootstrap CIs, IC decay, diagnostics, robustness battery |
| [04 — Alpha Research & Combination](notebooks/04_alpha_research.ipynb) | 13-factor triage, IC-correlation redundancy map, three-combiner bake-off, regime/transition stress tests, and the **frozen combined alpha** |

---

## Quick start

```bash
git clone https://github.com/ahmer-cpu/statistical-alpha-signal-combination-backtesting.git
cd statistical-alpha-signal-combination-backtesting
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash;  use .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"

# Run the test suite
pytest -q

# (Optional) fetch data — requires Alpaca API keys
cp .env.example .env                 # then edit with your credentials
python -c "
from datetime import datetime
from qre.data.historical import HistoricalDataStore
store = HistoricalDataStore()
store.fetch_and_store('AAPL', '1d', datetime(2021, 1, 1), datetime.now())
"
```

The S&P 100 / S&P 500 constituent + GICS-sector files ship in `data/`. Price history is fetched
on demand via Alpaca and cached as Parquet under `data/adjusted/` (gitignored).

## Tech stack

| Component | Tool |
|-----------|------|
| Language | Python 3.13 |
| Data / broker | Alpaca |
| Storage | Apache Parquet (PyArrow) |
| Stats | statsmodels, scipy, scikit-learn |
| Type checking | mypy (strict) |
| Linting | ruff |
| Testing | pytest |

## License

MIT
