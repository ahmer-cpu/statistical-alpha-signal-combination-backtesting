"""SignalReport — single entry point for end-to-end signal validation.

Wires together diagnostics, IC analysis, and backtesting into one pipeline.
Usage:
    report = SignalReport(signal, prices, name="Momentum").run()
    report.print_summary()
    report.plot()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from qre.analytics.ic import (
    block_bootstrap_ci,
    compute_ic,
    ic_decay,
    ic_summary,
)
from qre.analytics.metrics import sharpe_ratio
from qre.analytics.validation.diagnostics import (
    cross_sectional_dispersion,
    signal_autocorrelation,
    signal_return_lead_lag,
    signal_turnover,
)
from qre.backtest.vectorized import VectorizedBacktester


def _natural_freq(decay: pd.Series, decay_ci: pd.DataFrame) -> int:
    """Pick the rebalance horizon where the IC is most *reliably* strong.

    The naive choice — ``decay.idxmax()`` — pins to the scan's right boundary
    for a monotone-rising decay (e.g. a momentum-family signal whose
    overlapping long-horizon IC keeps creeping up on an ever-shrinking
    independent sample), and ignores reliability entirely. Instead, rank lags by
    a block-bootstrap t-stat ``|IC| / se``, where ``se`` comes from
    ``block_bootstrap_ci`` with ``block_length=lag`` and so HAC-corrects for the
    return overlap. That down-weights long horizons whose point estimate is big
    but whose standard error has blown up. ``abs`` makes it correct for
    robustly-negative signals too (the old signed idxmax picked the least
    negative, near-zero lag for those).

    Fallbacks: raw ``|IC|`` argmax when the bootstrap is off (se all NaN), and
    daily (1) when the decay is fully degenerate.
    """
    se = (
        decay_ci["se_bootstrap"]
        if "se_bootstrap" in decay_ci
        else pd.Series(dtype=float)
    )
    point = (
        decay_ci["point_estimate"] if "point_estimate" in decay_ci else decay
    )
    t_like = (point.abs() / se).where(se > 0)
    if t_like.notna().any():
        return int(t_like.idxmax())
    if decay.notna().any():
        return int(decay.abs().idxmax())
    return 1


@dataclass
class SignalReportConfig:
    """Tunables for the validation pipeline."""

    horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 21])
    ic_decay_max_lag: int = 30
    n_bootstrap: int = 10_000
    cost_bps: float = 10.0
    seed: int | None = 42


class SignalReport:
    """Orchestrates diagnostics → IC → backtest for a single signal."""

    def __init__(
        self,
        signal: pd.DataFrame,
        prices: pd.DataFrame,
        name: str = "Signal",
        config: SignalReportConfig | None = None,
    ) -> None:
        self.signal = signal
        self.prices = prices
        self.name = name
        self.config = config or SignalReportConfig()
        self._results: dict[str, Any] | None = None

    def run(self) -> SignalReport:
        """Execute the full pipeline. Returns self for chaining."""
        cfg = self.config
        results: dict[str, Any] = {}

        # --- 1. Diagnostics ---
        results["autocorrelation"] = signal_autocorrelation(self.signal)
        results["turnover"] = signal_turnover(self.signal)
        results["dispersion"] = cross_sectional_dispersion(self.signal)
        results["lead_lag"] = signal_return_lead_lag(self.signal, self.prices)

        # --- 2. IC at multiple horizons ---
        horizon_table = []
        for h in cfg.horizons:
            fwd_ret = self.prices.pct_change(h).shift(-h)
            ic_series = compute_ic(self.signal, fwd_ret)
            stats = ic_summary(
                ic_series,
                self.signal,
                fwd_ret,
                horizon=h,
                n_bootstrap=cfg.n_bootstrap,
                bootstrap_seed=cfg.seed,
            )
            stats["horizon"] = h
            horizon_table.append(stats)

        results["ic_by_horizon"] = pd.DataFrame(horizon_table)

        # --- 3. IC decay with bootstrap CIs ---
        decay = ic_decay(
            self.signal, self.prices, max_lag=cfg.ic_decay_max_lag,
        )
        results["ic_decay"] = decay

        # Bootstrap CI for each lag's mean IC
        decay_ci = {}
        for lag in decay.index:
            fwd_ret = self.prices.pct_change(lag).shift(-lag)
            ic_series = compute_ic(self.signal, fwd_ret)
            ci = block_bootstrap_ci(
                ic_series,
                block_length=lag,
                n_bootstrap=cfg.n_bootstrap,
                seed=cfg.seed,
            )
            decay_ci[lag] = ci
        results["ic_decay_ci"] = pd.DataFrame(decay_ci).T

        # --- 4. Natural rebalance frequency ---
        # Reliability-weighted, not raw-IC-argmax: see _natural_freq. Uses the
        # per-lag bootstrap se (computed just above) so it won't pin to the scan
        # boundary for a monotone-rising decay.
        best_lag = _natural_freq(decay, results["ic_decay_ci"])
        results["natural_freq"] = best_lag

        # --- 5. Backtest comparison ---
        bt_daily = VectorizedBacktester(cost_bps=cfg.cost_bps, rebalance_freq=1)
        bt_natural = VectorizedBacktester(
            cost_bps=cfg.cost_bps, rebalance_freq=best_lag,
        )
        ret_daily = bt_daily.run(self.prices, self.signal)
        ret_natural = bt_natural.run(self.prices, self.signal)

        results["backtest_daily"] = ret_daily
        results["backtest_natural"] = ret_natural
        results["sharpe_daily"] = sharpe_ratio(ret_daily)
        results["sharpe_natural"] = sharpe_ratio(ret_natural)

        self._results = results
        return self

    @property
    def results(self) -> dict[str, Any]:
        if self._results is None:
            raise RuntimeError("Call .run() before accessing results.")
        return self._results

    def print_summary(self) -> None:
        """Formatted text summary of validation results."""
        r = self.results
        cfg = self.config

        print(f"\n{'=' * 60}")
        print(f"  Signal Report: {self.name}")
        print(f"{'=' * 60}")

        # Diagnostics flags
        autocorr = r["autocorrelation"]
        lag1 = autocorr.loc[0, "autocorr"] if len(autocorr) > 0 else float("nan")
        mean_turnover = r["turnover"].iloc[1:].mean()
        mean_dispersion = r["dispersion"].mean()

        print("\n  Diagnostics")
        print(f"  {'─' * 40}")
        print(f"  Lag-1 autocorrelation:  {lag1:>8.3f}", end="")
        if lag1 > 0.95:
            print("  ⚠ quasi-static")
        else:
            print()
        print(f"  Mean daily turnover:   {mean_turnover:>8.3f}", end="")
        if mean_turnover < 0.05:
            print("  ⚠ very low")
        else:
            print()
        print(f"  Mean dispersion:       {mean_dispersion:>8.3f}", end="")
        if mean_dispersion < 0.01:
            print("  ⚠ near zero")
        else:
            print()

        # IC table
        ic_df = r["ic_by_horizon"]
        print("\n  IC by Horizon")
        print(f"  {'─' * 56}")
        print(f"  {'Horizon':>7}  {'Mean IC':>8}  {'ICIR':>6}  {'t-stat':>7}"
              f"  {'t(NW)':>7}  {'95% CI':>16}")
        print(f"  {'─' * 56}")
        for _, row in ic_df.iterrows():
            h = int(row["horizon"])
            ci_str = f"[{row['ci_lower']:+.3f}, {row['ci_upper']:+.3f}]"
            print(f"  {h:>5}d  {row['mean']:>+8.4f}  {row['icir']:>6.2f}"
                  f"  {row['t_stat']:>7.2f}  {row['t_stat_nw']:>7.2f}"
                  f"  {ci_str:>16}")

        # Natural frequency and backtest
        print(f"\n  Backtest Comparison (cost = {cfg.cost_bps:.0f} bps)")
        print(f"  {'─' * 40}")
        print(f"  Natural rebalance freq:  {r['natural_freq']}d")
        print(f"  Sharpe (daily):          {r['sharpe_daily']:>+.3f}")
        print(
            f"  Sharpe ({r['natural_freq']}d rebal):      "
            f"{r['sharpe_natural']:>+.3f}"
        )

        # Lead-lag quick read
        ll = r["lead_lag"]
        best_lead = ll[ll.index > 0].idxmax()
        best_lag_neg = ll[ll.index < 0].idxmax()
        print("\n  Lead-Lag Profile")
        print(f"  {'─' * 40}")
        print(f"  Strongest lead (predictive): offset {best_lead:+d}"
              f"  IC = {ll[best_lead]:+.4f}")
        print(f"  Strongest lag (reactive):    offset {best_lag_neg:+d}"
              f"  IC = {ll[best_lag_neg]:+.4f}")

        print(f"\n{'=' * 60}\n")

    def plot(self) -> Figure:
        """4-panel validation figure."""
        r = self.results

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(f"Signal Report: {self.name}", fontsize=14, fontweight="bold")

        # Panel 1: Autocorrelation
        ax = axes[0, 0]
        ac = r["autocorrelation"]
        ax.bar(ac["lag"], ac["autocorr"], color="#4a90d9", alpha=0.8)
        ax.axhline(
            0.95, color="red", linestyle="--", linewidth=0.8, label="0.95 threshold"
        )
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.set_title("Signal Autocorrelation")
        ax.legend(fontsize=8)

        # Panel 2: IC decay with bootstrap CIs
        ax = axes[0, 1]
        decay = r["ic_decay"]
        decay_ci = r["ic_decay_ci"]
        lags = decay.index.values
        ax.plot(lags, decay.values, "o-", color="#4a90d9", markersize=4)
        ax.fill_between(
            lags,
            decay_ci["ci_lower"].values,
            decay_ci["ci_upper"].values,
            alpha=0.2, color="#4a90d9",
        )
        ax.axhline(0, color="grey", linewidth=0.5)
        best = r["natural_freq"]
        ax.axvline(best, color="red", linestyle="--", linewidth=0.8,
                    label=f"Peak = {best}d")
        ax.set_xlabel("Horizon (days)")
        ax.set_ylabel("Mean IC")
        ax.set_title("IC Decay with 95% CI")
        ax.legend(fontsize=8)

        # Panel 3: Lead-lag profile
        ax = axes[1, 0]
        ll = r["lead_lag"]
        colors = ["#d9534f" if k < 0 else "#4a90d9" for k in ll.index]
        ax.bar(ll.index, ll.values, color=colors, alpha=0.8)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.axvline(0, color="grey", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Offset (negative = reactive, positive = predictive)")
        ax.set_ylabel("Mean IC")
        ax.set_title("Lead-Lag Profile")

        # Panel 4: Cumulative returns comparison
        ax = axes[1, 1]
        cum_daily = (1 + r["backtest_daily"]).cumprod()
        cum_natural = (1 + r["backtest_natural"]).cumprod()
        ax.plot(cum_daily.index, cum_daily.values, label="Daily rebalance",
                color="#6f6f6f")
        ax.plot(cum_natural.index, cum_natural.values,
                label=f"{best}d rebalance", color="#4a90d9")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.set_title(
            f"Backtest: Sharpe {r['sharpe_daily']:+.2f} (daily)"
            f" vs {r['sharpe_natural']:+.2f} ({best}d)"
        )
        ax.legend(fontsize=8)

        fig.tight_layout()
        return fig
