"""Tear sheet generation for strategy evaluation."""

from __future__ import annotations

from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from qre.analytics.metrics import full_summary, summary

# Website-matched color palette
_COLORS = {
    "bg": "#fbfbfb",
    "panel": "#ffffff",
    "text": "#111111",
    "muted": "#6f6f6f",
    "accent": "#6b1f2b",
    "grid": "#e6e6e6",
    "drawdown": "#6b1f2b",
}

# Metric display configuration grouped by category: (key, label, format)
_PERFORMANCE_METRICS = [
    ("annualized_return", "Ann. Return", "{:+.2%}"),
    ("annualized_volatility", "Ann. Vol", "{:.2%}"),
    ("sharpe_ratio", "Sharpe", "{:.2f}"),
    ("sortino_ratio", "Sortino", "{:.2f}"),
    ("max_drawdown", "Max DD", "{:.2%}"),
    ("calmar_ratio", "Calmar", "{:.2f}"),
    ("win_rate", "Win Rate", "{:.1%}"),
    ("profit_factor", "Profit Factor", "{:.2f}"),
]

_BENCHMARK_METRICS = [
    ("beta", "Beta", "{:.2f}"),
    ("alpha", "Alpha", "{:+.2%}"),
    ("tracking_error", "Track. Error", "{:.2%}"),
    ("information_ratio", "Info Ratio", "{:.2f}"),
    ("up_capture", "Up Capture", "{:.2f}"),
    ("down_capture", "Down Capture", "{:.2f}"),
]

_DISTRIBUTION_METRICS = [
    ("skewness", "Skew", "{:.2f}"),
    ("excess_kurtosis", "Kurtosis", "{:.2f}"),
    ("value_at_risk_95", "VaR 95%", "{:.2%}"),
    ("cvar_95", "CVaR 95%", "{:.2%}"),
    ("tail_ratio", "Tail Ratio", "{:.2f}"),
    ("expected_tail_ratio", "Exp. Tail", "{:.2f}"),
    ("best_day", "Best Day", "{:+.2%}"),
    ("worst_day", "Worst Day", "{:+.2%}"),
]

# Left-margin width reserved for row labels (fraction of x-axis)
_LABEL_MARGIN = 0.10


def _draw_metric_row(
    fig: Any,
    ax: Any,
    metrics_config: list[tuple[str, str, str]],
    stats: dict[str, float],
    y_label: float,
    y_value: float,
    color_cfg: dict[str, str],
    row_label: str | None = None,
    label_x_fig: float | None = None,
) -> None:
    """Draw a single row of metrics on an axis, with optional row label.

    Row label is placed using figure coordinates so it aligns with chart
    y-labels regardless of axes position.
    """
    n = len(metrics_config)

    # Draw row label at fixed figure x-coordinate (matches chart ylabels)
    if row_label and label_x_fig is not None:
        ax_pos = ax.get_position()
        y_mid = (y_label + y_value) / 2
        fig_y = ax_pos.y0 + y_mid * ax_pos.height
        fig.text(
            label_x_fig, fig_y, row_label.upper(),
            ha="center", va="center",
            fontsize=7, color=color_cfg["muted"],
            fontfamily="sans-serif", fontweight="bold",
            alpha=0.6,
        )

    for i, (key, label, fmt) in enumerate(metrics_config):
        x = (i + 0.5) / n
        value = stats[key]
        value_str = fmt.format(value)

        ax.text(
            x, y_label, label.upper(),
            ha="center", va="center",
            fontsize=7.5, color=color_cfg["muted"],
            fontfamily="sans-serif", fontweight="bold",
        )
        ax.text(
            x, y_value, value_str,
            ha="center", va="center",
            fontsize=12, color=color_cfg["text"],
            fontfamily="sans-serif", fontweight="bold",
        )


def plot_tearsheet(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    title: str = "Strategy",
) -> None:
    """Plot full tear sheet with metrics, equity curve, drawdown, and heatmap.

    If benchmark_returns is provided, displays all metrics including
    benchmark-relative analytics. Otherwise shows standalone + distribution.
    """
    c = _COLORS

    has_benchmark = benchmark_returns is not None
    if has_benchmark and benchmark_returns is not None:
        stats = full_summary(returns, benchmark_returns)
    else:
        standalone = summary(returns)
        from qre.analytics.metrics import (
            best_day,
            cvar,
            excess_kurtosis,
            expected_tail_ratio,
            skewness,
            tail_ratio,
            value_at_risk,
            worst_day,
        )
        stats = {
            **standalone,
            "skewness": skewness(returns),
            "excess_kurtosis": excess_kurtosis(returns),
            "value_at_risk_95": value_at_risk(returns, confidence=0.95),
            "cvar_95": cvar(returns, confidence=0.95),
            "tail_ratio": tail_ratio(returns),
            "expected_tail_ratio": expected_tail_ratio(returns),
            "best_day": best_day(returns),
            "worst_day": worst_day(returns),
        }

    # Date range for subtitle
    start = returns.index[0].strftime("%b %Y")
    end = returns.index[-1].strftime("%b %Y")

    # Layout — manual positioning for consistent spacing
    has_3_rows = has_benchmark

    fig = plt.figure(figsize=(13, 17))
    fig.patch.set_facecolor(c["bg"])

    left, right = 0.10, 0.95
    metrics_gap = 0.02  # tighter gap: metrics → first chart
    chart_gap = 0.05   # breathing room between charts

    # Allocate vertical space from top to bottom
    title_y = 0.97
    metrics_top = 0.93
    metrics_height = 0.18 if has_3_rows else 0.13
    metrics_bottom = metrics_top - metrics_height

    # Remaining space split among 3 charts in ratio 3:2:2.8
    charts_top = metrics_bottom - metrics_gap
    charts_bottom = 0.04
    total_chart_space = charts_top - charts_bottom - 2 * chart_gap
    ratios = [3, 2, 2.8]
    ratio_sum = sum(ratios)
    chart_heights = [r / ratio_sum * total_chart_space for r in ratios]

    chart_bottoms = []
    y = charts_top
    for h in chart_heights:
        y -= h
        chart_bottoms.append(y)
        y -= chart_gap

    # Title
    fig.suptitle(
        f"{title.upper()}  ({start} – {end})",
        fontsize=13,
        fontweight="bold",
        color=c["text"],
        fontfamily="sans-serif",
        y=title_y,
    )

    # --- Metrics banner ---
    # x-coordinate in figure space where rotated labels sit (left of chart axes)
    label_x = left - 0.04

    ax_metrics = fig.add_axes((left, metrics_bottom, right - left, metrics_height))
    ax_metrics.set_facecolor(c["bg"])
    ax_metrics.set_xlim(0, 1)
    ax_metrics.set_ylim(0, 1)
    ax_metrics.axis("off")

    if has_benchmark:
        _draw_metric_row(fig, ax_metrics, _PERFORMANCE_METRICS, stats, 0.93, 0.82, c,
                         row_label="Performance", label_x_fig=label_x)
        ax_metrics.axhline(y=0.74, color=c["grid"], linewidth=0.6,
                           xmin=0.02, xmax=0.98)
        _draw_metric_row(fig, ax_metrics, _BENCHMARK_METRICS, stats, 0.65, 0.54, c,
                         row_label="Benchmark", label_x_fig=label_x)
        ax_metrics.axhline(y=0.46, color=c["grid"], linewidth=0.6,
                           xmin=0.02, xmax=0.98)
        _draw_metric_row(fig, ax_metrics, _DISTRIBUTION_METRICS, stats, 0.37, 0.26, c,
                         row_label="Statistical", label_x_fig=label_x)
        ax_metrics.axhline(y=0.16, color=c["grid"], linewidth=0.6,
                           xmin=0.02, xmax=0.98)
    else:
        _draw_metric_row(fig, ax_metrics, _PERFORMANCE_METRICS, stats, 0.90, 0.74, c,
                         row_label="Performance", label_x_fig=label_x)
        ax_metrics.axhline(y=0.58, color=c["grid"], linewidth=0.6,
                           xmin=0.02, xmax=0.98)
        _draw_metric_row(fig, ax_metrics, _DISTRIBUTION_METRICS, stats, 0.45, 0.28, c,
                         row_label="Statistical", label_x_fig=label_x)
        ax_metrics.axhline(y=0.12, color=c["grid"], linewidth=0.6,
                           xmin=0.02, xmax=0.98)

    # --- Chart panels ---
    chart_axes = []
    for i in range(3):
        ax = fig.add_axes((left, chart_bottoms[i], right - left, chart_heights[i]))
        ax.set_facecolor(c["panel"])
        ax.tick_params(colors=c["muted"], labelsize=9)
        if i < 2:
            ax.grid(True, alpha=0.4, color=c["grid"], linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color(c["grid"])
        chart_axes.append(ax)

    # Rotated titles — placed at fixed figure x so they align with metric labels
    for i, label in enumerate(["Equity Curve", "Drawdown"]):
        ax_pos = chart_axes[i].get_position()
        fig_y = ax_pos.y0 + ax_pos.height / 2
        fig.text(
            label_x, fig_y, label,
            ha="center", va="center",
            fontsize=10, color=c["text"],
            fontfamily="sans-serif", fontweight="bold",
            rotation=90,
        )

    # 1. Equity curve
    equity = (1 + returns).cumprod()
    eq_vals = equity.to_numpy()
    chart_axes[0].plot(
        equity.index, eq_vals, linewidth=1.2, color=c["accent"],
    )
    chart_axes[0].fill_between(
        equity.index, 1, eq_vals, alpha=0.06, color=c["accent"],
    )
    chart_axes[0].axhline(y=1, color=c["grid"], linewidth=0.8, linestyle="--")
    chart_axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # type: ignore[no-untyped-call]
    chart_axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))  # type: ignore[no-untyped-call]

    if has_benchmark and benchmark_returns is not None:
        bench_equity = (1 + benchmark_returns).cumprod()
        bench_equity = bench_equity.reindex(equity.index).ffill().bfill()
        chart_axes[0].plot(
            bench_equity.index, bench_equity.to_numpy(),
            linewidth=1.0, color=c["muted"],
            linestyle="--", alpha=0.7, label="Benchmark",
        )
        chart_axes[0].legend(loc="upper left", fontsize=9, framealpha=0.7)

    # 2. Drawdown
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    dd_vals = drawdown.to_numpy()
    chart_axes[1].fill_between(
        drawdown.index, dd_vals, 0, color=c["drawdown"], alpha=0.3,
    )
    chart_axes[1].plot(
        drawdown.index, dd_vals,
        color=c["drawdown"], linewidth=0.6, alpha=0.7,
    )
    chart_axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # type: ignore[no-untyped-call]
    chart_axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))  # type: ignore[no-untyped-call]

    # 3. Monthly returns heatmap
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_index = pd.DatetimeIndex(monthly.index)
    monthly_table = pd.pivot_table(
        pd.DataFrame({
            "year": monthly_index.year,
            "month": monthly_index.month,
            "return": monthly.values,
        }),
        values="return",
        index="year",
        columns="month",
    )

    # Ensure all 12 months are present as columns
    for m in range(1, 13):
        if m not in monthly_table.columns:
            monthly_table[m] = np.nan
    monthly_table = monthly_table[range(1, 13)]

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_table.columns = month_names

    chart_axes[2].grid(False)
    chart_axes[2].set_facecolor(c["bg"])

    # Build annotation labels
    annot_labels = monthly_table.copy()
    for col in annot_labels.columns:
        annot_labels[col] = annot_labels[col].apply(
            lambda v: "" if pd.isna(v) else f"{v:.1%}"
        )

    # NaN cells match background color
    cmap = sns.diverging_palette(10, 130, s=60, l=55, as_cmap=True)
    cmap.set_bad(color=c["bg"])

    sns.heatmap(
        monthly_table,
        ax=chart_axes[2],
        cmap=cmap,
        center=0,
        annot=annot_labels,
        fmt="",
        linewidths=0.5,
        linecolor=c["bg"],
        annot_kws={"fontsize": 8, "color": c["text"]},
        cbar=False,
    )

    hm_pos = chart_axes[2].get_position()
    fig.text(
        label_x, hm_pos.y0 + hm_pos.height / 2, "Monthly Returns",
        ha="center", va="center",
        fontsize=10, color=c["text"],
        fontfamily="sans-serif", fontweight="bold",
        rotation=90,
    )
    chart_axes[2].set_ylabel("")
    chart_axes[2].tick_params(axis="y", rotation=0)

    # Center heatmap relative to charts above
    fig.canvas.draw()
    equity_pos = chart_axes[0].get_position()
    heatmap_pos = chart_axes[2].get_position()

    equity_center = (equity_pos.x0 + equity_pos.x1) / 2
    hm_half_width = heatmap_pos.width / 2

    chart_axes[2].set_position((
        equity_center - hm_half_width,
        heatmap_pos.y0,
        heatmap_pos.width,
        heatmap_pos.height,
    ))

    plt.show()
