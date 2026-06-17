"""Quant Research Engine — a systematic equity research platform."""

from qre.alpha.base import Alpha
from qre.alpha.factors import RSI, BollingerZScore, CrossSectionalMomentum, Momentum
from qre.analytics import (
    compute_ic,
    full_summary,
    ic_decay,
    ic_summary,
    plot_tearsheet,
    summary,
)
from qre.backtest import VectorizedBacktester

__all__ = [
    # Alpha
    "Alpha",
    "BollingerZScore",
    "CrossSectionalMomentum",
    "Momentum",
    "RSI",
    # Analytics
    "compute_ic",
    "full_summary",
    "ic_decay",
    "ic_summary",
    "plot_tearsheet",
    "summary",
    # Backtest
    "VectorizedBacktester",
]
