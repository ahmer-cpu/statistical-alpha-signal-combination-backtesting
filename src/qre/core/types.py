"""Core data types for the Quant Research Engine."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Side(Enum):
    """Order side."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order execution type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(Enum):
    """Lifecycle status of an order."""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class Bar:
    """One OHLCV candle for a single ticker."""

    ticker: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True, slots=True)
class Signal:
    """A strategy's desired target weight for a ticker.

    weight is a fraction of portfolio value (e.g., 0.05 = 5% long, -0.05 = 5% short).
    """

    ticker: str
    timestamp: datetime
    weight: float


@dataclass(frozen=True, slots=True)
class Order:
    """An instruction to buy or sell."""

    ticker: str
    timestamp: datetime
    side: Side
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None


@dataclass(frozen=True, slots=True)
class Fill:
    """Broker confirmation that an order was executed."""

    ticker: str
    timestamp: datetime
    side: Side
    quantity: int
    price: float
    commission: float


@dataclass(slots=True)
class Position:
    """Current holding in a single ticker."""

    ticker: str
    quantity: int
    avg_entry_price: float
    market_price: float = 0.0

    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.quantity * self.market_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return self.quantity * (self.market_price - self.avg_entry_price)

    @property
    def side(self) -> Side | None:
        """Whether this is a long or short position, or None if flat."""
        if self.quantity > 0:
            return Side.BUY
        elif self.quantity < 0:
            return Side.SELL
        return None
