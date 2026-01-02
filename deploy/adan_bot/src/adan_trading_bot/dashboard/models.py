"""
Data models for ADAN BTC/USDT Terminal Dashboard

Defines core data structures for positions, trades, signals, and market context.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from enum import Enum


class SignalDirection(Enum):
    """Trading signal direction"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class CloseReason(Enum):
    """Reason for trade closure"""
    TP_HIT = "TP Hit"
    SL_HIT = "SL Hit"
    MANUAL = "Manual"
    TIME_BASED = "Time"


class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING = "Trending"
    RANGING = "Ranging"
    BREAKOUT = "Breakout"


class VolatilityRegime(Enum):
    """Volatility regime classification"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


@dataclass
class Position:
    """
    Represents an open trading position.
    
    Attributes:
        pair: Trading pair (e.g., "BTCUSDT")
        side: Position side ("LONG" or "SHORT")
        size_btc: Position size in BTC (4 decimal places)
        entry_price: Average entry price in USDT
        current_price: Current market price in USDT
        sl_price: Stop Loss price in USDT
        tp_price: Take Profit price in USDT
        open_time: Position open timestamp
        entry_signal_strength: ADAN signal strength at entry (0.0-1.0)
        entry_market_regime: Market regime at entry
        entry_volatility: Volatility (ATR %) at entry
        entry_rsi: RSI value at entry (0-100)
    """
    pair: str
    side: str  # "LONG" or "SHORT"
    size_btc: float
    entry_price: float
    current_price: float
    sl_price: float
    tp_price: float
    open_time: datetime
    entry_signal_strength: float
    entry_market_regime: str
    entry_volatility: float
    entry_rsi: int
    
    @property
    def unrealized_pnl_usd(self) -> float:
        """Calculate unrealized P&L in USD"""
        if self.side == "LONG":
            return (self.current_price - self.entry_price) * self.size_btc
        else:  # SHORT
            return (self.entry_price - self.current_price) * self.size_btc
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L as percentage"""
        if self.entry_price == 0:
            return 0.0
        if self.side == "LONG":
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:  # SHORT
            return ((self.entry_price - self.current_price) / self.entry_price) * 100
    
    @property
    def duration(self) -> timedelta:
        """Calculate position duration"""
        return datetime.now() - self.open_time
    
    @property
    def sl_distance_pct(self) -> float:
        """Calculate SL distance as percentage"""
        if self.entry_price == 0:
            return 0.0
        return abs((self.sl_price - self.entry_price) / self.entry_price) * 100
    
    @property
    def tp_distance_pct(self) -> float:
        """Calculate TP distance as percentage"""
        if self.entry_price == 0:
            return 0.0
        return abs((self.tp_price - self.entry_price) / self.entry_price) * 100
    
    @property
    def position_value_usd(self) -> float:
        """Calculate current position value in USD"""
        return self.size_btc * self.current_price


@dataclass
class ClosedTrade:
    """
    Represents a closed trading position with outcome.
    
    Attributes:
        pair: Trading pair (e.g., "BTCUSDT")
        side: Position side ("LONG" or "SHORT")
        size_btc: Position size in BTC
        entry_price: Entry price in USDT
        exit_price: Exit price in USDT
        open_time: Position open timestamp
        close_time: Position close timestamp
        close_reason: Reason for closure (TP/SL/Manual/Time)
        entry_confidence: ADAN confidence at entry (0.0-1.0)
    """
    pair: str
    side: str  # "LONG" or "SHORT"
    size_btc: float
    entry_price: float
    exit_price: float
    open_time: datetime
    close_time: datetime
    close_reason: str  # "TP Hit", "SL Hit", "Manual", "Time"
    entry_confidence: float
    
    @property
    def realized_pnl_usd(self) -> float:
        """Calculate realized P&L in USD"""
        if self.side == "LONG":
            return (self.exit_price - self.entry_price) * self.size_btc
        else:  # SHORT
            return (self.entry_price - self.exit_price) * self.size_btc
    
    @property
    def realized_pnl_pct(self) -> float:
        """Calculate realized P&L as percentage"""
        if self.entry_price == 0:
            return 0.0
        if self.side == "LONG":
            return ((self.exit_price - self.entry_price) / self.entry_price) * 100
        else:  # SHORT
            return ((self.entry_price - self.exit_price) / self.entry_price) * 100
    
    @property
    def duration(self) -> timedelta:
        """Calculate trade duration"""
        return self.close_time - self.open_time
    
    @property
    def is_win(self) -> bool:
        """Check if trade was profitable"""
        return self.realized_pnl_usd > 0
    
    @property
    def is_breakeven(self) -> bool:
        """Check if trade was breakeven"""
        return abs(self.realized_pnl_usd) < 0.01  # Within 1 cent


@dataclass
class Signal:
    """
    Represents ADAN's current trading signal.
    
    Attributes:
        direction: Signal direction (BUY/SELL/HOLD)
        confidence: Confidence score (0.0-1.0)
        horizon: Trading horizon (5m/1h/4h/1d)
        worker_votes: Individual worker votes {W1: score, W2: score, ...}
        decision_driver: Decision driver (Trend/MeanReversion/Breakout)
        timestamp: Signal generation timestamp
    """
    direction: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0-1.0
    horizon: str  # "5m", "1h", "4h", "1d"
    worker_votes: Dict[str, float]  # {"W1": 0.82, "W2": 0.91, ...}
    decision_driver: str  # "Trend", "MeanReversion", "Breakout"
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate signal data"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        for worker, vote in self.worker_votes.items():
            if not 0.0 <= vote <= 1.0:
                raise ValueError(f"Worker {worker} vote must be between 0.0 and 1.0, got {vote}")
    
    @property
    def average_worker_vote(self) -> float:
        """Calculate average of all worker votes"""
        if not self.worker_votes:
            return 0.0
        return sum(self.worker_votes.values()) / len(self.worker_votes)


@dataclass
class MarketContext:
    """
    Represents current market context and technical indicators.
    
    Attributes:
        price: Current BTC/USDT price
        volatility_atr: Volatility measured by ATR (%)
        rsi: RSI value (0-100)
        adx: ADX value (0-100, trend strength)
        trend_strength: Trend strength classification
        market_regime: Market regime (Trending/Ranging/Breakout)
        volume_change: Volume change vs average (%)
        timestamp: Data timestamp
    """
    price: float
    volatility_atr: float  # Percentage
    rsi: int  # 0-100
    adx: int  # 0-100
    trend_strength: str  # "Weak", "Moderate", "Strong"
    market_regime: str  # "Trending", "Ranging", "Breakout"
    volume_change: float  # Percentage
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate market context data"""
        if not 0 <= self.rsi <= 100:
            raise ValueError(f"RSI must be between 0 and 100, got {self.rsi}")
        if not 0 <= self.adx <= 100:
            raise ValueError(f"ADX must be between 0 and 100, got {self.adx}")
    
    @property
    def rsi_level(self) -> str:
        """Get RSI level classification"""
        if self.rsi < 30:
            return "Oversold"
        elif self.rsi > 70:
            return "Overbought"
        else:
            return "Neutral"


@dataclass
class PortfolioState:
    """
    Represents overall portfolio state.
    
    Attributes:
        total_value_usd: Total portfolio value in USDT
        available_capital_usd: Available capital not engaged in positions
        open_positions: List of open positions
        closed_trades: List of closed trades
        current_signal: Current ADAN signal
        market_context: Current market context
        system_health: System health metrics
        timestamp: State timestamp
    """
    total_value_usd: float
    available_capital_usd: float
    open_positions: List[Position] = field(default_factory=list)
    closed_trades: List[ClosedTrade] = field(default_factory=list)
    current_signal: Optional[Signal] = None
    market_context: Optional[MarketContext] = None
    system_health: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def position_count(self) -> int:
        """Get count of open positions"""
        return len(self.open_positions)
    
    @property
    def total_unrealized_pnl_usd(self) -> float:
        """Calculate total unrealized P&L"""
        return sum(pos.unrealized_pnl_usd for pos in self.open_positions)
    
    @property
    def total_realized_pnl_usd(self) -> float:
        """Calculate total realized P&L from closed trades"""
        return sum(trade.realized_pnl_usd for trade in self.closed_trades)
    
    @property
    def total_pnl_usd(self) -> float:
        """Calculate total P&L (realized + unrealized)"""
        return self.total_realized_pnl_usd + self.total_unrealized_pnl_usd
    
    @property
    def total_pnl_pct(self) -> float:
        """Calculate total P&L as percentage"""
        if self.available_capital_usd == 0:
            return 0.0
        return (self.total_pnl_usd / self.available_capital_usd) * 100
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate from closed trades"""
        if not self.closed_trades:
            return 0.0
        wins = sum(1 for trade in self.closed_trades if trade.is_win)
        return (wins / len(self.closed_trades)) * 100
    
    @property
    def profit_factor(self) -> float:
        """Calculate profit factor"""
        if not self.closed_trades:
            return 0.0
        
        gross_profit = sum(trade.realized_pnl_usd for trade in self.closed_trades if trade.is_win)
        gross_loss = abs(sum(trade.realized_pnl_usd for trade in self.closed_trades if not trade.is_win))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss



@dataclass
class Alert:
    """
    Represents a system alert.
    
    Attributes:
        message: Alert message
        severity: Alert severity level ("INFO", "WARNING", "ERROR", "CRITICAL")
        timestamp: When the alert was created
    """
    message: str
    severity: str = "INFO"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemHealth:
    """
    Represents system health status.
    
    Attributes:
        api_status: API connection status ("OK", "DEGRADED", "DOWN")
        feed_status: Market feed status ("OK", "DEGRADED", "DOWN")
        model_status: ML model status ("OK", "DEGRADED", "DOWN")
        database_status: Database status ("OK", "DEGRADED", "DOWN")
        api_latency_ms: API latency in milliseconds
        feed_latency_ms: Feed latency in milliseconds
        model_latency_ms: Model latency in milliseconds
        cpu_usage_percent: CPU usage percentage (0-100)
        memory_usage_percent: Memory usage percentage (0-100)
        thread_count: Number of active threads
        uptime_seconds: System uptime in seconds
        alerts: List of active alerts
    """
    api_status: str = "OK"
    feed_status: str = "OK"
    model_status: str = "OK"
    database_status: str = "OK"
    api_latency_ms: float = 0.0
    feed_latency_ms: float = 0.0
    model_latency_ms: float = 0.0
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    thread_count: int = 0
    uptime_seconds: float = 0.0
    alerts: List[Alert] = field(default_factory=list)
    
    @property
    def is_healthy(self) -> bool:
        """Check if system is healthy"""
        return all(
            status == "OK"
            for status in [
                self.api_status,
                self.feed_status,
                self.model_status,
                self.database_status,
            ]
        )
    
    @property
    def has_critical_alerts(self) -> bool:
        """Check if there are critical alerts"""
        return any(alert.severity == "CRITICAL" for alert in self.alerts)
    
    @property
    def alert_count(self) -> int:
        """Get number of alerts"""
        return len(self.alerts)
