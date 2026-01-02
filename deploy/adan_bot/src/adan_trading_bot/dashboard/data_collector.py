"""
Abstract data collector interface for ADAN Dashboard

Defines the interface for collecting data from various sources
(exchange, ADAN engine, portfolio manager, metrics database).
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from .models import Position, ClosedTrade, Signal, MarketContext, PortfolioState


class DataCollector(ABC):
    """
    Abstract base class for data collection.
    
    Implementations should fetch data from various sources and return
    structured data models for dashboard rendering.
    """
    
    @abstractmethod
    def get_portfolio_state(self) -> PortfolioState:
        """
        Get current portfolio state.
        
        Returns:
            PortfolioState: Current portfolio with all positions, trades, and metrics
        """
        pass
    
    @abstractmethod
    def get_open_positions(self) -> List[Position]:
        """
        Get list of open positions.
        
        Returns:
            List[Position]: List of currently open positions
        """
        pass
    
    @abstractmethod
    def get_closed_trades(self, limit: int = 5) -> List[ClosedTrade]:
        """
        Get list of recently closed trades.
        
        Args:
            limit: Maximum number of trades to return (default: 5)
        
        Returns:
            List[ClosedTrade]: List of closed trades in reverse chronological order
        """
        pass
    
    @abstractmethod
    def get_current_signal(self) -> Optional[Signal]:
        """
        Get current ADAN trading signal.
        
        Returns:
            Optional[Signal]: Current signal or None if not available
        """
        pass
    
    @abstractmethod
    def get_market_context(self) -> Optional[MarketContext]:
        """
        Get current market context and technical indicators.
        
        Returns:
            Optional[MarketContext]: Current market context or None if not available
        """
        pass
    
    @abstractmethod
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get system health metrics.
        
        Returns:
            Dict[str, Any]: System health data including:
                - api_status: bool (connected)
                - api_latency_ms: int
                - feed_status: bool (connected)
                - feed_lag_ms: int
                - model_status: bool (running)
                - model_latency_ms: int
                - db_status: bool (connected)
                - cpu_percent: float
                - memory_gb: float
                - memory_total_gb: float
                - threads: int
                - uptime_percent: float
                - alerts: List[Dict] with 'severity' and 'message'
        """
        pass
    
    @abstractmethod
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """
        Get portfolio performance metrics.
        
        Returns:
            Dict[str, Any]: Portfolio metrics including:
                - total_value_usd: float
                - available_capital_usd: float
                - total_pnl_usd: float
                - total_pnl_pct: float
                - realized_pnl_usd: float
                - unrealized_pnl_usd: float
                - win_rate: float (%)
                - profit_factor: float
                - sharpe_ratio: float
                - sortino_ratio: float
                - max_drawdown_pct: float
                - best_trade_usd: float
                - worst_trade_usd: float
                - avg_holding_time_hours: float
                - total_trades: int
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if data collector is connected to all required sources.
        
        Returns:
            bool: True if connected, False otherwise
        """
        pass
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connections to data sources.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Close connections to data sources.
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        pass
