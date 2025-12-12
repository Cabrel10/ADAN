"""
Real data collector for ADAN Dashboard

Integrates with existing ADAN components to fetch live trading data.
"""

from typing import List, Optional
from datetime import datetime, timedelta
import logging

from .data_collector import DataCollector
from .models import (
    Position,
    ClosedTrade,
    Signal,
    MarketContext,
    PortfolioState,
    SystemHealth,
    Alert,
)


logger = logging.getLogger(__name__)


class RealDataCollector(DataCollector):
    """
    Real data collector that integrates with ADAN system components.
    
    Fetches data from:
    - Portfolio manager for positions and capital
    - Metrics database for closed trades
    - ADAN engine for current signal
    - Exchange API for market context
    - System monitors for health status
    """
    
    def __init__(self):
        """Initialize real data collector"""
        super().__init__()
        self.portfolio_manager = None
        self.metrics_db = None
        self.adan_engine = None
        self.exchange_api = None
        self.system_monitor = None
        self._connected = False
    
    def connect(self) -> bool:
        """
        Connect to ADAN system components.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Import ADAN components
            from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager
            from adan_trading_bot.metrics.metrics_db import MetricsDatabase
            from adan_trading_bot.engine.adan_engine import AdanEngine
            from adan_trading_bot.exchange.exchange_api import ExchangeAPI
            from adan_trading_bot.system.system_monitor import SystemMonitor
            
            # Initialize components
            self.portfolio_manager = PortfolioManager()
            self.metrics_db = MetricsDatabase()
            self.adan_engine = AdanEngine()
            self.exchange_api = ExchangeAPI()
            self.system_monitor = SystemMonitor()
            
            self._connected = True
            logger.info("✅ Connected to ADAN system components")
            return True
            
        except ImportError as e:
            logger.warning(f"⚠️  Could not import ADAN components: {e}")
            logger.warning("📊 Falling back to mock data")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to ADAN system: {e}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from ADAN system components.
        
        Returns:
            True if disconnection successful
        """
        try:
            # Cleanup connections
            if self.portfolio_manager:
                self.portfolio_manager.close()
            if self.metrics_db:
                self.metrics_db.close()
            if self.exchange_api:
                self.exchange_api.close()
            
            self._connected = False
            logger.info("✅ Disconnected from ADAN system")
            return True
        except Exception as e:
            logger.error(f"❌ Error disconnecting: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to ADAN system"""
        return self._connected
    
    def get_portfolio_state(self) -> PortfolioState:
        """
        Get current portfolio state from portfolio manager.
        
        Returns:
            PortfolioState with current portfolio data
        """
        if not self._connected or not self.portfolio_manager:
            raise RuntimeError("Not connected to ADAN system")
        
        try:
            # Get portfolio data
            portfolio = self.portfolio_manager.get_portfolio()
            
            # Extract positions
            positions = [
                Position(
                    pair=pos.pair,
                    side=pos.side,
                    size_btc=pos.size_btc,
                    entry_price=pos.entry_price,
                    current_price=pos.current_price,
                    sl_price=pos.sl_price,
                    tp_price=pos.tp_price,
                    open_time=pos.open_time,
                    entry_signal_strength=pos.entry_signal_strength,
                    entry_market_regime=pos.entry_market_regime,
                    entry_volatility=pos.entry_volatility,
                    entry_rsi=pos.entry_rsi,
                )
                for pos in portfolio.positions
            ]
            
            # Get closed trades
            trades = self._get_closed_trades(limit=100)
            
            # Calculate metrics
            win_rate = self._calculate_win_rate(trades)
            profit_factor = self._calculate_profit_factor(trades)
            
            return PortfolioState(
                total_value_usd=portfolio.total_value,
                available_capital_usd=portfolio.available_capital,
                open_positions=positions,
                closed_trades=trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                current_signal=self.get_current_signal(),
                market_context=self.get_market_context(),
                system_health=self.get_system_health(),
            )
            
        except Exception as e:
            logger.error(f"❌ Error getting portfolio state: {e}")
            raise
    
    def get_open_positions(self) -> List[Position]:
        """
        Get list of open positions from portfolio manager.
        
        Returns:
            List of Position objects
        """
        if not self._connected or not self.portfolio_manager:
            raise RuntimeError("Not connected to ADAN system")
        
        try:
            portfolio = self.portfolio_manager.get_portfolio()
            
            positions = [
                Position(
                    pair=pos.pair,
                    side=pos.side,
                    size_btc=pos.size_btc,
                    entry_price=pos.entry_price,
                    current_price=pos.current_price,
                    sl_price=pos.sl_price,
                    tp_price=pos.tp_price,
                    open_time=pos.open_time,
                    entry_signal_strength=pos.entry_signal_strength,
                    entry_market_regime=pos.entry_market_regime,
                    entry_volatility=pos.entry_volatility,
                    entry_rsi=pos.entry_rsi,
                )
                for pos in portfolio.positions
            ]
            
            return positions
            
        except Exception as e:
            logger.error(f"❌ Error getting open positions: {e}")
            raise
    
    def get_closed_trades(self, limit: int = 5) -> List[ClosedTrade]:
        """
        Get closed trades from metrics database.
        
        Args:
            limit: Maximum number of trades to return
        
        Returns:
            List of ClosedTrade objects
        """
        return self._get_closed_trades(limit=limit)
    
    def _get_closed_trades(self, limit: int = 5) -> List[ClosedTrade]:
        """Internal method to get closed trades"""
        if not self._connected or not self.metrics_db:
            raise RuntimeError("Not connected to ADAN system")
        
        try:
            trades = self.metrics_db.get_recent_trades(limit=limit)
            
            closed_trades = [
                ClosedTrade(
                    pair=trade.pair,
                    side=trade.side,
                    size_btc=trade.size_btc,
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price,
                    open_time=trade.open_time,
                    close_time=trade.close_time,
                    close_reason=trade.close_reason,
                    entry_confidence=trade.entry_confidence,
                )
                for trade in trades
            ]
            
            return closed_trades
            
        except Exception as e:
            logger.error(f"❌ Error getting closed trades: {e}")
            raise
    
    def get_current_signal(self) -> Optional[Signal]:
        """
        Get current trading signal from ADAN engine.
        
        Returns:
            Current Signal or None if not available
        """
        if not self._connected or not self.adan_engine:
            return None
        
        try:
            signal_data = self.adan_engine.get_current_signal()
            
            if signal_data is None:
                return None
            
            return Signal(
                direction=signal_data.direction,
                confidence=signal_data.confidence,
                horizon=signal_data.horizon,
                worker_votes=signal_data.worker_votes,
                decision_driver=signal_data.decision_driver,
                timestamp=signal_data.timestamp,
            )
            
        except Exception as e:
            logger.warning(f"⚠️  Error getting current signal: {e}")
            return None
    
    def get_market_context(self) -> Optional[MarketContext]:
        """
        Get market context from exchange API.
        
        Returns:
            Current MarketContext or None if not available
        """
        if not self._connected or not self.exchange_api:
            return None
        
        try:
            market_data = self.exchange_api.get_market_context()
            
            if market_data is None:
                return None
            
            return MarketContext(
                price=market_data.price,
                volatility=market_data.volatility,
                rsi=market_data.rsi,
                adx=market_data.adx,
                trend_strength=market_data.trend_strength,
                market_regime=market_data.market_regime,
                volume_change=market_data.volume_change,
                timestamp=market_data.timestamp,
            )
            
        except Exception as e:
            logger.warning(f"⚠️  Error getting market context: {e}")
            return None
    
    def get_portfolio_metrics(self) -> dict:
        """
        Get portfolio metrics.
        
        Returns:
            Dictionary with portfolio metrics
        """
        try:
            trades = self._get_closed_trades(limit=100)
            
            return {
                'win_rate': self._calculate_win_rate(trades),
                'profit_factor': self._calculate_profit_factor(trades),
                'total_trades': len(trades),
            }
        except Exception as e:
            logger.warning(f"⚠️  Error getting portfolio metrics: {e}")
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0,
            }
    
    def get_system_health(self) -> SystemHealth:
        """
        Get system health status from system monitor.
        
        Returns:
            SystemHealth with current system status
        """
        if not self._connected or not self.system_monitor:
            # Return default healthy status
            return SystemHealth()
        
        try:
            health_data = self.system_monitor.get_health_status()
            
            # Extract alerts
            alerts = [
                Alert(
                    message=alert.message,
                    severity=alert.severity,
                    timestamp=alert.timestamp,
                )
                for alert in health_data.alerts
            ]
            
            return SystemHealth(
                api_status=health_data.api_status,
                feed_status=health_data.feed_status,
                model_status=health_data.model_status,
                database_status=health_data.database_status,
                api_latency_ms=health_data.api_latency_ms,
                feed_latency_ms=health_data.feed_latency_ms,
                model_latency_ms=health_data.model_latency_ms,
                cpu_usage_percent=health_data.cpu_usage_percent,
                memory_usage_percent=health_data.memory_usage_percent,
                thread_count=health_data.thread_count,
                uptime_seconds=health_data.uptime_seconds,
                alerts=alerts,
            )
            
        except Exception as e:
            logger.warning(f"⚠️  Error getting system health: {e}")
            # Return default status on error
            return SystemHealth()
    
    def _calculate_win_rate(self, trades: List[ClosedTrade]) -> float:
        """Calculate win rate from trades"""
        if not trades:
            return 0.0
        
        winning_trades = sum(1 for trade in trades if trade.realized_pnl_usd > 0)
        return winning_trades / len(trades)
    
    def _calculate_profit_factor(self, trades: List[ClosedTrade]) -> float:
        """Calculate profit factor from trades"""
        if not trades:
            return 0.0
        
        gross_profit = sum(trade.realized_pnl_usd for trade in trades if trade.realized_pnl_usd > 0)
        gross_loss = abs(sum(trade.realized_pnl_usd for trade in trades if trade.realized_pnl_usd < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
