"""
Logging system for ADAN Dashboard

Logs trading events, alerts, and system events with structured data.
"""

import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

from .models import Position, ClosedTrade, Alert


logger = logging.getLogger(__name__)


class DashboardLogger:
    """
    Structured logging for dashboard events.
    
    Logs:
    - Trade entries with timestamp, price, size, signal strength
    - Trade exits with timestamp, price, P&L, close reason
    - Alerts with severity and timestamp
    - System startup/shutdown with configuration
    """
    
    def __init__(self, log_dir: str = "logs/dashboard"):
        """
        Initialize dashboard logger.
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        self.setup_logging()
        
        # Event counters
        self.trade_entries = 0
        self.trade_exits = 0
        self.alerts_logged = 0
    
    def setup_logging(self):
        """Setup file logging"""
        log_file = self.log_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    def log_trade_entry(
        self,
        position: Position,
        signal_strength: float,
        market_price: float,
    ) -> None:
        """
        Log trade entry event.
        
        Args:
            position: Position object
            signal_strength: Signal strength at entry (0.0-1.0)
            market_price: Market price at entry
        """
        self.trade_entries += 1
        
        event = {
            'event_type': 'TRADE_ENTRY',
            'timestamp': datetime.now().isoformat(),
            'pair': position.pair,
            'side': position.side,
            'size_btc': position.size_btc,
            'entry_price': position.entry_price,
            'market_price': market_price,
            'stop_loss': position.sl_price,
            'take_profit': position.tp_price,
            'signal_strength': signal_strength,
            'market_regime': position.entry_market_regime,
            'volatility': position.entry_volatility,
            'rsi': position.entry_rsi,
        }
        
        logger.info(f"Trade Entry: {json.dumps(event)}")
    
    def log_trade_exit(
        self,
        trade: ClosedTrade,
        exit_price: float,
    ) -> None:
        """
        Log trade exit event.
        
        Args:
            trade: Closed trade object
            exit_price: Exit price
        """
        self.trade_exits += 1
        
        event = {
            'event_type': 'TRADE_EXIT',
            'timestamp': datetime.now().isoformat(),
            'pair': trade.pair,
            'side': trade.side,
            'size_btc': trade.size_btc,
            'entry_price': trade.entry_price,
            'exit_price': exit_price,
            'realized_pnl_usd': trade.realized_pnl_usd,
            'realized_pnl_pct': trade.realized_pnl_pct,
            'close_reason': trade.close_reason,
            'duration_seconds': trade.duration.total_seconds(),
            'entry_confidence': trade.entry_confidence,
        }
        
        logger.info(f"Trade Exit: {json.dumps(event)}")
    
    def log_alert(
        self,
        alert: Alert,
    ) -> None:
        """
        Log system alert.
        
        Args:
            alert: Alert object
        """
        self.alerts_logged += 1
        
        event = {
            'event_type': 'ALERT',
            'timestamp': alert.timestamp.isoformat(),
            'severity': alert.severity,
            'message': alert.message,
        }
        
        log_level = {
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL,
        }.get(alert.severity, logging.INFO)
        
        logger.log(log_level, f"Alert: {json.dumps(event)}")
    
    def log_system_startup(self, config: Dict[str, Any]) -> None:
        """
        Log system startup.
        
        Args:
            config: System configuration
        """
        event = {
            'event_type': 'SYSTEM_STARTUP',
            'timestamp': datetime.now().isoformat(),
            'config': config,
        }
        
        logger.info(f"System Startup: {json.dumps(event)}")
    
    def log_system_shutdown(self, runtime_seconds: float) -> None:
        """
        Log system shutdown.
        
        Args:
            runtime_seconds: Total runtime in seconds
        """
        event = {
            'event_type': 'SYSTEM_SHUTDOWN',
            'timestamp': datetime.now().isoformat(),
            'runtime_seconds': runtime_seconds,
            'trade_entries': self.trade_entries,
            'trade_exits': self.trade_exits,
            'alerts_logged': self.alerts_logged,
        }
        
        logger.info(f"System Shutdown: {json.dumps(event)}")
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get logging statistics.
        
        Returns:
            Dictionary with logging stats
        """
        return {
            'trade_entries': self.trade_entries,
            'trade_exits': self.trade_exits,
            'alerts_logged': self.alerts_logged,
        }
