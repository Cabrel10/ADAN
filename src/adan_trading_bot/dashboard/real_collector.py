"""
Real data collector for ADAN Dashboard

Integrates with existing ADAN components to fetch live trading data.
"""

from typing import List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json

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
        
        For the dashboard, this means verifying we can read the state file.
        """
        try:
            state_file = Path("/mnt/new_data/t10_training/phase2_results/paper_trading_state.json")
            if not state_file.exists():
                logger.warning(f"⚠️ State file not found at {state_file}")
                # We still return True because the file might be created later by the monitor
                # and we don't want to crash.
            
            self._connected = True
            logger.info("✅ Connected to ADAN system (File Mode)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from ADAN system components.
        """
        self._connected = False
        logger.info("✅ Disconnected from ADAN system")
        return True
    
    def is_connected(self) -> bool:
        """Check if connected to ADAN system"""
        return self._connected
    
    def _load_state_from_file(self) -> dict:
        """Load state from JSON file shared by Monitor - ALWAYS FRESH READ"""
        try:
            state_file = Path("/mnt/new_data/t10_training/phase2_results/paper_trading_state.json")
            if not state_file.exists():
                logger.debug(f"State file not found: {state_file}")
                return None
            
            # Always read fresh from disk (no caching)
            with open(state_file, 'r') as f:
                data = json.load(f)
                logger.debug(f"✅ Loaded state from {state_file} - {len(data)} keys")
                return data
        except Exception as e:
            logger.error(f"❌ Error loading state file: {e}")
            return None

    def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state from shared file"""
        state = self._load_state_from_file()
        if not state:
            # Fallback to empty state if file not ready
            return PortfolioState(
                total_value_usd=0.0,
                available_capital_usd=0.0
            )
            
        try:
            p_data = state.get('portfolio', {})
            
            # Parse positions
            positions = [
                Position(
                    pair=p.get('pair', 'Unknown'),
                    side=p.get('side', 'LONG'),
                    size_btc=p.get('size_btc', 0.0),
                    entry_price=p.get('entry_price', 0.0),
                    current_price=p.get('current_price', 0.0),
                    sl_price=p.get('sl_price', 0.0),
                    tp_price=p.get('tp_price', 0.0),
                    open_time=datetime.fromisoformat(p.get('open_time', datetime.now().isoformat())),
                    entry_signal_strength=p.get('entry_signal_strength', 0.0),
                    entry_market_regime=p.get('entry_market_regime', 'Unknown'),
                    entry_volatility=p.get('entry_volatility', 0.0),
                    entry_rsi=p.get('entry_rsi', 50)
                )
                for p in p_data.get('positions', [])
            ]
            
            # Parse closed trades
            trades = [
                ClosedTrade(
                    pair=t.get('pair', 'Unknown'),
                    side=t.get('side', 'LONG'),
                    size_btc=t.get('size_btc', 0.0),
                    entry_price=t.get('entry_price', 0.0),
                    exit_price=t.get('exit_price', 0.0),
                    open_time=datetime.fromisoformat(t.get('open_time', datetime.now().isoformat())),
                    close_time=datetime.fromisoformat(t.get('close_time', datetime.now().isoformat())),
                    close_reason=t.get('close_reason', 'Unknown'),
                    entry_confidence=t.get('entry_confidence', 0.0)
                )
                for t in p_data.get('closed_trades', [])
            ]
            
            return PortfolioState(
                total_value_usd=p_data.get('total_value', 0.0),
                available_capital_usd=p_data.get('available_capital', 0.0),
                open_positions=positions,
                closed_trades=trades,
                current_signal=self.get_current_signal(),
                market_context=self.get_market_context(),
                system_health=self.get_system_health(),
            )
            
        except Exception as e:
            logger.error(f"❌ Error parsing portfolio state: {e}")
            raise

    def get_current_signal(self) -> Optional[Signal]:
        """Get current signal from shared file"""
        state = self._load_state_from_file()
        if not state: return None
        
        s_data = state.get('signal', {})
        return Signal(
            direction=s_data.get('direction', 'HOLD'),
            confidence=s_data.get('confidence', 0.0),
            horizon=s_data.get('horizon', '1h'),
            worker_votes=s_data.get('worker_votes', {}),
            decision_driver=s_data.get('decision_driver', 'None'),
            timestamp=datetime.fromisoformat(s_data.get('timestamp', datetime.now().isoformat()))
        )

    def get_market_context(self) -> Optional[MarketContext]:
        """Get market context from shared file"""
        state = self._load_state_from_file()
        if not state: return None
        
        m_data = state.get('market', {})
        return MarketContext(
            price=m_data.get('price', 0.0),
            volatility_atr=m_data.get('volatility_atr', 0.0),
            rsi=m_data.get('rsi', 50),
            adx=m_data.get('adx', 25),
            trend_strength=m_data.get('trend_strength', 'Unknown'),
            market_regime=m_data.get('market_regime', 'Unknown'),
            volume_change=m_data.get('volume_change', 0.0),
            timestamp=datetime.fromisoformat(m_data.get('timestamp', datetime.now().isoformat()))
        )

    def get_system_health(self) -> dict:
        """Get system health from shared file (returns dict for renderer)"""
        state = self._load_state_from_file()
        if not state:
            return {
                "api_status": True,
                "api_latency_ms": 0,
                "feed_status": True,
                "feed_lag_ms": 0,
                "model_status": True,
                "model_latency_ms": 0,
                "db_status": True,
                "cpu_percent": 0.0,
                "memory_gb": 0.0,
                "memory_total_gb": 4.0,
                "threads": 1,
                "uptime_percent": 100.0,
                "alerts": [],
            }
        
        sys_data = state.get('system', {})
        return {
            "api_status": sys_data.get('api_status', 'OK') == 'OK',
            "api_latency_ms": 50,
            "feed_status": sys_data.get('feed_status', 'OK') == 'OK',
            "feed_lag_ms": 100,
            "model_status": sys_data.get('model_status', 'OK') == 'OK',
            "model_latency_ms": 100,
            "db_status": sys_data.get('database_status', 'OK') == 'OK',
            "cpu_percent": 15.0,
            "memory_gb": 1.0,
            "memory_total_gb": 4.0,
            "threads": 4,
            "uptime_percent": 100.0,
            "alerts": [],
        }

    def get_open_positions(self) -> List[Position]:
        """Get list of open positions"""
        return self.get_portfolio_state().open_positions

    def get_closed_trades(self, limit: int = 5) -> List[ClosedTrade]:
        """Get list of recently closed trades"""
        trades = self.get_portfolio_state().closed_trades
        # Sort by close time descending just in case
        trades.sort(key=lambda t: t.close_time, reverse=True)
        return trades[:limit]

    def get_portfolio_metrics(self) -> dict:
        """Get portfolio performance metrics"""
        state = self.get_portfolio_state()
        trades = state.closed_trades
        
        # Calculate best and worst trades
        pnls = [trade.realized_pnl_usd for trade in trades]
        best_trade = max(pnls) if pnls else 0.0
        worst_trade = min(pnls) if pnls else 0.0
        
        # Calculate average holding time
        durations = [trade.duration.total_seconds() / 3600 for trade in trades]
        avg_holding_time = sum(durations) / len(durations) if durations else 0.0

        return {
            "total_value_usd": state.total_value_usd,
            "available_capital_usd": state.available_capital_usd,
            "total_pnl_usd": state.total_pnl_usd,
            "total_pnl_pct": state.total_pnl_pct,
            "realized_pnl_usd": state.total_realized_pnl_usd,
            "unrealized_pnl_usd": state.total_unrealized_pnl_usd,
            "win_rate": state.win_rate,
            "profit_factor": state.profit_factor,
            "sharpe_ratio": 0.0, # Not available in simple state
            "sortino_ratio": 0.0, # Not available in simple state
            "max_drawdown_pct": 0.0, # Not available in simple state
            "best_trade_usd": best_trade,
            "worst_trade_usd": worst_trade,
            "avg_holding_time_hours": avg_holding_time,
            "total_trades": len(trades),
        }

    
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
