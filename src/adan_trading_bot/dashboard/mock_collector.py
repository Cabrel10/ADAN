"""
Mock data collector for testing and development

Generates realistic test data for dashboard development and testing.
"""

import random
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from .data_collector import DataCollector
from .models import Position, ClosedTrade, Signal, MarketContext, PortfolioState


class MockDataCollector(DataCollector):
    """
    Mock data collector that generates realistic test data.
    
    Useful for:
    - Dashboard development without live data
    - Testing dashboard rendering
    - Performance testing
    - UI/UX testing
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize mock data collector.
        
        Args:
            seed: Random seed for reproducible data (optional)
        """
        if seed is not None:
            random.seed(seed)
        
        self.connected = False
        self.start_time = datetime.now()
        self._generate_initial_data()
    
    def _generate_initial_data(self):
        """Generate initial mock data"""
        self.positions = self._generate_positions()
        self.trades = self._generate_closed_trades()
        self.signal = self._generate_signal()
        self.market_context = self._generate_market_context()
    
    def _generate_positions(self, count: int = 2) -> List[Position]:
        """Generate random open positions"""
        positions = []
        base_price = 43000.0
        
        for i in range(count):
            entry_price = base_price + random.uniform(-2000, 2000)
            current_price = entry_price + random.uniform(-500, 500)
            
            pos = Position(
                pair="BTCUSDT",
                side=random.choice(["LONG", "SHORT"]),
                size_btc=round(random.uniform(0.01, 0.1), 4),
                entry_price=round(entry_price, 2),
                current_price=round(current_price, 2),
                sl_price=round(entry_price * (0.97 if random.random() > 0.5 else 1.03), 2),
                tp_price=round(entry_price * (1.03 if random.random() > 0.5 else 0.97), 2),
                open_time=datetime.now() - timedelta(hours=random.randint(1, 24)),
                entry_signal_strength=round(random.uniform(0.7, 0.95), 2),
                entry_market_regime=random.choice(["Trending", "Ranging", "Breakout"]),
                entry_volatility=round(random.uniform(1.0, 3.5), 2),
                entry_rsi=random.randint(20, 80),
            )
            positions.append(pos)
        
        return positions
    
    def _generate_closed_trades(self, count: int = 5) -> List[ClosedTrade]:
        """Generate random closed trades"""
        trades = []
        base_price = 43000.0
        now = datetime.now()
        
        for i in range(count):
            entry_price = base_price + random.uniform(-2000, 2000)
            # 60% win rate
            is_win = random.random() < 0.6
            exit_price = entry_price * (1 + random.uniform(0.005, 0.03)) if is_win else entry_price * (1 - random.uniform(0.005, 0.03))
            
            open_time = now - timedelta(hours=random.randint(1, 72))
            close_time = open_time + timedelta(hours=random.randint(1, 24))
            
            trade = ClosedTrade(
                pair="BTCUSDT",
                side=random.choice(["LONG", "SHORT"]),
                size_btc=round(random.uniform(0.01, 0.1), 4),
                entry_price=round(entry_price, 2),
                exit_price=round(exit_price, 2),
                open_time=open_time,
                close_time=close_time,
                close_reason=random.choice(["TP Hit", "SL Hit", "Manual", "Time"]),
                entry_confidence=round(random.uniform(0.65, 0.95), 2),
            )
            trades.append(trade)
        
        # Sort by close time descending
        trades.sort(key=lambda t: t.close_time, reverse=True)
        return trades
    
    def _generate_signal(self) -> Signal:
        """Generate random ADAN signal"""
        return Signal(
            direction=random.choice(["BUY", "SELL", "HOLD"]),
            confidence=round(random.uniform(0.65, 0.95), 2),
            horizon=random.choice(["5m", "1h", "4h", "1d"]),
            worker_votes={
                "W1": round(random.uniform(0.6, 0.95), 2),
                "W2": round(random.uniform(0.6, 0.95), 2),
                "W3": round(random.uniform(0.6, 0.95), 2),
                "W4": round(random.uniform(0.6, 0.95), 2),
            },
            decision_driver=random.choice(["Trend", "MeanReversion", "Breakout"]),
        )
    
    def _generate_market_context(self) -> MarketContext:
        """Generate random market context"""
        return MarketContext(
            price=round(random.uniform(42000, 45000), 2),
            volatility_atr=round(random.uniform(1.0, 3.5), 2),
            rsi=random.randint(20, 80),
            adx=random.randint(15, 50),
            trend_strength=random.choice(["Weak", "Moderate", "Strong"]),
            market_regime=random.choice(["Trending", "Ranging", "Breakout"]),
            volume_change=round(random.uniform(-30, 50), 1),
        )
    
    def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state"""
        total_value = 1000 + sum(pos.position_value_usd for pos in self.positions)
        available_capital = 1000 - sum(pos.position_value_usd for pos in self.positions)
        
        return PortfolioState(
            total_value_usd=round(total_value, 2),
            available_capital_usd=round(max(0, available_capital), 2),
            open_positions=self.positions,
            closed_trades=self.trades,
            current_signal=self.signal,
            market_context=self.market_context,
            system_health=self.get_system_health(),
        )
    
    def get_open_positions(self) -> List[Position]:
        """Get list of open positions"""
        return self.positions
    
    def get_closed_trades(self, limit: int = 5) -> List[ClosedTrade]:
        """Get list of recently closed trades"""
        return self.trades[:limit]
    
    def get_current_signal(self) -> Optional[Signal]:
        """Get current ADAN trading signal"""
        return self.signal
    
    def get_market_context(self) -> Optional[MarketContext]:
        """Get current market context"""
        return self.market_context
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics"""
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        uptime_percent = min(100.0, (uptime_seconds / 86400) * 100)  # Assume 24h max
        
        return {
            "api_status": True,
            "api_latency_ms": random.randint(10, 100),
            "feed_status": True,
            "feed_lag_ms": random.randint(50, 500),
            "model_status": True,
            "model_latency_ms": random.randint(50, 200),
            "db_status": True,
            "cpu_percent": round(random.uniform(10, 60), 1),
            "memory_gb": round(random.uniform(0.5, 2.0), 2),
            "memory_total_gb": 4.0,
            "threads": random.randint(8, 16),
            "uptime_percent": round(uptime_percent, 1),
            "alerts": self._generate_alerts(),
        }
    
    def _generate_alerts(self) -> List[Dict[str, str]]:
        """Generate random alerts"""
        possible_alerts = [
            {"severity": "INFO", "message": "New signal generated"},
            {"severity": "WARNING", "message": "High volatility detected (3.2%)"},
            {"severity": "WARNING", "message": "Spread widening (0.05% → 0.08%)"},
            {"severity": "WARNING", "message": "Model confidence dropping (0.91 → 0.84)"},
            {"severity": "INFO", "message": "Position opened"},
            {"severity": "INFO", "message": "Position closed"},
        ]
        
        # 50% chance of having alerts
        if random.random() < 0.5:
            return random.sample(possible_alerts, k=random.randint(1, 3))
        return []
    
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get portfolio performance metrics"""
        if not self.trades:
            return {
                "total_value_usd": 1000.0,
                "available_capital_usd": 1000.0,
                "total_pnl_usd": 0.0,
                "total_pnl_pct": 0.0,
                "realized_pnl_usd": 0.0,
                "unrealized_pnl_usd": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "best_trade_usd": 0.0,
                "worst_trade_usd": 0.0,
                "avg_holding_time_hours": 0.0,
                "total_trades": 0,
            }
        
        portfolio = self.get_portfolio_state()
        
        # Calculate best and worst trades
        pnls = [trade.realized_pnl_usd for trade in self.trades]
        best_trade = max(pnls) if pnls else 0.0
        worst_trade = min(pnls) if pnls else 0.0
        
        # Calculate average holding time
        durations = [trade.duration.total_seconds() / 3600 for trade in self.trades]
        avg_holding_time = sum(durations) / len(durations) if durations else 0.0
        
        return {
            "total_value_usd": portfolio.total_value_usd,
            "available_capital_usd": portfolio.available_capital_usd,
            "total_pnl_usd": portfolio.total_pnl_usd,
            "total_pnl_pct": portfolio.total_pnl_pct,
            "realized_pnl_usd": portfolio.total_realized_pnl_usd,
            "unrealized_pnl_usd": portfolio.total_unrealized_pnl_usd,
            "win_rate": portfolio.win_rate,
            "profit_factor": portfolio.profit_factor,
            "sharpe_ratio": round(random.uniform(1.5, 3.5), 2),
            "sortino_ratio": round(random.uniform(2.0, 4.5), 2),
            "max_drawdown_pct": round(random.uniform(1.0, 5.0), 2),
            "best_trade_usd": best_trade,
            "worst_trade_usd": worst_trade,
            "avg_holding_time_hours": round(avg_holding_time, 2),
            "total_trades": len(self.trades),
        }
    
    def is_connected(self) -> bool:
        """Check if data collector is connected"""
        return self.connected
    
    def connect(self) -> bool:
        """Establish connections"""
        self.connected = True
        return True
    
    def disconnect(self) -> bool:
        """Close connections"""
        self.connected = False
        return True
