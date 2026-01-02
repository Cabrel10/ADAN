"""
Data aggregator for ADAN Dashboard

Combines data from multiple sources into unified dashboard data structure.
"""

from typing import List, Optional
from datetime import datetime
import logging

from .data_collector import DataCollector
from .models import (
    Position,
    ClosedTrade,
    Signal,
    MarketContext,
    PortfolioState,
    SystemHealth,
)


logger = logging.getLogger(__name__)


class DataAggregator:
    """
    Aggregates data from multiple sources into unified dashboard state.
    
    Responsibilities:
    - Fetch data from collector
    - Enrich position data with market context
    - Calculate performance metrics
    - Combine all data into PortfolioState
    """
    
    def __init__(self, data_collector: DataCollector):
        """
        Initialize data aggregator.
        
        Args:
            data_collector: Data collector instance
        """
        self.collector = data_collector
    
    def aggregate(self) -> PortfolioState:
        """
        Aggregate all data into unified portfolio state.
        
        Returns:
            Complete PortfolioState with all data
        """
        try:
            # Fetch all data in parallel (conceptually)
            portfolio = self.collector.get_portfolio_state()
            positions = self.collector.get_open_positions()
            trades = self.collector.get_closed_trades(limit=100)
            signal = self.collector.get_current_signal()
            market_context = self.collector.get_market_context()
            health = self.collector.get_system_health()
            
            # Enrich positions with market context
            enriched_positions = self._enrich_positions(positions, market_context)
            
            # Calculate metrics
            metrics = self._calculate_metrics(trades)
            
            # Create aggregated state
            aggregated = PortfolioState(
                total_value_usd=portfolio.total_value_usd,
                available_capital_usd=portfolio.available_capital_usd,
                open_positions=enriched_positions,
                closed_trades=trades,
                current_signal=signal,
                market_context=market_context,
                system_health=health,
            )
            
            logger.debug(f"✅ Aggregated data: {len(enriched_positions)} positions, {len(trades)} trades")
            
            return aggregated
            
        except Exception as e:
            logger.error(f"❌ Error aggregating data: {e}")
            raise
    
    def _enrich_positions(
        self,
        positions: List[Position],
        market_context: Optional[MarketContext],
    ) -> List[Position]:
        """
        Enrich positions with market context.
        
        Args:
            positions: List of positions
            market_context: Current market context
        
        Returns:
            Enriched positions (market context is informational only)
        """
        # Positions are already complete, market context is just for reference
        return positions
    
    def _calculate_metrics(self, trades: List[ClosedTrade]) -> dict:
        """
        Calculate performance metrics from trades.
        
        Args:
            trades: List of closed trades
        
        Returns:
            Dictionary with calculated metrics
        """
        if not trades:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0,
                'best_trade_pnl': 0.0,
                'worst_trade_pnl': 0.0,
                'avg_holding_time_seconds': 0.0,
            }
        
        # Calculate win rate
        winning_trades = [t for t in trades if t.realized_pnl_usd > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0.0
        
        # Calculate profit factor
        gross_profit = sum(t.realized_pnl_usd for t in trades if t.realized_pnl_usd > 0)
        gross_loss = abs(sum(t.realized_pnl_usd for t in trades if t.realized_pnl_usd < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)
        
        # Find best and worst trades
        best_trade_pnl = max((t.realized_pnl_usd for t in trades), default=0.0)
        worst_trade_pnl = min((t.realized_pnl_usd for t in trades), default=0.0)
        
        # Calculate average holding time
        total_holding_time = 0.0
        for trade in trades:
            if trade.close_time and trade.open_time:
                duration = (trade.close_time - trade.open_time).total_seconds()
                total_holding_time += duration
        
        avg_holding_time = total_holding_time / len(trades) if trades else 0.0
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'best_trade_pnl': best_trade_pnl,
            'worst_trade_pnl': worst_trade_pnl,
            'avg_holding_time_seconds': avg_holding_time,
        }
    
    def get_portfolio_summary(self) -> dict:
        """
        Get high-level portfolio summary.
        
        Returns:
            Dictionary with portfolio summary
        """
        try:
            portfolio = self.collector.get_portfolio_state()
            positions = self.collector.get_open_positions()
            trades = self.collector.get_closed_trades(limit=100)
            
            metrics = self._calculate_metrics(trades)
            
            return {
                'total_value': portfolio.total_value_usd,
                'available_capital': portfolio.available_capital_usd,
                'position_count': len(positions),
                'trade_count': len(trades),
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'best_trade': metrics['best_trade_pnl'],
                'worst_trade': metrics['worst_trade_pnl'],
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting portfolio summary: {e}")
            raise
    
    def get_position_summary(self) -> dict:
        """
        Get summary of open positions.
        
        Returns:
            Dictionary with position summary
        """
        try:
            positions = self.collector.get_open_positions()
            
            if not positions:
                return {
                    'count': 0,
                    'total_size': 0.0,
                    'total_pnl': 0.0,
                    'avg_pnl_percent': 0.0,
                }
            
            total_size = sum(p.size_btc for p in positions)
            total_pnl = sum(p.unrealized_pnl_usd for p in positions)
            avg_pnl_percent = sum(p.unrealized_pnl_pct for p in positions) / len(positions)
            
            return {
                'count': len(positions),
                'total_size': total_size,
                'total_pnl': total_pnl,
                'avg_pnl_percent': avg_pnl_percent,
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting position summary: {e}")
            raise
    
    def get_trade_summary(self) -> dict:
        """
        Get summary of recent trades.
        
        Returns:
            Dictionary with trade summary
        """
        try:
            trades = self.collector.get_closed_trades(limit=100)
            
            if not trades:
                return {
                    'count': 0,
                    'total_pnl': 0.0,
                    'win_count': 0,
                    'loss_count': 0,
                }
            
            total_pnl = sum(t.realized_pnl_usd for t in trades)
            win_count = sum(1 for t in trades if t.realized_pnl_usd > 0)
            loss_count = sum(1 for t in trades if t.realized_pnl_usd < 0)
            
            return {
                'count': len(trades),
                'total_pnl': total_pnl,
                'win_count': win_count,
                'loss_count': loss_count,
                'win_rate': win_count / len(trades) if trades else 0.0,
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting trade summary: {e}")
            raise
