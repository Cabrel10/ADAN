"""
Structured log storage for ADAN Dashboard

Stores and retrieves historical trade data and performance metrics.
"""

import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path


class LogStorage:
    """
    JSON-based log storage for dashboard events.
    
    Stores:
    - Trade entries and exits
    - Performance metrics
    - System events
    """
    
    def __init__(self, storage_dir: str = "logs/storage"):
        """
        Initialize log storage.
        
        Args:
            storage_dir: Directory for storage files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.trades_file = self.storage_dir / "trades.jsonl"
        self.metrics_file = self.storage_dir / "metrics.jsonl"
        self.events_file = self.storage_dir / "events.jsonl"
    
    def store_trade_entry(self, trade_data: Dict[str, Any]) -> None:
        """
        Store trade entry event.
        
        Args:
            trade_data: Trade entry data
        """
        self._append_to_file(self.trades_file, {
            'type': 'ENTRY',
            'timestamp': datetime.now().isoformat(),
            **trade_data,
        })
    
    def store_trade_exit(self, trade_data: Dict[str, Any]) -> None:
        """
        Store trade exit event.
        
        Args:
            trade_data: Trade exit data
        """
        self._append_to_file(self.trades_file, {
            'type': 'EXIT',
            'timestamp': datetime.now().isoformat(),
            **trade_data,
        })
    
    def store_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Store performance metrics.
        
        Args:
            metrics: Performance metrics
        """
        self._append_to_file(self.metrics_file, {
            'timestamp': datetime.now().isoformat(),
            **metrics,
        })
    
    def store_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Store system event.
        
        Args:
            event_type: Type of event
            event_data: Event data
        """
        self._append_to_file(self.events_file, {
            'type': event_type,
            'timestamp': datetime.now().isoformat(),
            **event_data,
        })
    
    def get_recent_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent trades.
        
        Args:
            limit: Maximum number of trades to return
        
        Returns:
            List of recent trades
        """
        trades = self._read_file(self.trades_file)
        return trades[-limit:] if trades else []
    
    def get_trades_by_date(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Get trades within date range.
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            List of trades in date range
        """
        trades = self._read_file(self.trades_file)
        
        filtered = []
        for trade in trades:
            trade_time = datetime.fromisoformat(trade['timestamp'])
            if start_date <= trade_time <= end_date:
                filtered.append(trade)
        
        return filtered
    
    def get_recent_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent performance metrics.
        
        Args:
            limit: Maximum number of metrics to return
        
        Returns:
            List of recent metrics
        """
        metrics = self._read_file(self.metrics_file)
        return metrics[-limit:] if metrics else []
    
    def get_metrics_by_date(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Get metrics within date range.
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            List of metrics in date range
        """
        metrics = self._read_file(self.metrics_file)
        
        filtered = []
        for metric in metrics:
            metric_time = datetime.fromisoformat(metric['timestamp'])
            if start_date <= metric_time <= end_date:
                filtered.append(metric)
        
        return filtered
    
    def get_events_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """
        Get events by type.
        
        Args:
            event_type: Type of event to retrieve
        
        Returns:
            List of events of specified type
        """
        events = self._read_file(self.events_file)
        return [e for e in events if e.get('type') == event_type]
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """
        Get trade statistics from stored data.
        
        Returns:
            Dictionary with trade statistics
        """
        trades = self._read_file(self.trades_file)
        
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
            }
        
        # Count exits (completed trades)
        exits = [t for t in trades if t.get('type') == 'EXIT']
        
        if not exits:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
            }
        
        # Calculate statistics
        winning = [t for t in exits if t.get('realized_pnl_usd', 0) > 0]
        losing = [t for t in exits if t.get('realized_pnl_usd', 0) < 0]
        total_pnl = sum(t.get('realized_pnl_usd', 0) for t in exits)
        
        return {
            'total_trades': len(exits),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': len(winning) / len(exits) if exits else 0.0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(exits) if exits else 0.0,
        }
    
    def clear_old_data(self, days: int = 30) -> None:
        """
        Clear data older than specified days.
        
        Args:
            days: Number of days to keep
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for file_path in [self.trades_file, self.metrics_file, self.events_file]:
            if not file_path.exists():
                continue
            
            # Read all data
            data = self._read_file(file_path)
            
            # Filter recent data
            recent = []
            for item in data:
                item_time = datetime.fromisoformat(item['timestamp'])
                if item_time > cutoff_date:
                    recent.append(item)
            
            # Write back
            with open(file_path, 'w') as f:
                for item in recent:
                    f.write(json.dumps(item) + '\n')
    
    def _append_to_file(self, file_path: Path, data: Dict[str, Any]) -> None:
        """
        Append data to JSONL file.
        
        Args:
            file_path: Path to file
            data: Data to append
        """
        with open(file_path, 'a') as f:
            f.write(json.dumps(data) + '\n')
    
    def _read_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Read JSONL file.
        
        Args:
            file_path: Path to file
        
        Returns:
            List of data objects
        """
        if not file_path.exists():
            return []
        
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        
        return data
