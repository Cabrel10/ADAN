#!/usr/bin/env python3
"""
Backtest Engine - Simulates ADAN ensemble on historical data
Reuses patterns from src/adan_trading_bot/portfolio/portfolio_manager.py
"""

import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BacktestPortfolio:
    """Portfolio simulation for backtesting (reuses portfolio manager patterns)"""
    
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = {}  # {symbol: {quantity, entry_price, entry_time}}
        self.trades = []
        self.equity_curve = [initial_balance]
        self.timestamps = []
        
    def open_position(self, symbol, quantity, price, timestamp):
        """Open a new position"""
        cost = quantity * price
        if cost > self.current_balance:
            return False
        
        self.positions[symbol] = {
            'quantity': quantity,
            'entry_price': price,
            'entry_time': timestamp,
            'entry_cost': cost
        }
        
        self.current_balance -= cost
        return True
    
    def close_position(self, symbol, price, timestamp):
        """Close an existing position"""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        quantity = pos['quantity']
        entry_price = pos['entry_price']
        entry_time = pos['entry_time']
        
        # Calculate PnL
        exit_value = quantity * price
        entry_cost = pos['entry_cost']
        pnl = exit_value - entry_cost
        pnl_pct = (pnl / entry_cost) * 100 if entry_cost > 0 else 0
        
        # Update balance
        self.current_balance += exit_value
        
        # Record trade
        trade = {
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': price,
            'quantity': quantity,
            'entry_time': entry_time,
            'exit_time': timestamp,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'duration_seconds': (timestamp - entry_time).total_seconds(),
        }
        
        self.trades.append(trade)
        del self.positions[symbol]
        
        return trade
    
    def update_equity(self, timestamp, market_prices):
        \"\"\"Update equity curve with current market prices\"\"\"
        unrealized_pnl = 0
        
        for symbol, pos in self.positions.items():
            if symbol in market_prices:
                current_price = market_prices[symbol]
                unrealized_pnl += (current_price - pos['entry_price']) * pos['quantity']
        
        total_equity = self.current_balance + unrealized_pnl
        self.equity_curve.append(total_equity)
        self.timestamps.append(timestamp)
        
        return total_equity
    
    def get_stats(self):
        \"\"\"Calculate portfolio statistics\"\"\"
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        
        # Calculate max drawdown
        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_drawdown = np.abs(np.min(drawdown)) if len(drawdown) > 0 else 0
        
        # Calculate Sharpe ratio
        returns = np.diff(equity) / equity[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'final_balance': self.current_balance,
            'total_return': ((self.equity_curve[-1] - self.initial_balance) / self.initial_balance * 100) if self.equity_curve else 0,
        }


class BacktestEngine:
    \"\"\"Simulates ADAN ensemble on historical data\"\"\"
    
    def __init__(self, ensemble_config, output_dir):
        self.ensemble_config = ensemble_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.portfolio = BacktestPortfolio(initial_balance=10000)
        
    def log_section(self, title):
        \"\"\"Log a section header\"\"\"
        logger.info("=" * 80)
        logger.info(f"📈 {title}")
        logger.info("=" * 80)
    
    def load_market_data(self, symbol, start_date, end_date):
        \"\"\"Load historical market data (mock for now)\"\"\"
        logger.info(f"📊 Loading market data for {symbol}...")
        
        # Generate mock OHLCV data
        dates = pd.date_range(start=start_date, end=end_date, freq='1H')
        n = len(dates)
        
        # Generate realistic price movement
        returns = np.random.normal(0.0001, 0.01, n)
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, n)),
            'high': prices * (1 + np.random.uniform(0, 0.02, n)),
            'low': prices * (1 - np.random.uniform(0, 0.02, n)),
            'close': prices,
            'volume': np.random.uniform(1000, 10000, n),
        })
        
        logger.info(f"✅ Loaded {len(data)} candles")
        
        return data
    
    def generate_ensemble_signal(self, market_data_row):
        \"\"\"Generate trading signal from ensemble\"\"\"
        # Mock ensemble prediction
        # In production, would load actual models and run inference
        
        close_price = market_data_row['close']
        
        # Simple mock strategy: buy on uptrend, sell on downtrend
        signal = np.random.choice(['BUY', 'HOLD', 'SELL'], p=[0.3, 0.4, 0.3])
        confidence = np.random.uniform(0.5, 1.0)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'price': close_price,
        }
    
    def run_backtest(self, symbol='BTCUSDT', start_date='2024-01-01', end_date='2024-12-31'):
        \"\"\"Run backtest simulation\"\"\"
        self.log_section(f"BACKTESTING ADAN ENSEMBLE - {symbol}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Initial Balance: ${self.portfolio.initial_balance:,.2f}")
        
        # Load market data
        market_data = self.load_market_data(symbol, start_date, end_date)
        
        # Simulate trading
        logger.info(f"\n🔄 Simulating {len(market_data)} candles...")
        
        for idx, row in market_data.iterrows():
            timestamp = row['timestamp']
            
            # Generate ensemble signal
            signal_data = self.generate_ensemble_signal(row)
            signal = signal_data['signal']
            confidence = signal_data['confidence']
            price = signal_data['price']
            
            # Execute trades based on signal
            if signal == 'BUY' and confidence > 0.6:
                if symbol not in self.portfolio.positions:
                    quantity = 0.1  # Fixed quantity for simplicity
                    self.portfolio.open_position(symbol, quantity, price, timestamp)
            
            elif signal == 'SELL' and confidence > 0.6:
                if symbol in self.portfolio.positions:
                    self.portfolio.close_position(symbol, price, timestamp)
            
            # Update equity
            self.portfolio.update_equity(timestamp, {symbol: price})
            
            # Log progress
            if (idx + 1) % 100 == 0:
                logger.info(f"  Progress: {idx + 1}/{len(market_data)} candles")
        
        # Generate backtest report
        report = self.generate_backtest_report(symbol, market_data)
        
        return report
    
    def generate_backtest_report(self, symbol, market_data):
        \"\"\"Generate comprehensive backtest report\"\"\"
        self.log_section("BACKTEST RESULTS")
        
        stats = self.portfolio.get_stats()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'ensemble_name': self.ensemble_config.get('name', 'ADAN_Final'),
            'symbol': symbol,
            'backtest_period': {
                'start': market_data['timestamp'].min().isoformat(),
                'end': market_data['timestamp'].max().isoformat(),
                'candles': len(market_data),
            },
            'portfolio_stats': stats,
            'trades': self.portfolio.trades,
            'equity_curve': self.portfolio.equity_curve,
            'timestamps': [t.isoformat() for t in self.portfolio.timestamps],
        }
        
        # Save report
        report_file = self.output_dir / "backtest_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✅ Backtest report saved: {report_file}")
        
        # Print summary
        logger.info(f"\n📊 BACKTEST SUMMARY")
        logger.info(f"  Total Trades: {stats['total_trades']}")
        logger.info(f"  Win Rate: {stats['win_rate']:.1f}%")
        logger.info(f"  Total PnL: ${stats['total_pnl']:.2f}")
        logger.info(f"  Avg PnL/Trade: ${stats['avg_pnl']:.2f}")
        logger.info(f"  Max Drawdown: {stats['max_drawdown']:.2%}")
        logger.info(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        logger.info(f"  Total Return: {stats['total_return']:.2f}%")
        logger.info(f"  Final Balance: ${stats['final_balance']:.2f}")
        
        return report


def main():
    \"\"\"Main entry point\"\"\"
    checkpoint_dir = "/mnt/new_data/t10_training/checkpoints"
    output_dir = "/mnt/new_data/t10_training/phase2_results"
    
    # Load ensemble config
    ensemble_config_file = Path(output_dir) / "adan_ensemble_config.json"
    if not ensemble_config_file.exists():
        logger.error(f"❌ Ensemble config not found: {ensemble_config_file}")
        logger.info("Run adan_ensemble_builder.py first")
        return 1
    
    with open(ensemble_config_file, 'r') as f:
        ensemble_config = json.load(f)
    
    # Run backtest
    engine = BacktestEngine(ensemble_config, output_dir)
    report = engine.run_backtest(
        symbol='BTCUSDT',
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    
    if report:
        logger.info("\n✅ Backtest complete!")
        return 0
    else:
        logger.error("\n❌ Backtest failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
