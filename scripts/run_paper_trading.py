#!/usr/bin/env python3
"""
ADAN Paper Trading Bot
Real-time model evaluation with comprehensive performance analytics
"""

import os
import sys
import time
import json
import argparse
import logging
import traceback
import copy
import csv
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import pandas as pd

import ccxt
import psutil

# --- FORCE USE OF FIXED VIRTUAL CAPITAL ($29) FOR THIS MODEL ---
FORCED_USDT_CAPITAL = 29.0

def is_already_running(pid_file="/tmp/adan_pid"):
    try:
        if os.path.exists(pid_file):
            with open(pid_file) as f:
                pid = int(f.read().strip())
            if psutil.pid_exists(pid):
                logging.warning(f"Another paper_trading process exists (pid={pid}).")
                return True
    except Exception:
        pass
    return False

def write_pid_file(pid_file="/tmp/adan_pid"):
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))

def compute_order_usdt(requested_usdt: float, usdt_balance: float, capital_tier: dict):
    """
    Returns the USDT amount allowed for this order, enforcing:
     - do not use more than usdt_balance (the virtual $29)
     - do not exceed the tier max_position_size_pct
     - do not exceed tier max_capital (if set)
    """
    # 1) cap to available USDT (virtual capital)
    allowed = min(requested_usdt, float(usdt_balance))

    # 2) cap by tier percentage
    pct = float(capital_tier.get("max_position_size_pct", 100)) / 100.0
    by_pct = float(usdt_balance) * pct
    allowed = min(allowed, by_pct)

    # 3) cap by tier max_capital if set
    tier_max = capital_tier.get("max_capital", None)
    if tier_max is not None:
        allowed = min(allowed, float(tier_max))

    # ensure non-negative
    return max(0.0, round(allowed, 8))

def compute_unrealized_pnl(position: dict, current_price: float):
    """
    position: dict with keys 'qty' and 'entry_price' and 'side' ('LONG'|'SHORT' or 'NONE')
    returns (pnl_abs, pnl_pct)
    """
    try:
        qty = float(position.get("qty", 0.0))
        entry = float(position.get("entry_price", 0.0))
    except Exception:
        return 0.0, 0.0
    if qty == 0 or entry == 0.0:
        return 0.0, 0.0
    # long: pnl = (current - entry) * qty
    pnl_abs = (current_price - entry) * qty
    # percent relative to position not to capital: use entry*qty as cost
    cost = entry * qty
    pnl_pct = (pnl_abs / cost * 100.0) if cost != 0 else 0.0
    return round(pnl_abs, 2), round(pnl_pct, 2)

def dump_status_atomic(path: str,
                       symbol: str,
                       exchange: str,
                       mode: str,
                       usdt_balance: float,
                       btc_balance: float,
                       position: dict,
                       recent_trades: list,
                       current_price: float,
                       pid: int,
                       latency_ms: int):
    # ensure position is current before compute
    pos = position or {"side": "NONE", "qty": 0.0, "entry_price": 0.0}
    pnl_abs, pnl_pct = compute_unrealized_pnl(pos, current_price)

    status = {
        "symbol": symbol,
        "exchange": exchange,
        "mode": mode,
        "pid": pid,
        "exchange_connected": True,
        "latency_ms": latency_ms,
        "last_update": time.strftime("%H:%M:%S"),
        "position": {
            "side": pos.get("side", "NONE"),
            "qty": float(pos.get("qty", 0.0)),
            "entry_price": float(pos.get("entry_price", 0.0)),
            "unrealized_pnl": pnl_abs
        },
        "recent_trades": recent_trades,
        "price_series": [current_price],
        "account": {
            "balance": float(FORCED_USDT_CAPITAL),      # virtual usable capital
            "initial_balance": float(FORCED_USDT_CAPITAL), # keep as before
            "equity": float(FORCED_USDT_CAPITAL),       # virtual equity
            "pnl_absolute": pnl_abs,
            "pnl_percent": pnl_pct,
            "real_equity": round(btc_balance * current_price + usdt_balance, 2),
            "usdt_balance": float(usdt_balance),        # Real USDT
            "btc_balance": float(btc_balance),          # Real BTC
            "usdt_virtual": FORCED_USDT_CAPITAL
        },
        "system": {
            "pid": pid,
            "latency_ms": latency_ms,
            "mode": mode,
            "exchange": exchange
        },
        "timestamp": time.time()
    }

    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic replace on POSIX

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.data_processing.feature_engineer import FeatureEngineer
from adan_trading_bot.data_processing.state_builder import StateBuilder


# ═══════════════════════════════════════════════════════════════════════════
# 📊 PERFORMANCE TRACKER - Professional Metrics & Analytics
# ═══════════════════════════════════════════════════════════════════════════

class PerformanceTracker:
    """
    Professional-grade performance tracking for live trading evaluation
    Tracks 17+ metrics with CSV persistence and real-time calculations
    """
    
    def __init__(self, initial_balance: float, data_dir: Path):
        self.initial_balance = initial_balance
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Core tracking
        self.trades = []
        self.equity_curve = []
        self.peak_balance = initial_balance
        self.current_balance = initial_balance
        
        # Trade statistics
        self.total_pnl_realized = 0.0
        self.total_commission = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0
        
        # Time tracking
        self.start_time = datetime.now()
        self.total_position_time = timedelta(0)
        self.last_position_open = None
        
        # File paths
        self.trades_file = self.data_dir / "trades_history.csv"
        self.equity_file = self.data_dir / "equity_curve.csv"
        
        # Initialize CSV files
        self._init_csv_files()
        
        logging.info(f"📊 PerformanceTracker initialized with ${initial_balance:.2f}")
    
    def _init_csv_files(self):
        """Initialize CSV files with headers"""
        if not self.trades_file.exists():
            with open(self.trades_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'side', 'entry_price', 'exit_price', 
                    'qty', 'pnl', 'pnl_pct', 'commission', 'duration_sec', 
                    'balance_after', 'roi_pct'
                ])
        
        if not self.equity_file.exists():
            with open(self.equity_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'balance', 'pnl_realized', 'pnl_unrealized',
                    'drawdown_pct', 'peak_balance'
                ])
    
    def record_trade(self, trade_data: dict):
        """
        Record a completed trade and update all metrics
        
        Args:
            trade_data: {
                'timestamp': datetime,
                'side': 'BUY' or 'SELL',
                'entry_price': float,
                'exit_price': float,
                'qty': float,
                'pnl': float,
                'commission': float,
                'duration': timedelta
            }
        """
        # Calculate additional metrics
        pnl = trade_data['pnl']
        commission = trade_data.get('commission', 0.0)
        net_pnl = pnl - commission
        
        # Calculate PnL %
        cost_basis = trade_data['entry_price'] * trade_data['qty']
        pnl_pct = (net_pnl / cost_basis * 100) if cost_basis > 0 else 0.0
        
        # Update balance
        self.current_balance += net_pnl
        self.total_pnl_realized += net_pnl
        self.total_commission += commission
        
        # Update peak
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        # Track consecutive wins/losses
        if net_pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.max_consecutive_wins = max(self.max_consecutive_wins, self.consecutive_wins)
        elif net_pnl < 0:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
        
        # ROI from initial
        roi_pct = ((self.current_balance - self.initial_balance) / self.initial_balance * 100)
        
        # Store trade
        trade_record = {
            'timestamp': trade_data['timestamp'],
            'side': trade_data['side'],
            'entry_price': trade_data['entry_price'],
            'exit_price': trade_data.get('exit_price', 0.0),
            'qty': trade_data['qty'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'commission': commission,
            'duration_sec': trade_data.get('duration', timedelta(0)).total_seconds(),
            'balance_after': self.current_balance,
            'roi_pct': roi_pct
        }
        
        self.trades.append(trade_record)
        
        # Save to CSV
        self._save_trade_to_csv(trade_record)
        
        logging.info(f"📝 Trade recorded: {trade_data['side']} | PnL: ${net_pnl:+.2f} ({pnl_pct:+.2f}%) | Balance: ${self.current_balance:.2f}")
    
    def _save_trade_to_csv(self, trade: dict):
        """Append trade to CSV file"""
        with open(self.trades_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trade['timestamp'].isoformat(),
                trade['side'],
                trade['entry_price'],
                trade['exit_price'],
                trade['qty'],
                trade['pnl'],
                trade['pnl_pct'],
                trade['commission'],
                trade['duration_sec'],
                trade['balance_after'],
                trade['roi_pct']
            ])
    
    def update_equity_curve(self, timestamp: datetime, unrealized_pnl: float = 0.0):
        """Update equity curve snapshot"""
        drawdown_pct = ((self.current_balance - self.peak_balance) / self.peak_balance * 100) if self.peak_balance > 0 else 0.0
        
        snapshot = {
            'timestamp': timestamp,
            'balance': self.current_balance,
            'pnl_realized': self.total_pnl_realized,
            'pnl_unrealized': unrealized_pnl,
            'drawdown_pct': drawdown_pct,
            'peak_balance': self.peak_balance
        }
        
        self.equity_curve.append(snapshot)
        
        # Save to CSV (append)
        with open(self.equity_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp.isoformat(),
                self.current_balance,
                self.total_pnl_realized,
                unrealized_pnl,
                drawdown_pct,
                self.peak_balance
            ])
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe Ratio (annualized)
        Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
        """
        if len(self.trades) < 2:
            return 0.0
        
        returns = [t['pnl_pct'] / 100 for t in self.trades]
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize (assuming ~252 trading days)
        sharpe = (mean_return - risk_free_rate) / std_return * np.sqrt(252)
        
        return sharpe
    
    def calculate_sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sortino Ratio (Sharpe but only downside volatility)
        """
        if len(self.trades) < 2:
            return 0.0
        
        returns = [t['pnl_pct'] / 100 for t in self.trades]
        mean_return = np.mean(returns)
        
        # Only negative returns
        downside_returns = [r for r in returns if r < 0]
        
        if not downside_returns:
            return float('inf')  # No losses!
        
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return 0.0
        
        sortino = (mean_return - risk_free_rate) / downside_std * np.sqrt(252)
        
        return sortino
    
    def calculate_max_drawdown(self) -> tuple:
        """
        Calculate maximum drawdown from equity curve
        Returns: (max_dd_pct, max_dd_amount)
        """
        if not self.equity_curve:
            current_dd = ((self.current_balance - self.peak_balance) / self.peak_balance * 100) if self.peak_balance > 0 else 0.0
            return (current_dd, 0.0)
        
        balances = [e['balance'] for e in self.equity_curve] + [self.current_balance]
        
        max_dd_pct = 0.0
        max_dd_amount = 0.0
        peak = balances[0]
        
        for balance in balances:
            if balance > peak:
                peak = balance
            
            dd = balance - peak
            dd_pct = (dd / peak * 100) if peak > 0 else 0.0
            
            if dd_pct < max_dd_pct:
                max_dd_pct = dd_pct
                max_dd_amount = dd
        
        return (max_dd_pct, max_dd_amount)
    
    def calculate_win_rate(self) -> float:
        """Calculate percentage of winning trades"""
        if not self.trades:
            return 0.0
        
        winning_trades = sum(1 for t in self.trades if t['pnl'] > 0)
        return (winning_trades / len(self.trades) * 100)
    
    def calculate_profit_factor(self) -> float:
        """
        Calculate Profit Factor = Gross Profit / Gross Loss
        Values > 1.0 indicate profitability
        """
        if not self.trades:
            return 0.0
        
        gross_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def calculate_expectancy(self) -> float:
        """
        Calculate average $ expected per trade
        Expectancy = (Win% × Avg Win) - (Loss% × Avg Loss)
        """
        if not self.trades:
            return 0.0
        
        wins = [t['pnl'] for t in self.trades if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in self.trades if t['pnl'] < 0]
        
        win_rate = len(wins) / len(self.trades) if self.trades else 0
        loss_rate = 1 - win_rate
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        
        return expectancy
    
    def calculate_market_exposure(self) -> float:
        """
        Calculate % of time spent in position
        """
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        if total_time == 0:
            return 0.0
        
        # Add current position time if open
        position_time = self.total_position_time.total_seconds()
        if self.last_position_open:
            position_time += (datetime.now() - self.last_position_open).total_seconds()
        
        return (position_time / total_time * 100)
    
    def get_metrics(self) -> dict:
        """
        Get all performance metrics as a dictionary
        """
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] < 0]
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0.0
        
        avg_hold_time = np.mean([t['duration_sec'] for t in self.trades]) if self.trades else 0.0
        
        max_dd_pct, max_dd_amount = self.calculate_max_drawdown()
        
        roi_pct = ((self.current_balance - self.initial_balance) / self.initial_balance * 100) if self.initial_balance > 0 else 0.0
        
        return {
            # Core Performance
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'total_pnl': self.total_pnl_realized,
            'roi_pct': roi_pct,
            
            # Risk Metrics
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'sortino_ratio': self.calculate_sortino_ratio(),
            'max_drawdown_pct': max_dd_pct,
            'max_drawdown_amount': max_dd_amount,
            
            # Trade Statistics
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate_pct': self.calculate_win_rate(),
            'profit_factor': self.calculate_profit_factor(),
            'expectancy': self.calculate_expectancy(),
            
            # Averages
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_hold_time_sec': avg_hold_time,
            
            # Streaks
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'current_streak_wins': self.consecutive_wins,
            'current_streak_losses': self.consecutive_losses,
            
            # Exposure
            'market_exposure_pct': self.calculate_market_exposure(),
            'total_commission': self.total_commission,
            
            # Time
            'runtime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
        }
    
    def position_opened(self):
        """Mark position as opened"""
        self.last_position_open = datetime.now()
    
    def position_closed(self):
        """Mark position as closed and track duration"""
        if self.last_position_open:
            duration = datetime.now() - self.last_position_open
            self.total_position_time += duration
            self.last_position_open = None


# Add to path for ensemble import
sys.path.insert(0, str(Path(__file__).parent))
from ensemble_manager import EnsembleManager

# Setup logging
log_file = Path(__file__).parent.parent / "logs" / "paper_trading.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_file)),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PaperTrader")

class BinancePaperTrader:
    def __init__(self, config_path: str, models_dir: str, api_key: str, api_secret: str, strategy: str = "median"):
        self.config = ConfigLoader().load_config(config_path)
        self.symbol = "BTC/USDT" # Default target
        self.timeframes = ["5m", "1h", "4h"]
        
        # Initialize Exchange
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
            }
        })
        self.exchange.set_sandbox_mode(True) # Enable Testnet
        
        # Initialize Components
        # FeatureEngineer expects the full config containing 'feature_engineering'
        self.feature_engineer = FeatureEngineer(self.config, models_dir)
        
        # Flatten features_config for StateBuilder
        # Config has nested structure: timeframes -> tf -> category -> [features]
        # StateBuilder expects: timeframes -> tf -> [features]
        raw_features_config = self.config['data']['features_config']['timeframes']
        flat_features_config = {}
        for tf, categories in raw_features_config.items():
            features = []
            if isinstance(categories, dict):
                for cat, items in categories.items():
                    if isinstance(items, list):
                        features.extend(items)
            else:
                features = categories # Already a list?
            flat_features_config[tf] = features
            
        # StateBuilder needs specific initialization
        window_sizes_from_config = self.config['environment']['observation']['window_sizes']
        logger.info(f"🔍 DEBUG: window_sizes from config: {window_sizes_from_config}")
        
        self.state_builder = StateBuilder(
            features_config=flat_features_config,
            window_sizes=window_sizes_from_config,
            include_portfolio_state=True,
            normalize=True,
            adaptive_window=True
        )
        
        # Load Ensemble
        self.ensemble = EnsembleManager(models_dir=models_dir, strategy=strategy)
        
        # Trading State - Fixed initial capital as requested
        self.initial_balance = FORCED_USDT_CAPITAL  # User-specified starting capital
        logger.info(f"💰 Initial Capital (Fixed): ${self.initial_balance:.2f}")
            
        self.account_balance = self.initial_balance
        self.current_position = 0.0
        self.entry_price = 0.0
        self.current_step = 0  # Step counter for caching logic
        
        # Performance: OHLCV cache
        self._ohlcv_cache = {}  # {(symbol, tf): (last_ts, df)}
        self.entry_time = None
        self.recent_trades = deque(maxlen=10)
        self.price_series = deque(maxlen=50)
        self.status_file = Path("status.json")
        
        # Performance Tracking
        self.performance = PerformanceTracker(
            initial_balance=self.initial_balance,
            data_dir=Path("data/paper_trading")
        )
        
        logger.info("✅ Binance Paper Trader Initialized")
        logger.info(f"Target: {self.symbol}")
        logger.info(f"Strategy: {strategy}")
        logger.info(f"💰 Initial Balance: ${self.initial_balance:.2f}")


    def dump_status(self, market_data: dict = None, latency_ms: float = 0):
        """Export current status to JSON for dashboard using atomic dump"""
        try:
            # Gather real metrics
            balance = self.exchange.fetch_balance()
            usdt_real = balance['USDT']['total']
            btc_real = balance['BTC']['total']
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']
            
            # Position dict
            position = {
                "side": "LONG" if self.current_position > 0 else "SHORT" if self.current_position < 0 else "NONE",
                "qty": abs(self.current_position),
                "entry_price": self.entry_price
            }
            
            dump_status_atomic(
                path=self.status_file,
                symbol=self.symbol,
                exchange="binance-testnet",
                mode="paper",
                usdt_balance=usdt_real, # Real USDT balance passed here
                btc_balance=btc_real,   # Real BTC holding
                position=position,
                recent_trades=list(self.recent_trades),
                current_price=current_price,
                pid=os.getpid(),
                latency_ms=int(latency_ms)
            )
        except Exception as e:
            logger.error(f"Error dumping status: {e}")
            # Try to dump a minimal status to signal we are alive but failing
            try:
                with open(self.status_file, "w") as f:
                    json.dump({"error": str(e), "timestamp": time.time(), "pid": os.getpid()}, f)
            except:
                pass

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """Fetch OHLCV data from Binance"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Ensure numeric types
            cols = ['open', 'high', 'low', 'close', 'volume']
            df[cols] = df[cols].apply(pd.to_numeric)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching {timeframe} data: {type(e).__name__} - {e}")
            return pd.DataFrame()

    def prepare_data(self) -> dict[str, dict[str, pd.DataFrame]]:
        """Fetch and process data for all timeframes"""
        data = {self.symbol.replace('/', ''): {}}
        
        # 🔧 FIX: Different limits per timeframe to ensure enough data
        timeframe_limits = {
            '5m': 1000,
            '1h': 500, 
            '4h': 300  # Increased for 4h to avoid "Missing data"
        }
        
        for tf in self.timeframes:
            limit = timeframe_limits.get(tf, 500)
            df = self.fetch_ohlcv(self.symbol, tf, limit=limit)
            
            if df.empty:
                logger.warning(f"⚠️ Empty dataframe for {tf}")
                continue
            
            logger.info(f"✅ Fetched {len(df)} candles for {tf}")
                
            # Calculate indicators
            try:
                df_processed = self.feature_engineer.calculate_indicators_for_single_timeframe(df, tf)
                data[self.symbol.replace('/', '')][tf] = df_processed
            except Exception as e:
                logger.error(f"Error calculating indicators for {tf}: {e}")
            
        return data

    def get_portfolio_state(self) -> np.ndarray:
        """
        Construct portfolio state vector.
        In paper trading, we fetch real balance from testnet.
        """
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = balance['USDT']['free']
            btc_balance = balance['BTC']['free']
            
            # Get current price
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']
            
            # If we have BTC but no tracked entry_price, it's from a previous session
            # Reconstruct approximate entry_price from recent trades or use current price as estimate
            if btc_balance > 0.0001 and self.entry_price == 0:
                # Try to get from recent trades
                try:
                    trades = self.exchange.fetch_my_trades(self.symbol, limit=1)
                    if trades:
                        self.entry_price = float(trades[-1]['price'])
                        self.current_position = btc_balance
                        logger.info(f"📦 Detected existing BTC position: {btc_balance:.8f} BTC @ ${self.entry_price:.2f}")
                    else:
                        # No trades found, use current price as rough estimate
                        self.entry_price = current_price * 0.99  # Assume small profit
                        self.current_position = btc_balance
                        logger.warning(f"⚠️ BTC position detected but no trade history. Using estimated entry: ${self.entry_price:.2f}")
                except Exception as e:
                    # Fallback: use current price
                    self.entry_price = current_price
                    self.current_position = btc_balance
                    logger.warning(f"⚠️ Could not fetch trades: {e}. Using current price as entry.")
            
            # Estimate equity
            equity = usdt_balance + (btc_balance * current_price)
            
            state = np.zeros(20, dtype=np.float32)
            state[0] = usdt_balance
            state[1] = equity
            state[2] = btc_balance  # Position size - MODEL SEES THIS!
            state[3] = self.entry_price  # Entry price - now populated
            state[4] = current_price
            
            # PnL
            if self.current_position > 0 and self.entry_price > 0:
                pnl = (current_price - self.entry_price) / self.entry_price if self.entry_price != 0 else 0.0
                state[5] = pnl
            
            # Drawdown (simplified)
            peak = getattr(self, 'peak_equity', equity)
            if equity > peak:
                self.peak_equity = equity
                peak = equity
            drawdown = (peak - equity) / peak if peak > 0 else 0.0
            state[6] = drawdown
            
            # Log for debugging
            if btc_balance > 0.0001:
                logger.info(f"📊 Portfolio State -> USDT: ${usdt_balance:.2f}, BTC: {btc_balance:.8f}, Entry: ${self.entry_price:.2f}, PnL: {pnl*100:.2f}%")
            
            return state
            
        except Exception as e:
            logger.error(f"Error fetching portfolio state: {e}")
            return np.zeros(20, dtype=np.float32)

    def _get_capital_tier(self):
        """Get the capital tier configuration based on current balance"""
        tiers = self.config.get('capital_tiers', [])
        if not tiers:
            # Default tier if not configured
            return {
                'name': 'Default',
                'max_position_size_pct': 50,
                'max_concurrent_positions': 1,
                'risk_per_trade_pct': 2.0,
                'max_drawdown_pct': 5.0
            }
        
        # Find matching tier based on current balance
        balance = self.account_balance
        for tier in tiers:
            min_cap = tier.get('min_capital', 0)
            max_cap = tier.get('max_capital', float('inf'))
            if min_cap <= balance <= max_cap:
                logger.info(f"📊 Capital Tier: {tier.get('name', 'Unknown')} (${min_cap}-${max_cap})")
                return tier
        
        # Fallback to first tier if no match
        return tiers[0]
    
    def execute_trade(self, action: float):
        """Execute trade on Binance Testnet"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            price = ticker['last']
            
            balance = self.exchange.fetch_balance()
            usdt_free = balance['USDT']['free']
            btc_free = balance['BTC']['free']
            
            min_trade_value = 10.0 # Binance min trade
            
            trade_info = None
            
            if action > 0.2: # BUY Signal
                # 💰 CAPITAL CALCULATION (CRITICAL FIX)
                usdt_balance = FORCED_USDT_CAPITAL
                current_pos_value = self.current_position * price
                virtual_available = max(0.0, usdt_balance - current_pos_value)
                
                # Log capital status
                logger.info(f"💰 Capital: Total=${usdt_balance:.2f}, Used=${current_pos_value:.2f}, Available=${virtual_available:.2f}")
                
                # 🛡️ GUARD #1: Respect max_concurrent_positions = 1
                if self.current_position > 0:
                    logger.warning(f"⚠️ Position already open ({self.current_position:.6f} BTC @ ${self.entry_price:.2f}) - BUY IGNORED (action={action:.3f})")
                    return
                
                # 🛡️ GUARD #2: Ensure minimum capital available
                if virtual_available < min_trade_value:
                    logger.warning(f"⚠️ Insufficient virtual capital (${virtual_available:.2f} < ${min_trade_value}) - BUY IGNORED")
                    return
                
                # Calculate position size with tier limits
                tier = self._get_capital_tier()
                max_position_pct = tier.get('max_position_size_pct', 90) / 100.0
                
                # Use only AVAILABLE capital for sizing
                amount_usdt = virtual_available * max_position_pct
                
                if amount_usdt > min_trade_value:
                    quantity = amount_usdt / price
                    market = self.exchange.market(self.symbol)
                    quantity = self.exchange.amount_to_precision(self.symbol, quantity)
                    
                    logger.info(f"🚀 BUY SIGNAL (Action: {action:.2f}) | Price: {price} | Qty: {quantity}")

                    
                    order = self.exchange.create_market_buy_order(self.symbol, quantity)
                    logger.info(f"✅ BUY Executed: {order['id']}")
                    
                    self.current_position += float(order['amount'])
                    self.entry_price = float(order['average']) if order['average'] else price
                    
                    # Record trade with timestamp
                    from datetime import datetime
                    trade_info = {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "timestamp": datetime.now().isoformat(),
                        "side": "BUY",
                        "qty": float(quantity),
                        "price": float(self.entry_price),
                        "order_id": order['id'],
                        "pnl": 0.0
                    }
                else:
                    logger.info(f"⚠️ Buy signal but insufficient funds (Need: ${min_trade_value}, Available: ${amount_usdt:.2f})")

            elif action < -0.05: # SELL Signal (lowered from -0.2)
                # 🛡️ GUARD: Verify position exists and is sellable
                if btc_free * price > min_trade_value:
                    quantity = self.exchange.amount_to_precision(self.symbol, btc_free)
                    
                    logger.info(f"🔻 SELL SIGNAL (Action: {action:.2f}) | Price: {price} | Qty: {quantity}")
                    order = self.exchange.create_market_sell_order(self.symbol, quantity)
                    logger.info(f"✅ SELL Executed: {order['id']}")
                    
                    # Calculate PnL
                    exit_price = float(order['average']) if order['average'] else price
                    pnl = (exit_price - self.entry_price) * self.current_position
                    
                    # Record trade with timestamp
                    from datetime import datetime
                    trade_info = {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "timestamp": datetime.now().isoformat(),
                        "side": "SELL",
                        "qty": float(quantity),
                        "price": float(exit_price),
                        "order_id": order['id'],
                        "pnl": pnl
                    }
                    
                    self.current_position = 0.0
                    self.entry_price = 0.0
                else:
                    logger.info("⚠️ Sell signal but no position")
            
            else:
                logger.info(f"😴 HOLD (Action: {action:.2f})")
                
            if trade_info:
                self.recent_trades.appendleft(trade_info)

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")

    def run(self):
        logger.info("🔥 Starting Paper Trading Loop...")
        
        while True:
            try:
                start_time = time.time()
                
                # 1. Fetch & Process Data
                data = self.prepare_data()
                asset_key = self.symbol.replace('/', '')
                
                # Check if we have data for the essential timeframe (5m)
                if asset_key not in data or '5m' not in data[asset_key] or data[asset_key]['5m'].empty:
                    logger.warning("⚠️ Missing 5m data, retrying in 10s...")
                    time.sleep(10)
                    continue
                
                # Update Market Data for Dashboard
                df_5m = data[asset_key]['5m']
                last_candle = df_5m.iloc[-1]
                current_price = last_candle['close']
                
                self.price_series.append(float(current_price))
                
                market_data = {
                    "last_price": float(current_price),
                    "open": float(last_candle['open']),
                    "high": float(last_candle['high']),
                    "low": float(last_candle['low']),
                    "close": float(last_candle['close']),
                    "volume": float(last_candle['volume'])
                }
                
                # 2. Fit Scalers (on recent history)
                self.state_builder.fit_scalers(data)
                
                # 3. Build Observation with timing
                last_idx = len(df_5m) - 1
                
                class MockPortfolioManager:
                    def get_state_vector(self, *args, **kwargs):
                        return self.trader.get_portfolio_state()
                    def __init__(self, trader):
                        self.trader = trader
                
                t_data_start = time.perf_counter()
                obs_dict = self.state_builder.build_observation(
                    current_idx=last_idx,
                    data=data,
                    portfolio_manager=MockPortfolioManager(self)
                )
                t_data_end = time.perf_counter()
                data_prep_ms = int((t_data_end - t_data_start) * 1000)
                
                if data_prep_ms > 1000:
                    logger.warning(f"⚠️  Observation build took {data_prep_ms}ms (>1s)")
                
                # 4. Ensemble Prediction
                obs_batch = {}
                for key, val in obs_dict.items():
                    obs_batch[key] = np.expand_dims(val, axis=0)
                
                # Performance profiling
                t0 = time.perf_counter()
                
                # Sync with Exchange (OPTIMIZED - reuse ticker data)
                try:
                    # Use already-fetched price from data preparation
                    current_price = obs_dict.get('close', [0])[-1] if 'close' in obs_dict else 0.0
                    
                    # Only fetch balance every N iterations to reduce latency
                    if self.current_step % 5 == 0:  # Every 5 steps
                        balance = self.exchange.fetch_balance()
                        usdt_balance = float(balance['USDT']['total'])
                        btc_balance = float(balance['BTC']['total'])
                        
                        # Cache for next iterations
                        self._cached_usdt = usdt_balance
                        self._cached_btc = btc_balance
                    else:
                        # Use cached values
                        usdt_balance = getattr(self, '_cached_usdt', 0.0)
                        btc_balance = getattr(self, '_cached_btc', 0.0)
                    
                    t1 = time.perf_counter()
                    
                    # Sync position with actual BTC holdings
                    self.current_position = btc_balance
                    
                    # Calculate Total Equity (USDT + BTC value)
                    # For virtual $29 capital tracking, we ignore real balances and track internally
                    if self.current_position > 0 and self.entry_price > 0:
                        # We have a position - calculate based on virtual trades
                        position_value = self.current_position * current_price
                        # This is simplified - in reality we should track virtual USDT separately
                        total_equity = self.account_balance
                    else:
                        total_equity = self.account_balance

                except Exception as e:
                    logger.error(f"Failed to sync equity: {e}")
                    t1 = time.perf_counter()

                t2 = time.perf_counter()
                action, debug = self.ensemble.predict(obs_batch, deterministic=True)
                t3 = time.perf_counter()
                
                # Log performance metrics
                logger.info(f"⏱️  TIMING: sync={int((t1-t0)*1000)}ms | predict={int((t3-t2)*1000)}ms | total={int((t3-t0)*1000)}ms")
                
                # Log individual model predictions
                if 'predictions' in debug:
                    preds = debug['predictions']
                    pred_str = ", ".join([f"{k}: {float(np.mean(v)):.4f}" for k, v in preds.items()])
                    logger.info(f"🤖 Individual Predictions: {pred_str}")
                
                logger.info(f"🤖 Model Prediction: {action}")
                
                # 5. Execute Trade
                self.execute_trade(action)
                
                # Sync position and entry_price from exchange BEFORE dump_status
                try:
                    balance = self.exchange.fetch_balance()
                    btc_balance = balance['BTC']['free']
                    
                    # Update current_position from real balance
                    if btc_balance > 0.0001:
                        # If we have BTC but no entry_price tracked, reconstruct it
                        if self.entry_price == 0:
                            try:
                                trades = self.exchange.fetch_my_trades(self.symbol, limit=1)
                                if trades:
                                    self.entry_price = float(trades[-1]['price'])
                                    logger.info(f"🔄 Synced entry_price from trades: ${self.entry_price:.2f}")
                            except:
                                pass
                        self.current_position = btc_balance
                except Exception as e:
                    logger.error(f"Failed to sync position before dump_status: {e}")
                
                # 6. Update Dashboard Status
                latency = (time.time() - start_time) * 1000
                self.dump_status(market_data, latency)
                
                # 7. Wait for next candle (5m)
                now = datetime.now()
                minutes = now.minute
                seconds = now.second
                next_minute = (minutes // 5 + 1) * 5
                wait_seconds = (next_minute - minutes) * 60 - seconds
                
                if wait_seconds <= 0:
                    wait_seconds = 300 
                
                logger.info(f"⏳ Waiting {wait_seconds:.0f}s for next candle...")
                time.sleep(wait_seconds)
                
            except KeyboardInterrupt:
                logger.info("🛑 Stopping Paper Trader...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(60)

if __name__ == "__main__":
    if is_already_running():
        sys.exit(1)
    write_pid_file()

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True, help="Binance API Key")
    parser.add_argument("--api_secret", required=True, help="Binance API Secret")
    parser.add_argument("--strategy", default="median", help="Voting strategy")
    args = parser.parse_args()
    
    trader = BinancePaperTrader(
        config_path="config/config.yaml",
        models_dir="checkpoints_final/final",
        api_key=args.api_key,
        api_secret=args.api_secret,
        strategy=args.strategy
    )
    
    trader.run()
