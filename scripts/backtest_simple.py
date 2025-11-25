#!/usr/bin/env python3
"""
Simple backtest to verify if the model trades and is profitable.
"""
import sys
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stable_baselines3 import PPO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleBacktester:
    """Simple backtest without complex dependencies."""
    
    def __init__(self):
        # Load model
        model_path = Path('bot_pres/model/adan_model_checkpoint_640000_steps.zip')
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Loading model from {model_path}...")
        self.model = PPO.load(str(model_path))
        logger.info("✅ Model loaded successfully")
        
        # Load scalers
        scalers_path = Path('models/training_scalers.pkl')
        if scalers_path.exists():
            self.scalers = joblib.load(scalers_path)
            logger.info("✅ Training scalers loaded")
        else:
            logger.warning("⚠️  Training scalers not found")
            self.scalers = None
    
    def load_test_data(self):
        """Load test data from parquets."""
        logger.info("\n" + "="*80)
        logger.info("📊 LOADING TEST DATA")
        logger.info("="*80)
        
        data_dir = Path('data/processed/indicators/test/BTCUSDT')
        timeframes = ['5m', '1h', '4h']
        
        test_data = {}
        for tf in timeframes:
            parquet_file = data_dir / f'{tf}.parquet'
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
                logger.info(f"✅ Loaded {tf}: {len(df)} rows, {len(df.columns)} cols")
                test_data[tf] = df
            else:
                logger.error(f"❌ File not found: {parquet_file}")
        
        return test_data
    
    def scale_features(self, df, timeframe):
        """Scale features using training scalers."""
        if self.scalers is None or timeframe not in self.scalers:
            return df.values
        
        # Filter to training columns (same as in feature_engineer.py)
        train_columns = {
            '5m': ['open', 'high', 'low', 'close', 'volume', 'rsi_14',
                   'macd_12_26_9',
                   'bb_percent_b_20_2', 'atr_14', 'volume_ratio_20',
                   'ema_20_ratio', 'stoch_k_14_3_3', 'vwap_ratio',
                   'price_action'],
            '1h': ['open', 'high', 'low', 'close', 'volume', 'rsi_21',
                   'macd_21_42_9',
                   'bb_width_20_2', 'adx_14', 'obv_ratio_20',
                   'ema_50_ratio', 'ichimoku_base', 'fib_ratio',
                   'price_ema_ratio_50'],
            '4h': ['open', 'high', 'low', 'close', 'volume', 'rsi_28',
                   'macd_26_52_18',
                   'supertrend_10_3', 'volume_sma_20_ratio',
                   'ema_100_ratio', 'pivot_level', 'donchian_width_20',
                   'market_structure', 'volatility_ratio_14_50']
        }
        
        # Filter columns
        cols_to_keep = [c for c in train_columns[timeframe]
                        if c in df.columns]
        df_filtered = df[cols_to_keep]
        
        scaler = self.scalers[timeframe]
        return scaler.transform(df_filtered.values)
    
    def run_backtest(self, test_data, initial_balance=100.0):
        """Run simple backtest."""
        logger.info("\n" + "="*80)
        logger.info("🚀 RUNNING BACKTEST")
        logger.info("="*80)
        
        # Get minimum length
        min_len = min(len(df) for df in test_data.values())
        logger.info(f"Test period: {min_len} candles")
        
        # Trim all dataframes to same length
        for tf in test_data:
            test_data[tf] = test_data[tf].iloc[:min_len].reset_index(drop=True)
        
        # Initialize tracking
        balance = initial_balance
        position = 0.0
        entry_price = 0.0
        trades = []
        predictions = []
        prices_5m = test_data['5m']['close'].values
        
        logger.info(f"\n💰 Initial Balance: ${balance:.2f}")
        logger.info(f"📈 Price Range: ${prices_5m[0]:.2f} - ${prices_5m[-1]:.2f}")
        
        # Backtest loop
        window_size = 20
        for i in range(window_size, min_len, 5):  # Step by 5 for speed
            try:
                # Build simple observation from scaled features
                obs_dict = {}
                
                for tf in ['5m', '1h', '4h']:
                    # Get window of data
                    start_idx = max(0, i - window_size + 1)
                    window = test_data[tf].iloc[start_idx:i+1].copy()
                    
                    # Select only numeric columns
                    numeric_cols = window.select_dtypes(include=[np.number]).columns.tolist()
                    window = window[numeric_cols]
                    
                    # Scale
                    scaled = self.scale_features(window, tf)
                    
                    # Pad if needed
                    if len(scaled) < window_size:
                        pad_size = window_size - len(scaled)
                        scaled = np.vstack([np.zeros((pad_size, scaled.shape[1])), scaled])
                    
                    # Keep shape (window_size, n_features) NOT flattened
                    obs_dict[tf] = scaled[-window_size:].astype(np.float32)
                
                # Add portfolio state
                obs_dict['portfolio_state'] = np.array([
                    balance / initial_balance,
                    position,
                    entry_price / prices_5m[i] if entry_price > 0 else 0.0,
                    (prices_5m[i] - entry_price) / entry_price if entry_price > 0 else 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ], dtype=np.float32)
                
                # Expand dims for batch
                obs_batch = {}
                for key, val in obs_dict.items():
                    obs_batch[key] = np.expand_dims(val, axis=0)
                
                # Predict action
                action, _ = self.model.predict(obs_batch, deterministic=True)
                # Flatten and get first element
                if isinstance(action, np.ndarray):
                    action_scalar = float(np.asarray(action).flatten()[0])
                else:
                    action_scalar = float(action)
                current_price = prices_5m[i]
                
                # Execute action
                if action_scalar > 0.5 and position == 0:  # BUY
                    position = balance / current_price * 0.95
                    entry_price = current_price
                    balance = balance * 0.05
                    trades.append({
                        'step': i,
                        'type': 'BUY',
                        'price': current_price,
                        'size': position
                    })
                    logger.info(f"  🟢 BUY  @ step {i}: {position:.6f} BTC @ ${current_price:.2f}")
                
                elif action_scalar <= 0.5 and position > 0:  # SELL
                    pnl = (current_price - entry_price) * position
                    balance += position * current_price
                    trades.append({
                        'step': i,
                        'type': 'SELL',
                        'price': current_price,
                        'size': position,
                        'pnl': pnl
                    })
                    logger.info(f"  🔴 SELL @ step {i}: {position:.6f} BTC @ ${current_price:.2f} | PnL: ${pnl:.2f}")
                    position = 0.0
                    entry_price = 0.0
                
                predictions.append({
                    'step': i,
                    'action': action_scalar,
                    'price': current_price
                })
                
            except Exception as e:
                logger.error(f"Error at step {i}: {e}")
                continue
        
        # Close final position
        if position > 0:
            final_pnl = (prices_5m[-1] - entry_price) * position
            balance += position * prices_5m[-1]
            trades.append({
                'step': min_len - 1,
                'type': 'SELL (FINAL)',
                'price': prices_5m[-1],
                'size': position,
                'pnl': final_pnl
            })
            logger.info(f"  🔴 FINAL SELL @ step {min_len-1}: {position:.6f} BTC @ ${prices_5m[-1]:.2f} | PnL: ${final_pnl:.2f}")
        
        return {
            'trades': trades,
            'predictions': predictions,
            'final_balance': balance,
            'total_return': (balance - initial_balance) / initial_balance * 100
        }
    
    def print_results(self, results):
        """Print backtest results."""
        logger.info("\n" + "="*80)
        logger.info("📊 BACKTEST RESULTS")
        logger.info("="*80)
        
        trades = results['trades']
        predictions = results['predictions']
        
        logger.info(f"\n💰 PERFORMANCE")
        logger.info(f"  Initial Balance: $100.00")
        logger.info(f"  Final Balance: ${results['final_balance']:.2f}")
        logger.info(f"  Total Return: {results['total_return']:.2f}%")
        
        logger.info(f"\n📈 TRADING ACTIVITY")
        logger.info(f"  Total Trades: {len(trades)}")
        
        buys = [t for t in trades if t['type'] == 'BUY']
        sells = [t for t in trades if 'SELL' in t['type']]
        
        logger.info(f"  Buy Orders: {len(buys)}")
        logger.info(f"  Sell Orders: {len(sells)}")
        
        if sells:
            pnls = [t['pnl'] for t in sells if 'pnl' in t]
            wins = len([p for p in pnls if p > 0])
            losses = len([p for p in pnls if p < 0])
            total_pnl = sum(pnls)
            
            logger.info(f"\n📊 TRADE STATISTICS")
            logger.info(f"  Winning Trades: {wins}")
            logger.info(f"  Losing Trades: {losses}")
            if wins + losses > 0:
                logger.info(f"  Win Rate: {wins/(wins+losses)*100:.1f}%")
            logger.info(f"  Total PnL: ${total_pnl:.2f}")
            if len(sells) > 0:
                logger.info(f"  Avg PnL per Trade: ${total_pnl/len(sells):.2f}")
        
        logger.info(f"\n🤖 MODEL PREDICTIONS")
        if predictions:
            actions = np.array([p['action'] for p in predictions])
            logger.info(f"  Min Action: {actions.min():.4f}")
            logger.info(f"  Max Action: {actions.max():.4f}")
            logger.info(f"  Mean Action: {actions.mean():.4f}")
            logger.info(f"  Std Action: {actions.std():.4f}")
            
            buy_actions = len([a for a in actions if a > 0.5])
            sell_actions = len([a for a in actions if a <= 0.5])
            
            logger.info(f"  Buy Signals (>0.5): {buy_actions} ({buy_actions/len(actions)*100:.1f}%)")
            logger.info(f"  Sell Signals (<=0.5): {sell_actions} ({sell_actions/len(actions)*100:.1f}%)")
        
        # Diagnosis
        logger.info("\n" + "="*80)
        logger.info("🔍 DIAGNOSIS")
        logger.info("="*80)
        
        if len(trades) == 0:
            logger.warning("❌ NO TRADES EXECUTED - Model is completely blocked")
        elif len(buys) > 0 and len(sells) == 0:
            logger.warning("⚠️  ONLY BUY SIGNALS - Model never sells (stuck in position)")
        elif predictions and np.array([p['action'] for p in predictions]).std() < 0.01:
            logger.warning("⚠️  PREDICTIONS ARE UNIFORM - Model predicts same action always")
        elif results['total_return'] < -50:
            logger.warning("⚠️  LARGE LOSSES - Model is losing money significantly")
        elif results['total_return'] > 0:
            logger.info("✅ MODEL IS PROFITABLE - Backtest shows positive returns")
        else:
            logger.info("⚠️  MODEL IS BREAK-EVEN - Backtest shows minimal returns")

def main():
    try:
        backtester = SimpleBacktester()
        test_data = backtester.load_test_data()
        results = backtester.run_backtest(test_data)
        backtester.print_results(results)
        
        logger.info("\n" + "="*80)
        logger.info("✅ BACKTEST COMPLETED")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Backtest failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
