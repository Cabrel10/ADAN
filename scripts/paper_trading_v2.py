#!/usr/bin/env python3
"""
ADAN PAPER TRADING V2 - CLEAN REWRITE
Production-grade paper trading bot
Ingénieur mode: Rigoureux, testé, production-ready
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import ccxt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stable_baselines3 import PPO
from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.data_processing.state_builder import StateBuilder
from adan_trading_bot.data_processing.feature_engineer import FeatureEngineer


class PaperTradingBot:
    """Production paper trading bot"""

    def __init__(self, config_path: str, model_path: str, api_key: str, 
                 api_secret: str, capital: float = 29.0):
        """Initialize paper trading bot"""
        logger.info("="*80)
        logger.info("ADAN PAPER TRADING V2 - INITIALIZATION")
        logger.info("="*80)
        
        # Load config
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.load_config(config_path)
        self.config['initial_capital'] = capital
        logger.info(f"✅ Config loaded: {config_path}")
        
        # Initialize components
        self.state_builder = StateBuilder(
            features_config=self._get_features_config(),
            window_sizes=self.config['environment']['observation']['window_sizes'],
            include_portfolio_state=True,
            normalize=True,
            adaptive_window=True
        )
        logger.info("✅ StateBuilder initialized")
        
        self.feature_engineer = FeatureEngineer(self.config, "models")
        logger.info("✅ FeatureEngineer initialized")
        
        # Load model
        self.model = PPO.load(model_path)
        logger.info(f"✅ Model loaded: {model_path}")
        
        # Exchange
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        logger.info("✅ Exchange connected")
        
        # Trading state
        self.capital = capital
        self.position = None
        self.entry_price = None
        self.entry_time = None
        self.trades = []
        self.step_count = 0
        
        logger.info(f"✅ Paper trading initialized with ${capital:.2f}")
        logger.info("="*80)

    def _get_features_config(self) -> Dict:
        """Extract features config from config"""
        raw = self.config['data']['features_config']['timeframes']
        flat = {}
        for tf, categories in raw.items():
            if isinstance(categories, dict):
                features = []
                for cat_features in categories.values():
                    if isinstance(cat_features, list):
                        features.extend(cat_features)
                flat[tf] = features
            else:
                flat[tf] = categories
        return flat

    def fetch_ohlcv(self, symbol: str = 'BTC/USDT', 
                   timeframe: str = '5m', limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"❌ Error fetching OHLCV: {e}")
            return None

    def build_observation(self, symbol: str = 'BTC/USDT') -> Optional[Dict]:
        """Build observation for model"""
        try:
            # Fetch data for all timeframes
            data = {}
            for tf in ['5m', '1h', '4h']:
                df = self.fetch_ohlcv(symbol, tf, limit=100)
                if df is None:
                    return None
                data[tf] = df
            
            # Calculate indicators
            for tf, df in data.items():
                df = self.feature_engineer.calculate_indicators(df, tf)
                data[tf] = df
            
            # Build observation
            obs = self.state_builder.build_observation(
                data=data,
                portfolio_state={
                    'balance': self.capital,
                    'position': self.position if self.position else 0,
                    'entry_price': self.entry_price if self.entry_price else 0
                }
            )
            
            return obs
        except Exception as e:
            logger.error(f"❌ Error building observation: {e}")
            return None

    def predict_action(self, obs: Dict) -> Tuple[float, float]:
        """Get model prediction"""
        try:
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Extract individual predictions (4 workers)
            if isinstance(action, np.ndarray) and len(action) >= 4:
                w1, w2, w3, w4 = action[:4]
                ensemble_action = np.median([w1, w2, w3, w4])
            else:
                ensemble_action = float(action) if isinstance(action, np.ndarray) else action
            
            logger.info(f"🤖 Individual Predictions: w1: {w1:.4f}, w2: {w2:.4f}, w3: {w3:.4f}, w4: {w4:.4f}")
            logger.info(f"🤖 Ensemble Action: {ensemble_action:.4f}")
            
            return ensemble_action, action
        except Exception as e:
            logger.error(f"❌ Error predicting action: {e}")
            return 0.0, None

    def execute_trade(self, symbol: str, action: float, 
                     current_price: float) -> bool:
        """Execute trade based on action"""
        try:
            # BUY signal
            if action > 0.5 and self.position is None:
                amount = (self.capital * 0.9) / current_price
                logger.info(f"📈 BUY signal: {amount:.6f} {symbol} @ ${current_price:.2f}")
                
                self.position = amount
                self.entry_price = current_price
                self.entry_time = datetime.now()
                logger.info(f"✅ POSITION OUVERTE: {amount:.6f} {symbol} @ ${current_price:.2f}")
                return True
            
            # SELL signal
            elif action < -0.5 and self.position is not None:
                pnl = (current_price - self.entry_price) * self.position
                pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
                
                logger.info(f"📉 SELL signal: {self.position:.6f} {symbol} @ ${current_price:.2f}")
                logger.info(f"✅ POSITION FERMÉE: PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
                
                self.trades.append({
                    'entry_price': self.entry_price,
                    'exit_price': current_price,
                    'amount': self.position,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'entry_time': self.entry_time,
                    'exit_time': datetime.now()
                })
                
                self.capital += pnl
                self.position = None
                self.entry_price = None
                return True
            
            return False
        except Exception as e:
            logger.error(f"❌ Error executing trade: {e}")
            return False

    def run_loop(self, symbol: str = 'BTC/USDT', interval: int = 60):
        """Main trading loop"""
        logger.info(f"\n🚀 Starting paper trading loop (interval: {interval}s)")
        logger.info("="*80)
        
        try:
            while True:
                self.step_count += 1
                logger.info(f"\n[STEP {self.step_count}] {datetime.now().isoformat()}")
                
                # Build observation
                obs = self.build_observation(symbol)
                if obs is None:
                    logger.warning("⚠️ Failed to build observation, retrying...")
                    time.sleep(interval)
                    continue
                
                # Get current price
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                except Exception as e:
                    logger.error(f"❌ Error fetching price: {e}")
                    time.sleep(interval)
                    continue
                
                # Predict action
                action, raw_action = self.predict_action(obs)
                
                # Execute trade
                self.execute_trade(symbol, action, current_price)
                
                # Log state
                logger.info(f"💰 Capital: ${self.capital:.2f}")
                if self.position:
                    unrealized_pnl = (current_price - self.entry_price) * self.position
                    logger.info(f"📊 Position: {self.position:.6f} {symbol} @ ${self.entry_price:.2f} (unrealized: ${unrealized_pnl:.2f})")
                
                # Log stats
                if self.trades:
                    total_pnl = sum(t['pnl'] for t in self.trades)
                    win_count = sum(1 for t in self.trades if t['pnl'] > 0)
                    logger.info(f"📈 Trades: {len(self.trades)}, Win rate: {win_count}/{len(self.trades)}, Total PnL: ${total_pnl:.2f}")
                
                # Wait for next interval
                logger.info(f"⏳ Waiting {interval}s for next cycle...")
                time.sleep(interval)
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Paper trading stopped by user")
            self._print_summary()
        except Exception as e:
            logger.error(f"❌ Fatal error in trading loop: {e}")
            import traceback
            traceback.print_exc()
            self._print_summary()

    def _print_summary(self):
        """Print trading summary"""
        logger.info("\n" + "="*80)
        logger.info("PAPER TRADING SUMMARY")
        logger.info("="*80)
        logger.info(f"Steps: {self.step_count}")
        logger.info(f"Trades: {len(self.trades)}")
        logger.info(f"Final Capital: ${self.capital:.2f}")
        
        if self.trades:
            total_pnl = sum(t['pnl'] for t in self.trades)
            win_count = sum(1 for t in self.trades if t['pnl'] > 0)
            avg_pnl = total_pnl / len(self.trades)
            
            logger.info(f"Total PnL: ${total_pnl:.2f}")
            logger.info(f"Win Rate: {win_count}/{len(self.trades)} ({100*win_count/len(self.trades):.1f}%)")
            logger.info(f"Avg PnL: ${avg_pnl:.2f}")
        
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description='ADAN Paper Trading V2')
    parser.add_argument('--config', default='config/config.yaml', help='Config path')
    parser.add_argument('--model', default='checkpoints_final/adan_model_checkpoint_640000_steps.zip', help='Model path')
    parser.add_argument('--api-key', required=True, help='API key')
    parser.add_argument('--api-secret', required=True, help='API secret')
    parser.add_argument('--capital', type=float, default=29.0, help='Initial capital')
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading symbol')
    parser.add_argument('--interval', type=int, default=60, help='Trading interval (seconds)')
    
    args = parser.parse_args()
    
    bot = PaperTradingBot(
        config_path=args.config,
        model_path=args.model,
        api_key=args.api_key,
        api_secret=args.api_secret,
        capital=args.capital
    )
    
    bot.run_loop(symbol=args.symbol, interval=args.interval)


if __name__ == '__main__':
    main()
