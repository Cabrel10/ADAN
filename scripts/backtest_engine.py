#!/usr/bin/env python3
"""
Backtest Engine - Production Version
Runs backtest using Real Environment and Ensemble Logic.
"""

import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from stable_baselines3 import PPO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """Runs backtest with real environment and ensemble"""
    
    def __init__(self, config_path="config/config.yaml"):
        self.config = ConfigLoader.load_config(config_path)
        self.output_dir = Path("/mnt/new_data/t10_training/phase2_results")
        self.checkpoint_dir = Path("/mnt/new_data/t10_training/checkpoints")
        
    def log_section(self, title):
        logger.info("=" * 80)
        logger.info(f"📈 {title}")
        logger.info("=" * 80)

    def load_ensemble_config(self):
        path = self.output_dir / "adan_ensemble_config.json"
        if not path.exists():
            logger.error("❌ Ensemble config not found")
            return None
        with open(path, 'r') as f:
            return json.load(f)

    def load_worker_models(self, workers):
        models = {}
        for w in workers:
            try:
                # Find checkpoint
                cp_dir = self.checkpoint_dir / w
                cps = list(cp_dir.glob(f"{w}_model_*.zip"))
                if not cps: continue
                latest = max(cps, key=lambda p: p.stat().st_mtime)
                
                # Load model on CPU
                models[w] = PPO.load(latest, device='cpu')
                logger.info(f"   Loaded {w} from {latest.name}")
            except Exception as e:
                logger.error(f"❌ Failed to load {w}: {e}")
        return models

    def get_ensemble_action(self, models, obs, weights):
        """Get action from ensemble voting"""
        # Collect predictions
        actions = {}
        for w, model in models.items():
            action, _ = model.predict(obs, deterministic=True)
            actions[w] = action
            
        # Weighted Vote (Assuming continuous action space or discrete?)
        # PPO usually outputs continuous actions for trading (e.g. position size)
        # We will do a weighted average of the actions.
        
        final_action = np.zeros_like(list(actions.values())[0])
        total_weight = 0
        
        for w, action in actions.items():
            weight = weights.get(w, 0.25)
            final_action += action * weight
            total_weight += weight
            
        if total_weight > 0:
            final_action /= total_weight
            
        return final_action

    def run_backtest(self):
        self.log_section("BACKTESTING ADAN ENSEMBLE (PRODUCTION)")
        
        # 1. Load Configs
        ensemble_config = self.load_ensemble_config()
        if not ensemble_config: return False
        
        workers = ensemble_config['workers']
        weights = ensemble_config['weights']
        
        # 2. Load Models
        models = self.load_worker_models(workers)
        if len(models) < len(workers):
            logger.warning(f"⚠️  Only loaded {len(models)}/{len(workers)} models")
            
        # 3. Setup Environment
        # Use a specific backtest config (e.g., test data split)
        backtest_config = self.config.copy()
        # Ensure we use test data if available, or just run on default for now
        # backtest_config['data_split'] = 'test' 
        
        env = MultiAssetChunkedEnv(config=backtest_config)
        
        # 4. Run Simulation
        obs, _ = env.reset() # Unpack tuple (obs, info)
        done = False
        total_reward = 0
        steps = 0
        max_steps = 2000 # Limit for demo/safety, increase for full backtest
        
        logger.info(f"   Running backtest for {max_steps} steps...")
        
        while not done and steps < max_steps:
            action = self.get_ensemble_action(models, obs, weights)
            step_result = env.step(action)
            
            # Handle Gym API (4 or 5 values)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
                
            total_reward += reward
            steps += 1
            
            if steps % 100 == 0:
                logger.info(f"   Step {steps}: Reward {reward:.4f}")
            
            # Capture results immediately if done (before any potential auto-reset)
            if done:
                if hasattr(env, 'envs'):
                    real_env_inner = env.envs[0]
                else:
                    real_env_inner = env
                self.captured_trades = list(real_env_inner.portfolio_manager.trade_log)
                self.captured_value = real_env_inner.portfolio_manager.portfolio_value
                logger.info(f"✅ Captured {len(self.captured_trades)} trades at episode end. Value: {self.captured_value}")
                
        # 5. Collect Results
        if hasattr(env, 'envs'):
            real_env = env.envs[0]
        else:
            real_env = env
        portfolio = real_env.portfolio_manager

        if hasattr(self, 'captured_trades'):
            trades = self.captured_trades
            final_value = self.captured_value
        else:
            trades = list(portfolio.trade_log)
            final_value = portfolio.portfolio_value
        
        # 6. Generate Report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_steps': steps,
            'total_reward': float(total_reward),
            'trades_count': len(trades),
            'final_balance': float(final_value),
            'return_pct': ((final_value - portfolio.initial_capital) / portfolio.initial_capital) * 100,
            'trades': trades
        }
        
        report_path = self.output_dir / "backtest_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"✅ Backtest complete. Report: {report_path}")
        logger.info(f"   Trades: {len(trades)} | Reward: {total_reward:.2f}")
        
        return True

def main():
    engine = BacktestEngine()
    success = engine.run_backtest()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
