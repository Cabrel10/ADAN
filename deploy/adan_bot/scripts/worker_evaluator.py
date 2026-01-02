#!/usr/bin/env python3
"""
Worker Evaluator - Production Version
Evaluates trained worker models using real environment and data.
"""

import sys
import json
import logging
import yaml
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.evaluation.decision_quality_analyzer import DecisionQualityAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('worker_evaluator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WorkerEvaluator:
    """Evaluates worker profiles using real models and environment"""
    
    def __init__(self, config_path="config/config.yaml"):
        self.config = ConfigLoader.load_config(config_path)
        self.checkpoint_dir = Path("/mnt/new_data/t10_training/checkpoints")
        self.output_dir = Path("/mnt/new_data/t10_training/phase2_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workers = ['w1', 'w2', 'w3', 'w4']
        
    def log_section(self, title):
        logger.info("=" * 80)
        logger.info(f"📊 {title}")
        logger.info("=" * 80)

    def load_optuna_params(self, worker):
        """Load optimized hyperparameters for the worker"""
        try:
            param_file = Path(f"optuna_results/{worker}_ppo_best_params.yaml")
            if param_file.exists():
                with open(param_file, 'r') as f:
                    return yaml.safe_load(f)
            return {}
        except Exception as e:
            logger.warning(f"Could not load Optuna params for {worker}: {e}")
            return {}

    def evaluate_worker(self, worker):
        logger.info(f"\n🔍 Evaluating {worker.upper()}...")
        
        try:
            # 1. Find latest checkpoint
            checkpoint_path = self.checkpoint_dir / worker
            if not checkpoint_path.exists():
                logger.error(f"❌ Directory not found: {checkpoint_path}")
                return None
                
            checkpoints = list(checkpoint_path.glob(f"{worker}_model_*.zip"))
            if not checkpoints:
                logger.error(f"❌ No checkpoints found for {worker}")
                return None
            
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            steps = int(latest.stem.split('_')[-2])
            logger.info(f"   Checkpoint: {latest.name} ({steps} steps)")

            # 2. Setup Environment
            eval_config = self.config.copy()
            eval_config['worker_id'] = worker
            
            # Inject Optuna params if available
            optuna_params = self.load_optuna_params(worker)
            if optuna_params:
                if 'n_steps' in optuna_params:
                    eval_config['n_steps'] = optuna_params['n_steps']
                if 'batch_size' in optuna_params:
                    eval_config['batch_size'] = optuna_params['batch_size']
            
            env = MultiAssetChunkedEnv(config=eval_config)
            
            # 3. Load Model
            model = PPO.load(latest, env=env, device='cpu')
            
            # 4. Run Evaluation Episode
            obs, _ = env.reset()
            done = False
            total_reward = 0
            
            eval_steps = 1000
            logger.info(f"   Running evaluation for {eval_steps} steps...")
            
            for _ in range(eval_steps):
                action, _ = model.predict(obs, deterministic=True)
                step_result = env.step(action)
                
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result
                
                total_reward += reward
            
            # 5. Extract Real Performance Metrics
            real_env = env.envs[0] if hasattr(env, 'envs') else env
            
            # Get trades from portfolio manager
            trades_data = pd.DataFrame()
            if hasattr(real_env, 'portfolio_manager'):
                portfolio_manager = real_env.portfolio_manager
                if hasattr(portfolio_manager, 'trade_log'):
                    # Filter for closed trades (where PnL is recorded)
                    raw_trades = list(portfolio_manager.trade_log)
                    closed_trades = [t for t in raw_trades if t.get('event') == 'close' or t.get('action') == 'close']
                    if closed_trades:
                        trades_data = pd.DataFrame(closed_trades)
            
            # 6. Analyze Quality (Robust Calculation)
            # Calculate metrics directly from trades to ensure validity
            pnl_series = trades_data['pnl'] if 'pnl' in trades_data.columns else pd.Series([])
            total_trades = len(pnl_series)
            
            trade_metrics = {} # Initialize to avoid NameError
            
            if total_trades > 0:
                winning_trades = pnl_series[pnl_series > 0]
                losing_trades = pnl_series[pnl_series <= 0]
                
                win_rate = len(winning_trades) / total_trades
                total_profit = winning_trades.sum()
                total_loss = abs(losing_trades.sum())
                
                profit_factor = total_profit / total_loss if total_loss > 0 else (10.0 if total_profit > 0 else 0.0)
                avg_pnl = pnl_series.mean()
                
                # Normalize metrics to 0-100 scale
                # Win Rate: 0.5 -> 50, 1.0 -> 100
                score_win_rate = win_rate * 100
                
                # Profit Factor: 1.0 -> 50, 2.0 -> 80, 3.0+ -> 100
                score_pf = min(max((profit_factor - 0.5) * 33.3, 0), 100)
                
                # PnL Consistency (Sharpe-like): Mean / Std
                pnl_std = pnl_series.std()
                sharpe_proxy = avg_pnl / pnl_std if pnl_std > 0 else 0
                score_sharpe = min(max((sharpe_proxy + 1) * 33.3, 0), 100) # -1 -> 0, 0 -> 33, 2 -> 100
                
                # Weighted Score
                quality_score = (score_win_rate * 0.3 + score_pf * 0.4 + score_sharpe * 0.3)
                
                trade_metrics = {
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'sharpe_proxy': sharpe_proxy,
                    'total_trades': total_trades
                }
                
                logger.info(f"   Metrics: WR={win_rate:.2f} PF={profit_factor:.2f} Sharpe={sharpe_proxy:.2f} -> Score={quality_score:.1f}")
            else:
                quality_score = 0.0

            # 7. Calculate Confidence
            confidence_score = self._calculate_confidence(quality_score, steps, trades_data)

            # 8. Report
            evaluation = {
                'worker': worker,
                'checkpoint': str(latest),
                'steps': steps,
                'quality_score': float(quality_score),
                'confidence_score': float(confidence_score),
                'total_reward': float(total_reward),
                'trades_count': len(trades_data) if not trades_data.empty else 0,
                'trade_metrics': trade_metrics,
                'optuna_params': optuna_params,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ {worker}: Quality {quality_score:.1f} | Conf {confidence_score:.2f} | Trades {evaluation['trades_count']} | Reward {total_reward:.2f}")
            
            env.close()
            return evaluation

        except Exception as e:
            logger.error(f"❌ {worker}: Evaluation failed - {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_quality_score(self, trades_df, total_reward):
        """Calculate real quality score based on actual trading performance"""
        trade_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_pnl': 0.0,
            'total_pnl': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0
        }
        
        if trades_df.empty or len(trades_df) == 0:
            logger.warning("   No trades executed - quality score set to 20")
            return 20.0, trade_metrics
        
        # Extract PnL from trades
        if 'pnl' in trades_df.columns:
            pnls = trades_df['pnl'].values
        else:
            logger.warning("   No PnL column found in trades")
            return 30.0, trade_metrics
        
        # Calculate metrics
        total_trades = len(pnls)
        winning_trades = np.sum(pnls > 0)
        losing_trades = np.sum(pnls < 0)
        total_pnl = np.sum(pnls)
        
        trade_metrics['total_trades'] = int(total_trades)
        trade_metrics['winning_trades'] = int(winning_trades)
        trade_metrics['losing_trades'] = int(losing_trades)
        trade_metrics['total_pnl'] = float(total_pnl)
        
        if total_trades > 0:
            win_rate = winning_trades / total_trades
            avg_pnl = total_pnl / total_trades
            trade_metrics['win_rate'] = float(win_rate)
            trade_metrics['avg_pnl'] = float(avg_pnl)
        
        if winning_trades > 0:
            avg_win = np.mean(pnls[pnls > 0])
            trade_metrics['avg_win'] = float(avg_win)
        
        if losing_trades > 0:
            avg_loss = np.mean(pnls[pnls < 0])
            trade_metrics['avg_loss'] = float(avg_loss)
            
            # Profit factor
            if abs(np.sum(pnls[pnls < 0])) > 0:
                profit_factor = np.sum(pnls[pnls > 0]) / abs(np.sum(pnls[pnls < 0]))
                trade_metrics['profit_factor'] = float(profit_factor)
        
        # Calculate quality score (0-100)
        # Components:
        # - Win rate (40%): Higher is better
        # - Profit factor (30%): Higher is better (capped at 2.0)
        # - Trade count (20%): More trades = more data = higher confidence
        # - Avg PnL (10%): Positive is better
        
        win_rate_score = (win_rate * 100) * 0.4  # 0-40
        
        profit_factor = trade_metrics['profit_factor']
        pf_score = min(profit_factor / 2.0, 1.0) * 100 * 0.3  # 0-30
        
        trade_count_score = min(total_trades / 100, 1.0) * 100 * 0.2  # 0-20
        
        avg_pnl_score = (1.0 if avg_pnl > 0 else 0.5) * 100 * 0.1  # 0-10
        
        quality_score = win_rate_score + pf_score + trade_count_score + avg_pnl_score
        quality_score = min(max(quality_score, 0), 100)
        
        logger.info(f"   Trade Metrics: {total_trades} trades, {win_rate:.1%} win rate, PF={profit_factor:.2f}, Avg PnL=${avg_pnl:.4f}")
        
        return quality_score, trade_metrics

    def _calculate_confidence(self, quality_score, steps, trades_df):
        """Calculate confidence score (0.0 - 1.0)"""
        # Base confidence from quality (0-100 -> 0-1)
        score = quality_score / 100.0
        
        # Penalty for low steps (target 350k)
        step_factor = min(steps / 350000, 1.0)
        
        # Penalty for no trades
        trade_factor = 1.0 if len(trades_df) > 5 else 0.5
        if len(trades_df) == 0: trade_factor = 0.1
        
        final_confidence = score * 0.5 + step_factor * 0.3 + trade_factor * 0.2
        return min(max(final_confidence, 0.1), 1.0)

    def run_evaluation(self):
        self.log_section("WORKER PROFILE EVALUATION (PRODUCTION)")
        results = {}
        
        for worker in self.workers:
            res = self.evaluate_worker(worker)
            if res:
                results[worker] = res
                
        # Save results
        with open(self.output_dir / "worker_evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"\n✅ Evaluation complete. Saved to {self.output_dir}")
        return len(results) > 0

def main():
    evaluator = WorkerEvaluator()
    success = evaluator.run_evaluation()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
