#!/usr/bin/env python3
"""
ADAN Ensemble Manager - True Multi-Model Voting System
Loads 4 independent PPO models and implements voting strategies.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from stable_baselines3 import PPO
import logging

logger = logging.getLogger(__name__)


class EnsembleManager:
    """
    Manages 4 independent PPO models and implements voting strategies.
    
    Strategies:
        - median: Robust to outliers (default)
        - mean: Simple average
        - best_sharpe: Dynamic switching based on recent performance
    """
    
    def __init__(self, models_dir: str = "checkpoints_final/final", strategy: str = "median"):
        """
        Initialize ensemble by loading 4 models.
        
        Args:
            models_dir: Directory containing w1_final.zip, w2_final.zip, w3_final.zip, w4_final.zip
            strategy: Voting strategy (median, mean, best_sharpe)
        """
        self.models_dir = Path(models_dir)
        self.strategy = strategy
        self.models = {}
        self.worker_ids = ["w1", "w2", "w3", "w4"]
        
        # Performance tracking for dynamic strategies
        self.performance_scores = {wid: 0.0 for wid in self.worker_ids}
        self.recent_rewards = {wid: [] for wid in self.worker_ids}
        
        self._load_models()
        logger.info(f"✅ Ensemble initialized with {len(self.models)} models (strategy: {strategy})")
    
    def _load_models(self):
        """Load all 4 worker models from disk."""
        for worker_id in self.worker_ids:
            model_path = self.models_dir / f"{worker_id}_final.zip"
            
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found: {model_path}\n"
                    f"Make sure you've trained all 4 workers first."
                )
            
            try:
                model = PPO.load(str(model_path), env=None)
                self.models[worker_id] = model
                logger.info(f"  ✅ Loaded {worker_id} from {model_path.name}")
            except Exception as e:
                raise RuntimeError(f"Failed to load {worker_id}: {e}")
        
        logger.info(f"✅ All {len(self.models)} models loaded successfully")
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Get action from ensemble by having all 4 models vote.
        
        Args:
            observation: Environment observation
            deterministic: Use deterministic policy
        
        Returns:
            (final_action, debug_info)
        """
        predictions = {}
        
        # Each model predicts
        for worker_id, model in self.models.items():
            action, _states = model.predict(observation, deterministic=deterministic)
            predictions[worker_id] = action[0] if action.ndim > 0 else action
        
        # Aggregate votes based on strategy
        if self.strategy == "median":
            final_action = np.median(list(predictions.values()))
        
        elif self.strategy == "mean":
            final_action = np.mean(list(predictions.values()))
        
        elif self.strategy == "best_sharpe":
            # Choose action from best-performing worker
            best_worker = max(self.performance_scores, key=self.performance_scores.get)
            final_action = predictions[best_worker]
        
        else:
            logger.warning(f"Unknown strategy '{self.strategy}', falling back to median")
            final_action = np.median(list(predictions.values()))
        
        debug_info = {
            "predictions": predictions,
            "final_action": final_action,
            "strategy": self.strategy
        }
        
        return final_action, debug_info
    
    def update_performance(self, worker_id: str, reward: float):
        """
        Update performance tracking for a worker.
        Used by dynamic strategies like best_sharpe.
        """
        if worker_id in self.recent_rewards:
            self.recent_rewards[worker_id].append(reward)
            
            # Keep only recent history (last 100 steps)
            if len(self.recent_rewards[worker_id]) > 100:
                self.recent_rewards[worker_id].pop(0)
            
            # Update Sharpe-like score
            rewards = np.array(self.recent_rewards[worker_id])
            if len(rewards) > 10:
                mean_reward = rewards.mean()
                std_reward = rewards.std()
                self.performance_scores[worker_id] = mean_reward / (std_reward + 1e-8)


# ============================================================================
# STANDALONE EVALUATION SCRIPT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ADAN Ensemble Inference")
    parser.add_argument("--models_dir", type=str, default="checkpoints_final/final",
                       help="Directory containing w1-w4 models")
    parser.add_argument("--strategy", type=str, default="median",
                       choices=["median", "mean", "best_sharpe"],
                       help="Voting strategy")
    parser.add_argument("--episodes", type=int, default=1,
                       help="Number of evaluation episodes")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    # Load ensemble
    try:
        ensemble = EnsembleManager(models_dir=args.models_dir, strategy=args.strategy)
    except Exception as e:
        logger.error(f"Failed to initialize ensemble: {e}")
        sys.exit(1)
    
    # Create evaluation environment
    from adan_trading_bot.common.config_loader import ConfigLoader
    from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
    
    config = ConfigLoader().load_config("config/config.yaml")
    env = MultiAssetChunkedEnv(config=config, worker_id=0, log_level="INFO")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🤖 ADAN ENSEMBLE EVALUATION")
    logger.info(f"{'='*80}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Episodes: {args.episodes}\n")
    
    for episode in range(args.episodes):
        obs, info = env.reset()
        done = False
        step = 0
        episode_reward = 0
        
        logger.info(f"\n📊 Episode {episode + 1}/{args.episodes}")
        
        while not done:
            step += 1
            
            # Ensemble prediction
            action, debug = ensemble.predict(obs, deterministic=True)
            
            # Execute
            obs, reward, terminated, truncated, info = env.step([action])
            done = terminated or truncated
            episode_reward += reward
            
            # Log periodically
            if step % 100 == 0:
                equity = env.portfolio_manager.equity
                logger.info(f"  Step {step:4d} | Equity: ${equity:,.2f} | Reward: {reward:+.4f}")
        
        final_equity = env.portfolio_manager.equity
        initial_capital = config["portfolio"]["initial_balance"]
        total_return = (final_equity - initial_capital) / initial_capital * 100
        
        logger.info(f"  ✅ Episode complete | Final Equity: ${final_equity:.2f} | Return: {total_return:+.2f}%")
    
    env.close()
    logger.info(f"\n{'='*80}")
    logger.info("✅ Ensemble evaluation complete")
    logger.info(f"{'='*80}")
