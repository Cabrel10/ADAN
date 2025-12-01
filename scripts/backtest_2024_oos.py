#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simplified Out-of-Sample Backtest for ADAN 2.0
Uses existing trained model on 2024+ data
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.environment.realistic_trading_env import RealisticTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", default="models/rl_agents/final/w1_final.zip")
    parser.add_argument("--vecnormalize", default="models/rl_agents/vecnormalize.pkl")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("ADAN 2.0 - Simplified Backtest (Recent Model)")
    logger.info("="*60)
    
    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load_config(args.config)
    
    # Use worker 1 config for backtest
    worker_config = config["workers"]["w1"]
    
    # Create environment factory
    def make_env():
        return RealisticTradingEnv(
            config=config,
            worker_config=worker_config,
            worker_id=0,
            live_mode=False,
            enable_market_friction=False,  # Disable for compatibility
            use_stable_reward=False
        )
    
    # Wrap environment
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize stats
    try:
        env = VecNormalize.load(args.vecnormalize, env)
        env.training = False
        env.norm_reward = False
        logger.info(f"✅ Loaded VecNormalize: {args.vecnormalize}")
    except Exception as e:
        logger.warning(f"⚠️ Could not load VecNormalize: {e}")
        logger.info("Continuing WITHOUT VecNormalize...")
    
    # Load model
    try:
        model = PPO.load(args.model, env=env)
        logger.info(f"✅ Loaded model: {args.model}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return 1
    
    # Run episodes
    logger.info(f"\nRunning {args.episodes} episodes...")
    episode_rewards = []
    
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            steps += 1
            
            if done[0]:
                break
        
        episode_rewards.append(ep_reward)
        logger.info(f"Episode {ep+1}: Reward={ep_reward:.2f}, Steps={steps}")
    
    # Summary
    import numpy as np
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Episodes:      {args.episodes}")
    logger.info(f"Mean Reward:   {np.mean(episode_rewards):.2f}")
    logger.info(f"Std Reward:    {np.std(episode_rewards):.2f}")
    logger.info(f"Min Reward:    {np.min(episode_rewards):.2f}")
    logger.info(f"Max Reward:    {np.max(episode_rewards):.2f}")
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
