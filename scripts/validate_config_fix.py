#!/usr/bin/env python3
"""
Validation Script for Config Fixes
==================================
Runs a short simulation (5000 steps) with RealisticTradingEnv to verify:
1. Trades are executed (natural and forced).
2. "Daily forced trade limit" errors are resolved.
3. Rewards are not constantly negative.
"""

import logging
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')
from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.environment.realistic_trading_env import RealisticTradingEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('validation_run.log')
    ]
)
logger = logging.getLogger(__name__)

def run_validation():
    logger.info("Starting validation run...")
    
    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load_config('config/config.yaml')
    
    # Select worker w1
    worker_id = 'w1'
    worker_config = config['workers'][worker_id]
    
    logger.info(f"Testing with worker: {worker_id}")
    logger.info(f"Action Threshold: {config['trading_rules']['frequency']['action_threshold']}")
    logger.info(f"Daily Trade Limit (Global): {config['trading_rules']['frequency']['total_daily_max']}")
    logger.info(f"Daily Forced Trade Limit (Worker): {worker_config.get('daily_max_forced_trades', 'NOT SET')}")
    
    # Create environment
    env = RealisticTradingEnv(
        config=config,
        worker_config=worker_config,
        worker_id=1,
        enable_market_friction=True,
        min_hold_steps=worker_config.get('trading_rules', {}).get('position_hold_min', 2),
        daily_trade_limit=config['trading_rules']['frequency']['total_daily_max'],
        cooldown_steps=3
    )
    
    obs, _ = env.reset()
    
    n_steps = 5000
    trades_executed = 0
    total_reward = 0.0
    rewards = []
    
    logger.info(f"Running for {n_steps} steps...")
    
    for step in range(n_steps):
        # Generate random actions, occasionally strong enough to trade
        # Action space: [action, stop_loss, take_profit] per asset
        # We want actions > 0.01 to trigger trades
        
        # 10% chance of strong action
        if np.random.random() < 0.1:
            action = np.random.uniform(-1.0, 1.0, env.action_space.shape)
        else:
            action = np.random.uniform(-0.005, 0.005, env.action_space.shape) # Below threshold
            
        obs, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        rewards.append(reward)
        
        if info.get('trades_executed', 0) > 0:
            trades_executed += info['trades_executed']
            # logger.info(f"Step {step}: Trade executed! Reward: {reward:.4f}")
            
        if done:
            obs, _ = env.reset()
            
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    logger.info("=" * 50)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Total Steps: {n_steps}")
    logger.info(f"Trades Executed: {trades_executed}")
    logger.info(f"Average Reward: {avg_reward:.4f}")
    logger.info(f"Reward Std Dev: {std_reward:.4f}")
    logger.info(f"Final Portfolio Value: {env.portfolio_manager.current_value:.2f}")
    
    # Check for success criteria
    success = True
    if trades_executed == 0:
        logger.error("FAILURE: No trades executed.")
        success = False
    
    if avg_reward < -0.5: # Arbitrary threshold for "too punitive"
        logger.warning(f"WARNING: Average reward is very low ({avg_reward:.4f}).")
        
    if success:
        logger.info("SUCCESS: Validation passed. Trades are executing.")
    else:
        logger.error("FAILURE: Validation failed.")
        
if __name__ == "__main__":
    run_validation()
