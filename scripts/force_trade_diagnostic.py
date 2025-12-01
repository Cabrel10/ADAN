#!/usr/bin/env python3
"""
Diagnostic rapide pour _force_trade
===================================
Active le niveau DEBUG et exécute seulement jusqu'au premier force trade
"""

import logging
import numpy as np
import sys

# Add src to path
sys.path.insert(0, 'src')
from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.environment.realistic_trading_env import RealisticTradingEnv

# Configure logging au niveau DEBUG
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('force_trade_debug.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Load config
config_loader = ConfigLoader()
config = config_loader.load_config('config/config.yaml')

# Select worker w1
worker_id = 'w1'
worker_config = config['workers'][worker_id]

logger.info(f"Testing with worker: {worker_id}")
logger.info(f"Daily Max Forced Trades: {worker_config.get('trading_rules', {}).get('daily_max_forced_trades', 'NOT SET')}")

# Create environment
env = RealisticTradingEnv(
    config=config,
    worker_config=worker_config,
    worker_id=1,
    enable_market_friction=True,
    min_hold_steps=2,
    daily_trade_limit=50,
    cooldown_steps=3
)

obs, _ = env.reset()

# Run only until force trade is triggered
max_steps = 100
for step in range(max_steps):
    # Random actions
    action = np.random.uniform(-0.005, 0.005, env.action_space.shape)
    
    obs, reward, done, truncated, info = env.step(action)
    
    # Check if force trade was attempted
    if step > 20:  # After force_after threshold for 5m (15 steps)
        logger.info(f"Step {step}: Checking for force trade logs...")
        break
    
    if done:
        obs, _ = env.reset()

logger.info("=" * 50)
logger.info("TEST COMPLETE - Check force_trade_debug.log for DEBUG_OPTUNA messages")
logger.info("=" * 50)
