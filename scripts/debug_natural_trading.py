#!/usr/bin/env python3
"""
Debug Natural Trading - Progressive Diagnostic Script
=====================================================
Runs the environment in 'Natural Mode' (Force Trades DISABLED)
Logs every step of the action pipeline to find where trades are blocked.
"""

import logging
import sys
import time
import numpy as np
from pathlib import Path

# Setup paths
sys.path.insert(0, 'src')
from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.environment.realistic_trading_env import RealisticTradingEnv

# Configure Logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug_natural.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DebugNatural")

def run_diagnostic(duration_seconds=10):
    logger.info(f"🚀 STARTING DIAGNOSTIC RUN ({duration_seconds}s)")
    
    # Load Config
    config_loader = ConfigLoader()
    config = config_loader.load_config('config/config.yaml')
    
    # DISABLE FORCE TRADES for this test
    if 'workers' in config:
        for w in config['workers']:
            config['workers'][w]['trading_rules']['frequency']['daily_max_forced_trades'] = 0
            logger.info(f"Disabled force trades for {w}")
            
    # Initialize Environment (Worker 1 config merged with full config)
    worker_config = config['workers']['w1']
    # Ensure environment config is present in worker config
    if 'environment' not in worker_config:
        worker_config['environment'] = config.get('environment', {})
    if 'trading_rules' not in worker_config:
        worker_config['trading_rules'] = config.get('trading_rules', {})
        
    env = RealisticTradingEnv(config=config, worker_config=worker_config)
    
    obs, info = env.reset()
    logger.info("✅ Environment Reset Complete")
    
    start_time = time.time()
    steps = 0
    trades = 0
    
    while time.time() - start_time < duration_seconds:
        steps += 1
        
        # Generate a STRONG action to provoke a trade
        # Alternating Buy/Sell to test both sides
        if steps % 20 == 0:
            action = np.array([0.9, 0.5, 0.8]) # Strong BUY
            action_type = "BUY"
        elif steps % 40 == 0:
             action = np.array([-0.9, 0.5, 0.8]) # Strong SELL
             action_type = "SELL"
        else:
            action = env.action_space.sample() # Random
            action_type = "RND"
            
        # Log Pre-Step State
        can_trade, reason = env.freq_controller.can_open_trade(
            asset='BTCUSDT', 
            current_step=env.current_step,
            check_daily=True
        )
        
        if abs(action[0]) > 0.01:
            logger.debug(f"\n[STEP {steps}] Attempting {action_type} | Action: {action[0]:.3f}")
            logger.debug(f"   FreqController: CanTrade={can_trade} Reason={reason}")
            logger.debug(f"   DailyTotal: {env.positions_count.get('daily_total', 0)}")
        
        # Execute Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check if trade happened
        if info.get('trades_executed', 0) > 0:
            trades += 1
            logger.info(f"🎉 TRADE EXECUTED at Step {steps}!")
            
        if terminated or truncated:
            obs, info = env.reset()
            
    logger.info("="*50)
    logger.info(f"🏁 DIAGNOSTIC COMPLETE")
    logger.info(f"Duration: {time.time() - start_time:.2f}s")
    logger.info(f"Steps: {steps}")
    logger.info(f"Natural Trades: {trades}")
    logger.info("="*50)

if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_diagnostic(duration)
