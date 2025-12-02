#!/usr/bin/env python3
"""
Live Debug Natural Trading - Real-Time Logging
==============================================
Comprehensive logging of every action, filter, and execution step.
NO file redirection - everything to stdout for live viewing.
"""

import logging
import sys
import time
import numpy as np

sys.path.insert(0, 'src')
from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.environment.realistic_trading_env import RealisticTradingEnv

# Force ALL loggers to DEBUG and stdout only
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

# Set ALL adan_trading_bot loggers to DEBUG
for logger_name in logging.root.manager.loggerDict:
    if 'adan_trading_bot' in logger_name:
        logging.getLogger(logger_name).setLevel(logging.DEBUG)

logger = logging.getLogger("LiveDebug")
logger.setLevel(logging.DEBUG)

def run_live_debug(duration_seconds=30):
    logger.info("="*80)
    logger.info(f"🔴 LIVE DEBUG SESSION START - {duration_seconds}s")
    logger.info("="*80)
    
    # Load Config
    config_loader = ConfigLoader()
    config = config_loader.load_config('config/config.yaml')
    
    # DISABLE force trades
    if 'workers' in config:
        for w in config['workers']:
            config['workers'][w]['trading_rules']['frequency']['daily_max_forced_trades'] = 0
            logger.info(f"✓ Disabled force trades for {w}")
    
    # Get worker config
    worker_config = config['workers']['w1']
    if 'environment' not in worker_config:
        worker_config['environment'] = config.get('environment', {})
    if 'trading_rules' not in worker_config:
        worker_config['trading_rules'] = config.get('trading_rules', {})
    
    logger.info("="*80)
    logger.info("🏗️  INITIALIZING ENVIRONMENT")
    logger.info("="*80)
    
    env = RealisticTradingEnv(config=config, worker_config=worker_config)
    
    logger.info("="*80)
    logger.info("🔄 RESETTING ENVIRONMENT")
    logger.info("="*80)
    
    obs, info = env.reset()
    
    logger.info("="*80)
    logger.info("✅ ENVIRONMENT READY - STARTING TRADING LOOP")
    logger.info("="*80)
    
    start_time = time.time()
    steps = 0
    trades = 0
    
    while time.time() - start_time < duration_seconds:
        steps += 1
        
        # Generate STRONG action every 10 steps to force trade attempts
        # Updated for 25-dim action space (5 assets * 5 dims: Action, Size, TF, SL, TP)
        if steps % 10 == 0:
            # Strong BUY for Asset 0
            # [Action=0.95, Size=0.5, TF=0.8(4h), SL=-0.5(Tight), TP=0.5(Medium)]
            action = np.zeros(25, dtype=np.float32)
            action[0:5] = [0.95, 0.5, 0.8, -0.5, 0.5] 
            action_type = "STRONG_BUY"
        elif steps % 15 == 0:
            # Strong SELL for Asset 0
            # [Action=-0.95, Size=0.5, TF=0.8(4h), SL=-0.5, TP=0.5]
            action = np.zeros(25, dtype=np.float32)
            action[0:5] = [-0.95, 0.5, 0.8, -0.5, 0.5]
            action_type = "STRONG_SELL"
        else:
            action = env.action_space.sample()
            action_type = "RANDOM"
        
        # LOG PRE-STEP STATE
        logger.info("━"*80)
        logger.info(f"📍 STEP {steps}/{int(duration_seconds*30)} | Type: {action_type}")
        logger.info(f"   Action Generated: [{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}]")
        logger.info(f"   Current Step: {env.current_step}")
        logger.info(f"   Daily Total Trades: {env.positions_count.get('daily_total', 0)}")
        
        # Check FreqController status
        can_trade, reason = env.freq_controller.can_open_trade(
            asset='BTCUSDT',
            current_step=env.current_step,
            check_daily=True,
            check_asset=True,
            check_global=True
        )
        logger.info(f"   FreqController.can_open_trade: {can_trade} | Reason: {reason}")
        logger.info(f"   FreqController.daily_trade_count: {env.freq_controller.daily_trade_count}")
        logger.info(f"   FreqController.asset_last_trade: {env.freq_controller.asset_last_trade}")
        
        # Check positions
        positions = env.portfolio_manager.positions
        logger.info(f"   Open Positions: {len([p for p in positions.values() if p.is_open])}")
        for asset, pos in positions.items():
            if pos.is_open:
                logger.info(f"      → {asset}: Size={pos.size:.6f}, Entry={pos.entry_price:.2f}")
        
        # EXECUTE STEP
        logger.info("   ⚙️  Executing env.step()...")
        obs, reward, terminated, truncated, info = env.step(action)
        
        # LOG POST-STEP STATE
        new_daily_total = env.positions_count.get('daily_total', 0)
        trades_this_step = info.get('trades_executed', 0)
        
        if trades_this_step > 0:
            trades += trades_this_step
            logger.info(f"   🎉 TRADE EXECUTED! Total natural trades: {trades}")
        else:
            logger.info(f"   ⭕ No trade executed")
        
        logger.info(f"   Reward: {reward:.4f}")
        logger.info(f"   Info: {info}")
        
        if terminated or truncated:
            logger.info(f"   🔁 Episode ended ({'terminated' if terminated else 'truncated'})")
            obs, info = env.reset()
        
        # Small delay for readability
        time.sleep(0.1)
    
    logger.info("="*80)
    logger.info("🏁 SESSION COMPLETE")
    logger.info(f"   Total Steps: {steps}")
    logger.info(f"   Natural Trades: {trades}")
    logger.info(f"   Duration: {time.time() - start_time:.2f}s")
    logger.info("="*80)
    
    return trades

if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    result = run_live_debug(duration)
    sys.exit(0 if result > 0 else 1)
