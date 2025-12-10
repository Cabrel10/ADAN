#!/usr/bin/env python3
"""Diagnostic script to verify that observations change during evaluation."""
import sys
import logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adan_trading_bot.environment.multi_asset_chunked_env import (
    MultiAssetChunkedEnv,
)
from adan_trading_bot.common.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_observations():
    """Check if observations change during evaluation."""

    # Load config
    config_loader = ConfigLoader()
    base_config = config_loader.load_config("config/config.yaml")
    
    # Create env
    env = MultiAssetChunkedEnv(config=base_config)
    
    logger.info("🔍 DIAGNOSTIC: Checking observation changes\n")
    
    # Reset
    obs, _ = env.reset()
    logger.info(f"Initial observation keys: {obs.keys()}")
    
    # Store first observation
    first_obs = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in obs.items()}
    
    # Track changes
    changes = {k: 0 for k in obs.keys()}
    identical_steps = 0
    
    # Run 500 steps
    for step in range(500):
        # Random action
        action = env.action_space.sample()
        
        # Step
        obs, reward, done, truncated, info = env.step(action)
        
        # Check changes
        current_identical = True
        for key in obs.keys():
            if key == "portfolio_state":
                continue
            
            if not np.allclose(obs[key], first_obs[key], atol=1e-5):
                changes[key] += 1
                current_identical = False
        
        if current_identical:
            identical_steps += 1
        
        if step % 20 == 0:
            logger.info(f"Step {step}: Portfolio Value={info.get('portfolio_value', 0):.2f}, "
                       f"Trades={info.get('total_trades', 0)}, "
                       f"Identical to first={current_identical}")
        
        if done or truncated:
            logger.info(f"Episode ended at step {step}")
            break
    
    logger.info(f"\n📊 RESULTS:")
    logger.info(f"  Identical steps: {identical_steps}/100")
    logger.info(f"  Changes per observation key:")
    for key, count in changes.items():
        logger.info(f"    {key}: {count} changes")
    
    if identical_steps > 50:
        logger.error("❌ OBSERVATIONS ARE STATIC! More than 50% identical steps")
        return False
    else:
        logger.info("✅ OBSERVATIONS ARE CHANGING")
        return True

if __name__ == "__main__":
    success = diagnose_observations()
    sys.exit(0 if success else 1)
