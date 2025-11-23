#!/usr/bin/env python3
"""
Diagnostic: Pourquoi le trade_log est vide?
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from stable_baselines3 import PPO
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.common.config_loader import ConfigLoader

logger.info("=" * 80)
logger.info("DIAGNOSTIC: Trade Log Investigation")
logger.info("=" * 80)

# Load
config_loader = ConfigLoader()
config = config_loader.load_config("config/config.yaml")
config['initial_capital'] = 20.5
config['environment']['assets'] = ['XRPUSDT']

logger.info("Création environnement...")
env = MultiAssetChunkedEnv(config=config, worker_id=0, log_level="WARNING")

logger.info("Chargement modèle...")
model = PPO.load("checkpoints_final/adan_model_checkpoint_640000_steps.zip", env=env)

logger.info("Lancement évaluation (100 steps)...")
obs, _ = env.reset()
done = False
step = 0

while not done and step < 100:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    step += 1

logger.info(f"Évaluation terminée: {step} steps")

# Diagnostic
logger.info("=" * 80)
logger.info("DIAGNOSTIC PORTFOLIO MANAGER")
logger.info("=" * 80)

pm = env.portfolio_manager

logger.info(f"Portfolio Manager type: {type(pm)}")
logger.info(f"Portfolio Manager attributes: {dir(pm)}")

# Check trade_log
if hasattr(pm, 'trade_log'):
    logger.info(f"✅ trade_log exists")
    logger.info(f"   Type: {type(pm.trade_log)}")
    logger.info(f"   Length: {len(pm.trade_log)}")
    if pm.trade_log:
        logger.info(f"   First entry: {pm.trade_log[0]}")
else:
    logger.warning("❌ trade_log NOT FOUND")

# Check closed_positions
if hasattr(pm, 'metrics'):
    logger.info(f"✅ metrics exists")
    if hasattr(pm.metrics, 'closed_positions'):
        logger.info(f"   closed_positions: {len(pm.metrics.closed_positions)}")
        if pm.metrics.closed_positions:
            logger.info(f"   First: {pm.metrics.closed_positions[0]}")
    else:
        logger.warning("   ❌ closed_positions NOT FOUND")
else:
    logger.warning("❌ metrics NOT FOUND")

# Check active_positions
if hasattr(pm, 'active_positions'):
    logger.info(f"✅ active_positions exists: {len(pm.active_positions)}")
else:
    logger.warning("❌ active_positions NOT FOUND")

# Check portfolio value
logger.info(f"Portfolio value: ${pm.equity:.2f}")
logger.info(f"Initial capital: ${pm.initial_capital:.2f}")
logger.info(f"Return: {(pm.equity - pm.initial_capital) / pm.initial_capital * 100:.2f}%")

logger.info("=" * 80)
