#!/usr/bin/env python3
"""Test diagnostic pour identifier les clés disponibles dans info dict"""
import numpy as np
import yaml
from src.adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

# Force trade très agressif
config["trading_rules"]["frequency"]["force_trade_steps"] = {"5m": 10, "1h": 15, "4h": 25}

env = MultiAssetChunkedEnv(config=config)
obs = env.reset()
np.random.seed(42)

print("=== TEST DIAGNOSTIC INFO DICT ===")

# 1. Faire un step avec action=0
print("\n[Step 0] Action=0")
result = env.step(np.zeros(env.action_space.shape))
info = result[3] if len(result) == 4 else result[-1]
trade_keys = [k for k in info.keys() if 'trade' in k.lower() or 'position' in k.lower()]
print(f"Clés trade-related: {trade_keys}")

# 2. Faire un step avec action forte
print("\n[Step 1] Action=0.6 (forte)")
action = np.zeros(env.action_space.shape)
action[0] = 0.6
result = env.step(action)
info = result[3] if len(result) == 4 else result[-1]
print(f"trades: {info.get('trades')}")
print(f"valid_trades: {info.get('valid_trades')}")
print(f"num_positions: {info.get('num_positions')}")
print(f"positions (raw): {info.get('positions')}")

# 3. Faire 20 steps pour atteindre force trade
print("\n[Steps 2-22] Attendre force trade...")
for i in range(20):
    result = env.step(np.zeros(env.action_space.shape))
    
info = result[3] if len(result) == 4 else result[-1]
print(f"Après 22 steps:")
print(f"  trades: {info.get('trades')}")
print(f"  valid_trades: {info.get('valid_trades')}")
print(f"  num_positions: {info.get('num_positions')}")
