#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DIAGNOSTIC: Pourquoi aucun worker ne trade?
Test empirique pour trouver le blocage
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.environment.realistic_trading_env import RealisticTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

print("="*60)
print("DIAGNOSTIC: Pourquoi aucun trade?")
print("="*60)

config_loader = ConfigLoader()
config = config_loader.load_config("config/config.yaml")

# Test avec worker 2
def make_env():
    return RealisticTradingEnv(
        config=config,
        worker_config=config["workers"]["w2"],
        worker_id=0,
        enable_market_friction=False,
        use_stable_reward=False  # IMPORTANT: reward simple pour diagnostiquer
    )

env = DummyVecEnv([make_env])
model = PPO.load("models/rl_agents/final/w2_final.zip", env=env)

print("\n1. TEST AVEC ACTIONS MODÈLE")
print("-" * 60)
obs = env.reset()

for step in range(10):
    action, _ = model.predict(obs, deterministic=True)
    print(f"Step {step+1}: Action min={action.min():.4f}, max={action.max():.4f}, mean={action.mean():.4f}")
    
    obs, reward, done, info = env.step(action)
    
    if isinstance(info, list) and len(info) > 0:
        trades = info[0].get('trades_executed', 0)
        balance = info[0].get('balance', 0)
        print(f"   → Trades: {trades}, Balance: {balance:.2f}, Reward: {reward[0]:.4f}")
    
    if done[0]:
        print("   → Episode terminé")
        obs = env.reset()

print("\n2. TEST AVEC ACTIONS FORCÉES (STRONG)")
print("-" * 60)
obs = env.reset()

for step in range(5):
    # Action forte: 1.0 = BUY FULL
    forced_action = np.ones(15) * 0.8  # Strong buy signal
    print(f"Step {step+1}: Forced action = 0.8 (STRONG BUY)")
    
    obs, reward, done, info = env.step(forced_action[np.newaxis, :])
    
    if isinstance(info, list) and len(info) > 0:
        trades = info[0].get('trades_executed', 0)
        balance = info[0].get('balance', 0)
        print(f"   → Trades: {trades}, Balance: {balance:.2f}, Reward: {reward[0]:.4f}")
    
    if done[0]:
        print("   → Episode terminé")
        break

print("\n3. VÉRIFICATION DES CONTRAINTES")
print("-" * 60)

# Accéder à l'env réel (pas le wrapper)
real_env = env.envs[0]

print(f"Action threshold: {getattr(real_env, 'action_threshold', 'N/A')}")
print(f"Min notional: ${real_env.min_notional}")
print(f"Daily trade limit: {real_env.freq_controller.config.daily_trade_limit}")
print(f"Cooldown steps: {real_env.freq_controller.config.asset_cooldown_steps}")
print(f"Circuit breaker: {real_env.circuit_breaker_pct:.1%}")

print("\n" + "="*60)
print("DIAGNOSTIC TERMINÉ")
print("="*60)
