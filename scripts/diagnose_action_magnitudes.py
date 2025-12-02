#!/usr/bin/env python3
"""
Action Magnitude Diagnostic Script
Tracks model action outputs to diagnose why natural trades aren't happening
"""

import sys
import numpy as np
sys.path.insert(0, 'src')

from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.environment.realistic_trading_env import RealisticTradingEnv

def diagnose_action_magnitudes(duration_steps=100):
    """Run environment and track action magnitudes."""
    print("="*70)
    print("ACTION MAGNITUDE DIAGNOSTIC")
    print("="*70)
    
    config_loader = ConfigLoader()
    config = config_loader.load_config('config/config.yaml')
    
    worker_config = config['workers']['w1']
    if 'environment' not in worker_config:
        worker_config['environment'] = config.get('environment', {})
    if 'trading_rules' not in worker_config:
        worker_config['trading_rules'] = config.get('trading_rules', {})
    
    env = RealisticTradingEnv(config=config, worker_config=worker_config)
    obs, info = env.reset()
    
    action_magnitudes = []
    natural_trades = 0
    force_trades = 0
    
    print(f"\nRunning {duration_steps} steps...")
    print(f"Action space: {env.action_space.shape}")
    print(f"Action thresholds from config:")
    print(f"  5m: 0.002")
    print(f"  1h: 0.003")
    print(f"  4h: 0.005")
    print()
    
    for step in range(duration_steps):
        action = env.action_space.sample()
        action_magnitudes.append(np.max(np.abs(action)))
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 25 == 0:
            max_mag = np.max(np.abs(action))
            print(f"Step {step:3d}: Max action magnitude = {max_mag:.4f}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    # Analysis
    action_magnitudes = np.array(action_magnitudes)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total steps: {duration_steps}")
    print(f"Natural trades: {info.get('frequency', {}).get('counts', {}).get('daily_total', 0) - force_trades}")
    print(f"\nAction Magnitude Statistics:")
    print(f"  Mean: {np.mean(action_magnitudes):.4f}")
    print(f"  Std:  {np.std(action_magnitudes):.4f}")
    print(f"  Max:  {np.max(action_magnitudes):.4f}")
    print(f"  Min:  {np.min(action_magnitudes):.4f}")
    print(f"\nExceeding Thresholds:")
    print(f"  > 0.002 (5m):  {np.sum(action_magnitudes > 0.002)} / {duration_steps} ({100*np.sum(action_magnitudes > 0.002)/duration_steps:.1f}%)")
    print(f"  > 0.003 (1h):  {np.sum(action_magnitudes > 0.003)} / {duration_steps} ({100*np.sum(action_magnitudes > 0.003)/duration_steps:.1f}%)")
    print(f"  > 0.005 (4h):  {np.sum(action_magnitudes > 0.005)} / {duration_steps} ({100*np.sum(action_magnitudes > 0.005)/duration_steps:.1f}%)")
    
    print("\n" + "="*70)
    print("✓ Diagnostic complete")
    print("="*70)

if __name__ == "__main__":
    diagnose_action_magnitudes(100)
