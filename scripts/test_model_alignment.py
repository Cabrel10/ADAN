#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DIAGNOSTIC TEST: Verify PPO-CNN-LSTM Alignment
Tests actual model architecture vs config expectations
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import numpy as np
from stable_baselines3 import PPO
from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.environment.realistic_trading_env import RealisticTradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv

print("="*60)
print("DIAGNOSTIC TEST: PPO-CNN-LSTM ALIGNMENT")
print("="*60)

# Load config
config_loader = ConfigLoader()
config = config_loader.load_config("config/config.yaml")

# Create test environment
def make_env():
    return RealisticTradingEnv(
        config=config,
        worker_config=config["workers"]["w1"],
        worker_id=0,
        enable_market_friction=False,
        use_stable_reward=False
    )

env = DummyVecEnv([make_env])

# Check observation space
obs = env.reset()
print("\n1. OBSERVATION SPACE CHECK")
print(f"   Type: {type(obs)}")
if isinstance(obs, dict):
    for key, value in obs.items():
        print(f"   {key}: {value.shape} (dtype: {value.dtype})")
        
        # Check for NaN or Inf
        if np.isnan(value).any():
            print(f"      ⚠️ WARNING: NaN detected in {key}")
        if np.isinf(value).any():
            print(f"      ⚠️ WARNING: Inf detected in {key}")
        
        # Check value range
        print(f"      Range: [{value.min():.4f}, {value.max():.4f}]")
else:
    print(f"   Shape: {obs.shape}")

# Load model and check architecture
print("\n2. MODEL ARCHITECTURE CHECK")
try:
    model = PPO.load("models/rl_agents/final/w1_final.zip", env=env)
    print("   ✅ Model loaded successfully")
    
    # Check policy network
    policy = model.policy
    print(f"\n   Policy type: {type(policy).__name__}")
    
    # Check feature extractor
    if hasattr(policy, 'features_extractor'):
        extractor = policy.features_extractor
        print(f"   Feature extractor: {type(extractor).__name__}")
        
        # Print network structure
        print("\n   Network layers:")
        for name, module in extractor.named_modules():
            if len(name) > 0:  # Skip root
                print(f"      {name}: {type(module).__name__}")
    
    # Check if CNN or LSTM is present
    has_cnn = any('Conv' in str(type(m)) for m in policy.modules())
    has_lstm = any('LSTM' in str(type(m)) for m in policy.modules())
    
    print(f"\n   Contains CNN: {has_cnn}")
    print(f"   Contains LSTM: {has_lstm}")
    
    # Test forward pass
    print("\n3. FORWARD PASS TEST")
    with torch.no_grad():
        action, _ = model.predict(obs, deterministic=True)
        print(f"   ✅ Forward pass successful")
        print(f"   Action shape: {action.shape}")
        print(f"   Action range: [{action.min():.4f}, {action.max():.4f}]")
    
    # Check for exploded parameters
    print("\n4. PARAMETER HEALTH CHECK")
    total_params = 0
    exploded_params = 0
    for name, param in model.policy.named_parameters():
        total_params += param.numel()
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"   ⚠️ WARNING: {name} has NaN/Inf values")
            exploded_params += 1
        
        # Check if std is reasonable
        if param.std() > 100:
            print(f"   ⚠️ WARNING: {name} has very high std: {param.std().item():.2f}")
    
    print(f"\n   Total parameters: {total_params:,}")
    print(f"   Exploded parameters: {exploded_params}")
    
    if exploded_params == 0:
        print("   ✅ All parameters healthy")
    else:
        print(f"   ❌ {exploded_params} parameters have issues")

except Exception as e:
    print(f"   ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test with random actions
print("\n5. ENVIRONMENT STEP TEST")
for i in range(5):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"   Step {i+1}: Reward={reward[0]:.4f}, Done={done[0]}")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
