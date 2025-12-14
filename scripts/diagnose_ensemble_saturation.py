#!/usr/bin/env python3
"""
Diagnostic script to investigate ensemble action saturation.
Tests if all workers are returning exactly 1.0 (BUY) or if there's variation.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from stable_baselines3 import PPO
from adan_trading_bot.normalization import ObservationNormalizer

def diagnose_ensemble_saturation():
    """Diagnose why all workers return 1.0"""
    
    print("🔍 ENSEMBLE ACTION SATURATION DIAGNOSTIC")
    print("=" * 60)
    
    # Load config
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print("❌ Config not found")
        return
    
    # Load models
    checkpoint_dir = Path("/mnt/new_data/t10_training/checkpoints/final")
    models = {}
    
    for wid in ['w1', 'w2', 'w3', 'w4']:
        model_path = checkpoint_dir / f"{wid}_final.zip"
        if model_path.exists():
            try:
                models[wid] = PPO.load(model_path)
                print(f"✅ Loaded {wid}")
            except Exception as e:
                print(f"❌ Failed to load {wid}: {e}")
        else:
            print(f"❌ Model not found: {model_path}")
    
    if not models:
        print("❌ No models loaded")
        return
    
    print("\n📊 MODEL ARCHITECTURE ANALYSIS")
    print("-" * 60)
    
    for wid, model in models.items():
        print(f"\n{wid}:")
        print(f"  Policy: {model.policy}")
        print(f"  Action space: {model.action_space}")
        print(f"  Observation space: {model.observation_space}")
        
        # Check policy network
        if hasattr(model.policy, 'action_net'):
            print(f"  Action net: {model.policy.action_net}")
        if hasattr(model.policy, 'net_arch'):
            print(f"  Net arch: {model.policy.net_arch}")
    
    print("\n🧪 ACTION OUTPUT TEST")
    print("-" * 60)
    
    # Create dummy observation
    normalizer = ObservationNormalizer()
    
    # Create a simple observation dict
    dummy_obs = {
        '5m': np.random.randn(1, 20, 14).astype(np.float32),
        '1h': np.random.randn(1, 20, 14).astype(np.float32),
        '4h': np.random.randn(1, 20, 14).astype(np.float32),
        'portfolio_state': np.random.randn(1, 20).astype(np.float32)
    }
    
    print("\nTesting with random observations:")
    print(f"  5m shape: {dummy_obs['5m'].shape}")
    print(f"  1h shape: {dummy_obs['1h'].shape}")
    print(f"  4h shape: {dummy_obs['4h'].shape}")
    print(f"  portfolio_state shape: {dummy_obs['portfolio_state'].shape}")
    
    actions_raw = {}
    actions_first = {}
    
    for wid, model in models.items():
        try:
            # Test with deterministic=True
            action, _states = model.predict(dummy_obs, deterministic=True)
            actions_raw[wid] = action
            actions_first[wid] = float(action[0]) if hasattr(action, '__len__') else float(action)
            
            print(f"\n{wid}:")
            print(f"  Raw action shape: {action.shape if hasattr(action, 'shape') else 'scalar'}")
            print(f"  Raw action: {action}")
            print(f"  action[0]: {actions_first[wid]:.6f}")
            print(f"  Min: {action.min():.6f}, Max: {action.max():.6f}, Mean: {action.mean():.6f}")
            
        except Exception as e:
            print(f"\n{wid}: ❌ Error: {e}")
    
    print("\n📈 SATURATION ANALYSIS")
    print("-" * 60)
    
    if actions_first:
        values = list(actions_first.values())
        print(f"Action[0] values: {[f'{v:.6f}' for v in values]}")
        print(f"All equal to 1.0? {all(abs(v - 1.0) < 0.0001 for v in values)}")
        print(f"All equal to -1.0? {all(abs(v + 1.0) < 0.0001 for v in values)}")
        print(f"Variance: {np.var(values):.6f}")
        print(f"Std Dev: {np.std(values):.6f}")
        
        if np.var(values) < 0.0001:
            print("\n⚠️  SATURATION DETECTED!")
            print("   All workers are returning the same value")
            print("   This indicates overfitting or model collapse")
        else:
            print("\n✅ Diversity detected")
            print("   Workers are returning different values")
    
    print("\n🔧 CONFIGURATION CHECK")
    print("-" * 60)
    
    import yaml
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"deterministic_inference: {config['agent'].get('deterministic_inference', 'NOT SET')}")
    print(f"action_scale: {config['action'].get('action_scale', 'NOT SET')}")
    print(f"min_action_threshold: {config['action'].get('min_action_threshold', 'NOT SET')}")
    
    print("\n💡 RECOMMENDATIONS")
    print("-" * 60)
    
    if np.var(values) < 0.0001:
        print("1. Models are saturated - likely overfitted")
        print("2. Check training data distribution")
        print("3. Consider retraining with:")
        print("   - Better regularization (higher dropout)")
        print("   - More diverse training data")
        print("   - Exploration noise during training")
        print("4. For immediate fix:")
        print("   - Add exploration noise in inference")
        print("   - Use stochastic policy (deterministic_inference: false)")
        print("   - Add action clipping to prevent saturation")

if __name__ == "__main__":
    diagnose_ensemble_saturation()
