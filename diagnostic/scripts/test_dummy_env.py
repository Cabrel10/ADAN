#!/usr/bin/env python3
"""
Checkpoint 2.1 Validation: Test TradingEnvDummy
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("\n" + "="*70)
print("CHECKPOINT 2.1: Testing TradingEnvDummy")
print("="*70)

try:
    print("\n1️⃣  Importing TradingEnvDummy...")
    from src.adan_trading_bot.environment import TradingEnvDummy
    print("   ✅ Import successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

try:
    print("\n2️⃣  Instantiating TradingEnvDummy...")
    env = TradingEnvDummy()
    print("   ✅ Instantiation successful")
except Exception as e:
    print(f"   ❌ Instantiation failed: {e}")
    sys.exit(1)

try:
    print("\n3️⃣  Checking observation_space...")
    print(f"   Type: {type(env.observation_space)}")
    print(f"   Structure: {env.observation_space}")
    
    # Verify it's a Dict space
    from gymnasium import spaces
    assert isinstance(env.observation_space, spaces.Dict), "observation_space must be Dict"
    print("   ✅ observation_space is Dict")
    
    # Verify all required keys
    required_keys = {'5m', '1h', '4h', 'portfolio_state'}
    actual_keys = set(env.observation_space.spaces.keys())
    assert required_keys == actual_keys, f"Keys mismatch: {required_keys} vs {actual_keys}"
    print(f"   ✅ All required keys present: {actual_keys}")
    
    # Verify shapes
    print("\n   Shapes:")
    print(f"     '5m': {env.observation_space['5m'].shape}")
    print(f"     '1h': {env.observation_space['1h'].shape}")
    print(f"     '4h': {env.observation_space['4h'].shape}")
    print(f"     'portfolio_state': {env.observation_space['portfolio_state'].shape}")
    
except Exception as e:
    print(f"   ❌ observation_space check failed: {e}")
    sys.exit(1)

try:
    print("\n4️⃣  Checking action_space...")
    print(f"   Type: {type(env.action_space)}")
    print(f"   Shape: {env.action_space.shape}")
    print(f"   Bounds: [{env.action_space.low[0]}, {env.action_space.high[0]}]")
    
    assert isinstance(env.action_space, spaces.Box), "action_space must be Box"
    assert env.action_space.shape == (25,), f"action_space shape must be (25,), got {env.action_space.shape}"
    print("   ✅ action_space is correct")
    
except Exception as e:
    print(f"   ❌ action_space check failed: {e}")
    sys.exit(1)

try:
    print("\n5️⃣  Testing reset()...")
    obs, info = env.reset()
    print("   ✅ reset() successful")
    
    # Verify observation structure
    assert isinstance(obs, dict), "Observation must be dict"
    assert set(obs.keys()) == required_keys, "Observation keys mismatch"
    print(f"   ✅ Observation keys correct: {set(obs.keys())}")
    
    # Verify shapes
    print("\n   Observation shapes:")
    for key, val in obs.items():
        print(f"     '{key}': {val.shape}")
    
    assert obs['5m'].shape == (20, 15), f"'5m' shape mismatch: {obs['5m'].shape}"
    assert obs['1h'].shape == (10, 15), f"'1h' shape mismatch: {obs['1h'].shape}"
    assert obs['4h'].shape == (5, 15), f"'4h' shape mismatch: {obs['4h'].shape}"
    assert obs['portfolio_state'].shape == (20,), f"'portfolio_state' shape mismatch: {obs['portfolio_state'].shape}"
    print("   ✅ All shapes correct")
    
except Exception as e:
    print(f"   ❌ reset() test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n6️⃣  Testing step() raises NotImplementedError...")
    try:
        env.step(env.action_space.sample())
        print("   ❌ step() should raise NotImplementedError")
        sys.exit(1)
    except NotImplementedError as e:
        print(f"   ✅ step() correctly raises NotImplementedError")
        print(f"      Message: {e}")
    
except Exception as e:
    print(f"   ❌ step() test failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ CHECKPOINT 2.1 VALIDATION COMPLETE")
print("="*70)
print("\nAll tests passed! TradingEnvDummy is ready for use.")
print("\nNext: Checkpoint 2.2 - Validate observation_space coherence")
