#!/usr/bin/env python3
"""
Checkpoint 2.2: Validate observation_space coherence
Compares TradingEnvDummy with expected MultiAssetChunkedEnv structure.
"""

import sys
from pathlib import Path
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("\n" + "="*70)
print("CHECKPOINT 2.2: Validating Observation Space Coherence")
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
    env_dummy = TradingEnvDummy()
    print("   ✅ TradingEnvDummy instantiated")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Expected structure from MultiAssetChunkedEnv (from code analysis)
expected_structure = {
    '5m': {'shape': (20, 15), 'dtype': 'float32'},
    '1h': {'shape': (10, 15), 'dtype': 'float32'},
    '4h': {'shape': (5, 15), 'dtype': 'float32'},
    'portfolio_state': {'shape': (20,), 'dtype': 'float32'},
}

expected_action = {'shape': (25,), 'dtype': 'float32'}

print("\n3️⃣  Comparing observation_space with expected structure...")

dummy_obs_space = env_dummy.observation_space
comparison_results = {
    "timestamp": str(Path(__file__).stat().st_mtime),
    "dummy_env": str(dummy_obs_space),
    "expected_structure": expected_structure,
    "keys_match": True,
    "details": {}
}

# Check keys
dummy_keys = set(dummy_obs_space.spaces.keys())
expected_keys = set(expected_structure.keys())

print(f"\n   Dummy keys:    {dummy_keys}")
print(f"   Expected keys: {expected_keys}")

if dummy_keys != expected_keys:
    print(f"   ❌ Keys mismatch!")
    comparison_results["keys_match"] = False
    sys.exit(1)
else:
    print(f"   ✅ Keys match: {dummy_keys}")

# Compare each space
print("\n4️⃣  Comparing individual spaces...")

all_match = True
for key in sorted(dummy_keys):
    dummy_space = dummy_obs_space[key]
    expected = expected_structure[key]
    
    print(f"\n   Key: '{key}'")
    print(f"     Dummy:    shape={dummy_space.shape}, dtype={dummy_space.dtype}")
    print(f"     Expected: shape={expected['shape']}, dtype={expected['dtype']}")
    
    shape_match = dummy_space.shape == expected['shape']
    dtype_match = str(dummy_space.dtype) == expected['dtype']
    
    if shape_match and dtype_match:
        print(f"     ✅ MATCH")
        comparison_results["details"][key] = {
            "status": "MATCH",
            "dummy_shape": str(dummy_space.shape),
            "expected_shape": str(expected['shape']),
            "dummy_dtype": str(dummy_space.dtype),
            "expected_dtype": expected['dtype'],
        }
    else:
        print(f"     ❌ MISMATCH")
        if not shape_match:
            print(f"        Shape: {dummy_space.shape} vs {expected['shape']}")
        if not dtype_match:
            print(f"        Dtype: {dummy_space.dtype} vs {expected['dtype']}")
        comparison_results["details"][key] = {
            "status": "MISMATCH",
            "dummy_shape": str(dummy_space.shape),
            "expected_shape": str(expected['shape']),
            "dummy_dtype": str(dummy_space.dtype),
            "expected_dtype": expected['dtype'],
        }
        all_match = False

# Compare action spaces
print("\n5️⃣  Comparing action_space...")

dummy_action = env_dummy.action_space

print(f"   Dummy:    shape={dummy_action.shape}, dtype={dummy_action.dtype}")
print(f"   Expected: shape={expected_action['shape']}, dtype={expected_action['dtype']}")

action_match = (dummy_action.shape == expected_action['shape'] and 
                str(dummy_action.dtype) == expected_action['dtype'])

if action_match:
    print(f"   ✅ MATCH")
    comparison_results["action_space"] = "MATCH"
else:
    print(f"   ❌ MISMATCH")
    comparison_results["action_space"] = "MISMATCH"
    all_match = False

# Save results
print("\n6️⃣  Saving validation report...")

results_dir = project_root / "diagnostic" / "results"
results_dir.mkdir(parents=True, exist_ok=True)

report_file = results_dir / "observation_space_validation.json"
with open(report_file, 'w') as f:
    json.dump(comparison_results, f, indent=2)

print(f"   ✅ Report saved to: {report_file}")

# Final verdict
print("\n" + "="*70)
if all_match and action_match:
    print("✅ CHECKPOINT 2.2 VALIDATION COMPLETE - ALL SPACES MATCH")
    print("="*70)
    print("\nObservation spaces are PERFECTLY COHERENT.")
    print("Ready to proceed to Checkpoint 2.3.")
    sys.exit(0)
else:
    print("❌ CHECKPOINT 2.2 VALIDATION FAILED - SPACES DO NOT MATCH")
    print("="*70)
    print("\nPlease fix the mismatches before proceeding.")
    sys.exit(1)
