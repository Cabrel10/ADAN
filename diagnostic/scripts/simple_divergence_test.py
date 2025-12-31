#!/usr/bin/env python3
"""
Checkpoint 1.3 (Simplified): Direct measurement of normalization divergence
Compares paper_trading_monitor.py normalization vs expected VecNormalize behavior
"""

import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("\n" + "="*70)
print("CHECKPOINT 1.3 (SIMPLIFIED): Normalization Divergence Measurement")
print("="*70)

# ============================================================================
# STEP 1: Load real training data
# ============================================================================

print("\n1️⃣  Loading real training data...")
data_dir = project_root / "data" / "processed" / "indicators"
data_files = list(data_dir.glob("train/**/*.parquet"))

if not data_files:
    print(f"❌ No data files found in {data_dir}")
    sys.exit(1)

data_file = data_files[0]
print(f"   Loading: {data_file}")

try:
    df = pd.read_parquet(data_file)
    print(f"   ✅ Data loaded: {df.shape}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# ============================================================================
# STEP 2: Extract features (last 100 samples)
# ============================================================================

print("\n2️⃣  Extracting features...")
numeric_cols = df.select_dtypes(include=[np.number]).columns
features_raw = df[numeric_cols].tail(100).values.astype(np.float32)

# Normalize to [0, 1] for consistency
features_min = features_raw.min(axis=0)
features_max = features_raw.max(axis=0)
features = (features_raw - features_min) / (features_max - features_min + 1e-8)

print(f"   ✅ Features shape: {features.shape}")
print(f"   Features range: [{features.min():.6f}, {features.max():.6f}]")

# ============================================================================
# STEP 3: Method 1 - CURRENT (paper_trading_monitor.py)
# ============================================================================

print("\n3️⃣  Method 1: CURRENT normalization (paper_trading_monitor.py)")
print("   Using sliding window normalization...")

window_size = 100
window = features[-window_size:] if len(features) >= window_size else features

mean_current = np.mean(window, axis=0)
std_current = np.std(window, axis=0)
std_current = np.where(std_current < 1e-8, 1e-8, std_current)

norm_current = (features - mean_current) / std_current

print(f"   Mean: {mean_current.mean():.6f}")
print(f"   Std: {std_current.mean():.6f}")
print(f"   Normalized range: [{norm_current.min():.6f}, {norm_current.max():.6f}]")

# ============================================================================
# STEP 4: Method 2 - CORRECT (VecNormalize standard)
# ============================================================================

print("\n4️⃣  Method 2: CORRECT normalization (VecNormalize standard)")
print("   Using running mean/std from entire training set...")

# Simulate what VecNormalize should do:
# It accumulates statistics over the entire training set
all_numeric = df[numeric_cols].values.astype(np.float32)
all_numeric_norm = (all_numeric - all_numeric.min(axis=0)) / (
    all_numeric.max(axis=0) - all_numeric.min(axis=0) + 1e-8
)

mean_correct = np.mean(all_numeric_norm, axis=0)
std_correct = np.std(all_numeric_norm, axis=0)
std_correct = np.where(std_correct < 1e-8, 1e-8, std_correct)

norm_correct = (features - mean_correct) / std_correct

print(f"   Mean: {mean_correct.mean():.6f}")
print(f"   Std: {std_correct.mean():.6f}")
print(f"   Normalized range: [{norm_correct.min():.6f}, {norm_correct.max():.6f}]")

# ============================================================================
# STEP 5: Calculate divergence
# ============================================================================

print("\n5️⃣  Calculating divergence...")

# Flatten for comparison
norm_current_flat = norm_current.flatten()
norm_correct_flat = norm_correct.flatten()

# Euclidean distance
divergence_abs = np.linalg.norm(norm_current_flat - norm_correct_flat)

# Relative divergence
norm_magnitude = np.linalg.norm(norm_correct_flat)
divergence_rel = divergence_abs / (norm_magnitude + 1e-8)

print(f"   Absolute divergence: {divergence_abs:.6f}")
print(f"   Relative divergence: {divergence_rel*100:.2f}%")

# ============================================================================
# STEP 6: Interpretation
# ============================================================================

print("\n6️⃣  Interpretation...")

if divergence_abs > 0.1:
    status = "CRITICAL"
    print(f"   🔴 CRITICAL DIVERGENCE (> 0.1)")
    print(f"   → Normalization problem is CONFIRMED")
    print(f"   → The model receives incomprehensible data in production")
elif divergence_abs > 0.01:
    status = "MODERATE"
    print(f"   🟡 MODERATE divergence (0.01 - 0.1)")
    print(f"   → Problem detected but less severe")
else:
    status = "OK"
    print(f"   ✅ Acceptable divergence (< 0.01)")
    print(f"   → No normalization problem detected")

# ============================================================================
# STEP 7: Save results
# ============================================================================

print("\n7️⃣  Saving results...")

results_dir = project_root / "diagnostic" / "results"
results_dir.mkdir(parents=True, exist_ok=True)

result = {
    "timestamp": datetime.now().isoformat(),
    "divergence_absolute": float(divergence_abs),
    "divergence_relative": float(divergence_rel),
    "status": status,
    "features_shape": str(features.shape),
    "data_file": str(data_file),
    "interpretation": {
        "current_method": "Sliding window normalization (paper_trading_monitor.py)",
        "correct_method": "Running mean/std from training set (VecNormalize)",
        "problem": "The model was trained with one normalization method but receives data normalized differently in production"
    }
}

results_file = results_dir / "divergence_report_simple.json"
with open(results_file, 'w') as f:
    json.dump(result, f, indent=2)

print(f"   ✅ Results saved to: {results_file}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\n📊 Divergence Measurement Results:")
print(f"   Absolute divergence: {divergence_abs:.6f}")
print(f"   Relative divergence: {divergence_rel*100:.2f}%")
print(f"   Status: {status}")

if status == "CRITICAL":
    print(f"\n🔴 DIAGNOSIS: NORMALIZATION PROBLEM CONFIRMED")
    print(f"   → Proceed to Phase 2 (Correction)")
    print(f"   → The issue is: paper_trading_monitor.py uses sliding window normalization")
    print(f"   → But the model was trained with VecNormalize (running mean/std)")
    print(f"   → This causes covariate shift and erratic predictions")
elif status == "MODERATE":
    print(f"\n🟡 DIAGNOSIS: Moderate divergence detected")
    print(f"   → Investigate other possible causes")
else:
    print(f"\n✅ DIAGNOSIS: No normalization problem detected")
    print(f"   → Problem may be elsewhere (overfitting, architecture, etc.)")

print("\n" + "="*70)
