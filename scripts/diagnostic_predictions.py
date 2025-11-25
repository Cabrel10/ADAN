#!/usr/bin/env python3
"""
🔬 DIAGNOSTIC: Analyze live model predictions
Checks if model always predicts 1.0 (infinite BUY bug)
"""
import sys
import re
from pathlib import Path
from collections import Counter

def analyze_predictions(log_file='logs/paper_trading.log', num_lines=200):
    """Parse predictions from logs and analyze distribution"""
    
    log_path = Path(log_file)
    if not log_path.exists():
        print(f"❌ Log file not found: {log_file}")
        return
    
    # Read last N lines
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    recent_lines = lines[-num_lines:] if len(lines) > num_lines else lines
    
    # Extract predictions
    predictions = []
    for line in recent_lines:
        if 'Model Prediction:' in line:
            match = re.search(r'Model Prediction:\s*([-\d.]+)', line)
            if match:
                pred = float(match.group(1))
                predictions.append(pred)
    
    if not predictions:
        print("⚠️ No predictions found in logs")
        return
    
    print(f"\n📊 PREDICTION ANALYSIS ({len(predictions)} samples)")
    print("=" * 60)
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Min:    {min(predictions):.4f}")
    print(f"  Max:    {max(predictions):.4f}")
    print(f"  Mean:   {sum(predictions)/len(predictions):.4f}")
    print(f"  Unique: {len(set(predictions))} distinct values")
    
    # Distribution
    counter = Counter(predictions)
    print(f"\nTop 5 most frequent predictions:")
    for value, count in counter.most_common(5):
        pct = (count / len(predictions)) * 100
        print(f"  {value:+.4f}: {count:4d} times ({pct:5.1f}%)")
    
    # Warning if all predictions are 1.0
    if all(abs(p - 1.0) < 0.001 for p in predictions):
        print("\n🚨 CRITICAL: All predictions are ~1.0 (BUY)")
        print("   → Model is stuck in BUY mode")
        print("   → Check: data corruption, feature NaN, scaler mismatch")
    
    # Warning if no diversity
    if len(set(predictions)) < 3:
        print(f"\n⚠️ WARNING: Only {len(set(predictions))} unique prediction values")
        print("   → Model lacks diversity, likely data/feature issue")
    
    print("\n" + "=" * 60)
    
    return predictions

if __name__ == "__main__":
    analyze_predictions()
