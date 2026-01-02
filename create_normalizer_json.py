#!/usr/bin/env python3
"""
Créer un normalisateur portfolio en JSON (pas de pickle)
"""

import json
import numpy as np
from pathlib import Path

print("🔧 CRÉATION DU NORMALISATEUR PORTFOLIO (JSON)")
print("=" * 60)

# Statistiques approximatives pour un portefeuille de trading
mean = [
    50.0,      # 0: balance (USDT)
    0.0,       # 1: position BTC
    90000.0,   # 2: entry_price
    90000.0,   # 3: current_price
    0.0,       # 4: pnl
    0.0,       # 5: pnl_percent
    0.0,       # 6: drawdown
    1.0,       # 7: sharpe_ratio
    0.5,       # 8: win_rate
    0.0,       # 9: avg_trade_return
    0.0,       # 10: trades_count
    0.0,       # 11: consecutive_wins
    0.0,       # 12: consecutive_losses
    0.0,       # 13: max_drawdown
    0.0,       # 14: volatility
    0.0,       # 15: time_in_trade
    0.0,       # 16: trade_duration
    0.0,       # 17: portfolio_value
    0.0,       # 18: allocation_ratio
    0.0        # 19: risk_per_trade
]

# Variances typiques
var = [
    100.0,     # 0: balance
    1.0,       # 1: position
    1000000.0, # 2: entry_price
    1000000.0, # 3: current_price
    100.0,     # 4: pnl
    0.01,      # 5: pnl_percent
    0.01,      # 6: drawdown
    1.0,       # 7: sharpe_ratio
    0.25,      # 8: win_rate
    0.01,      # 9: avg_trade_return
    10.0,      # 10: trades_count
    1.0,       # 11: consecutive_wins
    1.0,       # 12: consecutive_losses
    0.01,      # 13: max_drawdown
    0.01,      # 14: volatility
    1000.0,    # 15: time_in_trade
    1000.0,    # 16: trade_duration
    100.0,     # 17: portfolio_value
    0.25,      # 18: allocation_ratio
    0.01       # 19: risk_per_trade
]

normalizer_data = {
    "mean": mean,
    "var": var,
    "dimensions": len(mean)
}

output_path = Path("models/portfolio_normalizer.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(normalizer_data, f, indent=2)

print(f"✅ Normalisateur sauvegardé dans {output_path}")

# Afficher les stats
print("\n📊 STATISTIQUES DU NORMALISATEUR:")
print(f"   Dimensions: {len(mean)}")
print(f"\n   Moyennes (premiers 5):")
for i in range(5):
    print(f"      [{i}] = {mean[i]:.2f}")
print(f"\n   Variances (premiers 5):")
for i in range(5):
    print(f"      [{i}] = {var[i]:.2f}")

print("\n" + "=" * 60)
print("✅ NORMALISATEUR JSON CRÉÉ AVEC SUCCÈS")
