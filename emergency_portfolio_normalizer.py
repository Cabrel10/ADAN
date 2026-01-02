#!/usr/bin/env python3
"""
Normalisateur d'urgence pour portfolio_state
Basé sur les statistiques typiques d'un portefeuille de trading
"""

import pickle
import numpy as np
from pathlib import Path

print("🔧 CRÉATION DU NORMALISATEUR PORTFOLIO D'URGENCE")
print("=" * 60)

class EmergencyPortfolioNormalizer:
    """
    Normalisateur minimaliste pour portfolio_state (20 dimensions)
    Basé sur les statistiques typiques d'entraînement
    """
    
    def __init__(self):
        # Statistiques approximatives pour un portefeuille de trading
        # Dimensions: [balance, position_size, entry_price, current_price, pnl, ...]
        
        self.mean = np.array([
            50.0,      # 0: balance (USDT) - moyenne ~50
            0.0,       # 1: position BTC - moyenne 0 (pas de position)
            90000.0,   # 2: entry_price - prix BTC typique
            90000.0,   # 3: current_price - prix BTC typique
            0.0,       # 4: pnl - moyenne 0
            0.0,       # 5: pnl_percent - moyenne 0
            0.0,       # 6: drawdown - moyenne 0
            1.0,       # 7: sharpe_ratio - moyenne 1
            0.5,       # 8: win_rate - moyenne 0.5
            0.0,       # 9: avg_trade_return - moyenne 0
            0.0,       # 10: trades_count - moyenne 0
            0.0,       # 11: consecutive_wins - moyenne 0
            0.0,       # 12: consecutive_losses - moyenne 0
            0.0,       # 13: max_drawdown - moyenne 0
            0.0,       # 14: volatility - moyenne 0
            0.0,       # 15: time_in_trade - moyenne 0
            0.0,       # 16: trade_duration - moyenne 0
            0.0,       # 17: portfolio_value - moyenne 0
            0.0,       # 18: allocation_ratio - moyenne 0
            0.0        # 19: risk_per_trade - moyenne 0
        ])
        
        # Variances typiques (écarts-types au carré)
        self.var = np.array([
            100.0,     # 0: balance - variance 100 (std=10)
            1.0,       # 1: position - variance 1
            1000000.0, # 2: entry_price - variance 1M (std=1000)
            1000000.0, # 3: current_price - variance 1M
            100.0,     # 4: pnl - variance 100
            0.01,      # 5: pnl_percent - variance 0.01
            0.01,      # 6: drawdown - variance 0.01
            1.0,       # 7: sharpe_ratio - variance 1
            0.25,      # 8: win_rate - variance 0.25
            0.01,      # 9: avg_trade_return - variance 0.01
            10.0,      # 10: trades_count - variance 10
            1.0,       # 11: consecutive_wins - variance 1
            1.0,       # 12: consecutive_losses - variance 1
            0.01,      # 13: max_drawdown - variance 0.01
            0.01,      # 14: volatility - variance 0.01
            1000.0,    # 15: time_in_trade - variance 1000
            1000.0,    # 16: trade_duration - variance 1000
            100.0,     # 17: portfolio_value - variance 100
            0.25,      # 18: allocation_ratio - variance 0.25
            0.01       # 19: risk_per_trade - variance 0.01
        ])
        
        self.is_loaded = True
    
    def normalize(self, obs):
        """
        Normalise l'observation portfolio
        
        Args:
            obs: numpy array de dimension 20
        
        Returns:
            Observation normalisée (moyenne 0, variance 1)
        """
        if len(obs) != len(self.mean):
            print(f"⚠️  Dimension mismatch: {len(obs)} vs {len(self.mean)}")
            # Adapter si nécessaire
            if len(obs) < len(self.mean):
                obs = np.pad(obs, (0, len(self.mean) - len(obs)))
            else:
                obs = obs[:len(self.mean)]
        
        return (obs - self.mean) / np.sqrt(self.var + 1e-8)
    
    def denormalize(self, obs_norm):
        """Dénormalise l'observation (inverse de normalize)"""
        return obs_norm * np.sqrt(self.var + 1e-8) + self.mean

# Créer et sauvegarder
print("\n📦 Création du normalisateur...")
normalizer = EmergencyPortfolioNormalizer()

output_path = Path("models/portfolio_normalizer.pkl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'wb') as f:
    pickle.dump(normalizer, f)

print(f"✅ Normalisateur sauvegardé dans {output_path}")

# Afficher les stats
print("\n📊 STATISTIQUES DU NORMALISATEUR:")
print(f"   Dimensions: {len(normalizer.mean)}")
print(f"\n   Moyennes (premiers 5):")
for i in range(5):
    print(f"      [{i}] = {normalizer.mean[i]:.2f}")
print(f"\n   Variances (premiers 5):")
for i in range(5):
    print(f"      [{i}] = {normalizer.var[i]:.2f}")

# Test de normalisation
print("\n🧪 TEST DE NORMALISATION:")
test_obs = np.array([
    29.0,      # balance brute
    0.0,       # position
    90000.0,   # entry_price
    90000.0,   # current_price
    0.0,       # pnl
    0.0, 0.0, 1.0, 0.5, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0
])

obs_norm = normalizer.normalize(test_obs)
print(f"   Avant: {test_obs[:3]} → Après: {obs_norm[:3]}")
print(f"   Moyenne normalisée: {obs_norm.mean():.6f}")
print(f"   Std normalisée: {obs_norm.std():.6f}")

print("\n" + "=" * 60)
print("✅ NORMALISATEUR D'URGENCE CRÉÉ AVEC SUCCÈS")
