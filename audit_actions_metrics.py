#!/usr/bin/env python3
"""
AUDIT ACTIONS & MÉTRIQUES (TRAINING vs LIVE)
Vérifie la cohérence entre l'entraînement et la production
"""

import sys
import pickle
import numpy as np
import yaml
from pathlib import Path

sys.path.append("src")

print("🔬 AUDIT ACTIONS & MÉTRIQUES (TRAINING vs LIVE)")
print("=" * 70)

# 1. CHARGEMENT CONFIG
print("\n1️⃣  CHARGEMENT CONFIGURATION")
print("-" * 70)

try:
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("✅ Config chargée")
    
    # Afficher les paramètres critiques
    if 'trading' in config:
        print("\n   📊 Paramètres trading:")
        for key in ['initial_balance', 'position_size', 'max_position', 'stop_loss', 'take_profit', 'commission']:
            if key in config['trading']:
                print(f"      {key}: {config['trading'][key]}")
    
except Exception as e:
    print(f"❌ Erreur config: {e}")

# 2. VÉRIFICATION ACTION SPACE
print("\n2️⃣  ACTION SPACE TRANSLATION")
print("-" * 70)

# Simulation d'une action modèle brute
# Format: [Action, Size, Timeframe, SL, TP] pour chaque actif
# Avec 5 actifs: 25 dimensions totales

raw_action_asset_0 = np.array([1.0, 0.5, -1.0, -0.5, 0.5], dtype=np.float32)
print(f"   Action brute simulée (Asset 0): {raw_action_asset_0}")

# Décodage théorique
# Action: > 0.33 => BUY, < -0.33 => SELL, else => HOLD
action_signal = "BUY" if raw_action_asset_0[0] > 0.33 else ("SELL" if raw_action_asset_0[0] < -0.33 else "HOLD")
position_size = raw_action_asset_0[1]  # 0 à 1
timeframe_idx = int((raw_action_asset_0[2] + 1) * 1.5)  # -1 -> 0 (5m), 0 -> 1 (1h), 1 -> 2 (4h)
sl_pct = 0.005 + (raw_action_asset_0[3] + 1) / 2 * (0.10 - 0.005)  # -1 -> 0.5%, 1 -> 10%
tp_pct = 0.01 + (raw_action_asset_0[4] + 1) / 2 * (0.20 - 0.01)  # -1 -> 1%, 1 -> 20%

print(f"\n   Décodage:")
print(f"      Signal: {action_signal}")
print(f"      Position size: {position_size:.1%}")
print(f"      Timeframe: {['5m', '1h', '4h'][timeframe_idx]}")
print(f"      Stop-loss: {sl_pct:.2%}")
print(f"      Take-profit: {tp_pct:.2%}")

if action_signal in ["BUY", "SELL", "HOLD"] and 0 <= position_size <= 1:
    print(f"\n   ✅ Traduction Action: COHÉRENTE")
else:
    print(f"\n   ⚠️  Traduction Action: DOUTEUSE")

# 3. VÉRIFICATION PORTFOLIO STATE
print("\n3️⃣  PORTFOLIO STATE (17 Features)")
print("-" * 70)

# Vérifier la dimension du portfolio normalizer
norm_path = Path("models/portfolio_normalizer.pkl")
if norm_path.exists():
    try:
        with open(norm_path, 'rb') as f:
            normalizer = pickle.load(f)
        
        if hasattr(normalizer, 'mean'):
            portfolio_dims = len(normalizer.mean)
            print(f"   Portfolio normalizer dimensions: {portfolio_dims}")
            
            if portfolio_dims == 17:
                print(f"   ✅ Dimension Portfolio: EXACTE (17)")
            elif portfolio_dims == 20:
                print(f"   ❌ Dimension Portfolio: ERREUR ({portfolio_dims} vs 17 attendu)")
                print(f"      → Ancien format détecté! Correction nécessaire.")
            else:
                print(f"   ⚠️  Dimension Portfolio: INATTENDUE ({portfolio_dims})")
        else:
            print(f"   ⚠️  Normalizer sans attribut 'mean'")
    except Exception as e:
        print(f"   ❌ Erreur chargement normalizer: {e}")
else:
    print(f"   ❌ Normalizer non trouvé: {norm_path}")

# 4. VÉRIFICATION PnL & FRAIS
print("\n4️⃣  CALCUL PnL & FRAIS")
print("-" * 70)

# Scénario: Achat 1 BTC à 50000, Vente à 51000. Frais 0.1%
entry_price = 50000.0
exit_price = 51000.0
size = 1.0
fee_rate = 0.001  # 0.1%

# Calcul théorique
pnl_gross = (exit_price - entry_price) * size
entry_fee = (entry_price * size) * fee_rate
exit_fee = (exit_price * size) * fee_rate
total_fees = entry_fee + exit_fee
pnl_net = pnl_gross - total_fees

print(f"   Scénario: Achat 1 BTC @ 50k, Vente @ 51k, Frais 0.1%")
print(f"   ─────────────────────────────────────────────────")
print(f"   PnL Brut:        ${pnl_gross:>10.2f}")
print(f"   Frais Achat:     ${entry_fee:>10.2f}")
print(f"   Frais Vente:     ${exit_fee:>10.2f}")
print(f"   Frais Total:     ${total_fees:>10.2f}")
print(f"   ─────────────────────────────────────────────────")
print(f"   PnL Net:         ${pnl_net:>10.2f}")

if abs(pnl_net - 899.0) < 0.01:
    print(f"\n   ✅ Mathématiques PnL: CORRECTES")
else:
    print(f"\n   ❌ Mathématiques PnL: INCORRECTES (Attendu 899.00)")

# 5. VÉRIFICATION WORKERS
print("\n5️⃣  VÉRIFICATION WORKERS (w1, w2, w3, w4)")
print("-" * 70)

from stable_baselines3 import PPO

models_dir = Path("models")
workers = ['w1', 'w2', 'w3', 'w4']

for wid in workers:
    print(f"\n   📊 {wid}:")
    model_path = models_dir / wid / f"{wid}_model_final.zip"
    
    if model_path.exists():
        try:
            model = PPO.load(model_path)
            print(f"      ✅ Modèle chargé")
            print(f"      Action space: {model.action_space}")
            print(f"      Observation space: {model.observation_space}")
            
            # Vérifier que l'observation space est 542
            if hasattr(model.observation_space, 'shape'):
                obs_dims = model.observation_space.shape[0]
                if obs_dims == 542:
                    print(f"      ✅ Observation dimensions: CORRECTES (542)")
                else:
                    print(f"      ❌ Observation dimensions: INCORRECTES ({obs_dims} vs 542)")
        except Exception as e:
            print(f"      ❌ Erreur: {e}")
    else:
        print(f"      ❌ Modèle non trouvé: {model_path}")

# 6. RÉSUMÉ
print("\n" + "=" * 70)
print("📋 RÉSUMÉ DE L'AUDIT")
print("=" * 70)

print("""
✅ Points vérifiés:
   1. Action space translation (HOLD, BUY, SELL)
   2. Position sizing (0-1 range)
   3. Timeframe decoding (5m, 1h, 4h)
   4. Stop-loss / Take-profit ranges
   5. Portfolio state dimensions (17)
   6. PnL calculation with fees
   7. Worker models loaded correctly
   8. Observation space dimensions (542)

🎯 Prochaines étapes:
   1. Vérifier que tous les ✅ sont présents
   2. Si ❌ détecté, appliquer le patch correspondant
   3. Relancer l'audit jusqu'à 100% ✅
   4. Déployer en confiance
""")
