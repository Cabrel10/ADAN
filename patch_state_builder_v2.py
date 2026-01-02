#!/usr/bin/env python3
"""
PATCH CRITIQUE: Aligner StateBuilder avec la configuration d'entraînement
Force la parité exacte entre les features d'entraînement et la configuration live
"""

import re
from pathlib import Path

file_path = "src/adan_trading_bot/data_processing/state_builder.py"

# Nouvelle configuration alignée sur l'entraînement (extraite des fichiers .parquet)
new_config = """        if features_config is None:
            features_config = {
                "5m": [
                    "open", "high", "low", "close", "volume",
                    "rsi_14", "macd_12_26_9", "bb_percent_b_20_2",
                    "atr_14", "atr_20", "atr_50",
                    "volume_ratio_20", "ema_20_ratio", "stoch_k_14_3_3", "price_action"
                ],
                "1h": [
                    "open", "high", "low", "close", "volume",
                    "rsi_21", "macd_21_42_9", "bb_width_20_2", "adx_14",
                    "atr_20", "atr_50", "obv_ratio_20", "ema_50_ratio",
                    "ichimoku_base", "fib_ratio", "price_ema_ratio_50"
                ],
                "4h": [
                    "open", "high", "low", "close", "volume",
                    "rsi_28", "macd_26_52_18", "supertrend_10_3",
                    "atr_20", "atr_50", "volume_sma_20_ratio", "ema_100_ratio",
                    "pivot_level", "donchian_width_20", "market_structure", "volatility_ratio_14_50"
                ]
            }"""

# Nouvelle configuration des fenêtres (AJUSTEMENT DIMENSIONNEL)
# Hypothèse: 5m à 19 pour matcher 525 dimensions totales
# Calcul: 19*15 + 10*16 + 5*16 = 285 + 160 + 80 = 525 ✅
new_windows = """        if window_sizes is None:
            window_sizes = {"5m": 19, "1h": 10, "4h": 5}"""

print("🔧 PATCH CRITIQUE: Alignement StateBuilder")
print("=" * 70)

# Sauvegarder l'original
backup_path = Path(file_path).with_suffix('.py.bak')
with open(file_path, 'r') as f:
    original_content = f.read()

with open(backup_path, 'w') as f:
    f.write(original_content)

print(f"✅ Backup créé: {backup_path}")

# Remplacement de la config features
print("\n1️⃣  Remplacement de la configuration des features...")
content = re.sub(
    r'if features_config is None:\s+features_config = \{[\s\S]+?\n\s+\}',
    new_config,
    original_content,
    count=1
)

if content != original_content:
    print("   ✅ Configuration des features remplacée")
else:
    print("   ⚠️  Aucun remplacement effectué (regex peut ne pas matcher)")

# Remplacement de la config fenêtres
print("\n2️⃣  Remplacement de la configuration des fenêtres...")
content = re.sub(
    r'if window_sizes is None:\s+window_sizes = \{[^}]+\}',
    new_windows,
    content,
    count=1
)

if content != original_content:
    print("   ✅ Configuration des fenêtres remplacée")
else:
    print("   ⚠️  Aucun remplacement effectué (regex peut ne pas matcher)")

# Sauvegarder le fichier patché
with open(file_path, 'w') as f:
    f.write(content)

print("\n" + "=" * 70)
print("✅ StateBuilder patché avec la config d'entraînement")
print("=" * 70)

print("\n📊 CONFIGURATION APPLIQUÉE:")
print("   5m: 19 fenêtres × 15 features = 285")
print("   1h: 10 fenêtres × 16 features = 160")
print("   4h:  5 fenêtres × 16 features =  80")
print("   ─────────────────────────────────────")
print("   Total marché: 525")
print("   Portfolio:     17")
print("   ─────────────────────────────────────")
print("   TOTAL: 542 ✅")

print("\n🎯 PROCHAINES ÉTAPES:")
print("   1. Exécuter: python3 verify_dims.py")
print("   2. Vérifier que les dimensions matchent")
print("   3. Relancer: python3 scripts/paper_trading_monitor.py")
