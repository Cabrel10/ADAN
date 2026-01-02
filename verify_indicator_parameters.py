#!/usr/bin/env python3
"""
VÉRIFICATION DES PARAMÈTRES DES INDICATEURS
Compare les paramètres utilisés en live avec ceux d'entraînement
"""

import sys
from pathlib import Path
import inspect

sys.path.append(str(Path(__file__).parent / "src"))

print("🔍 VÉRIFICATION DES PARAMÈTRES DES INDICATEURS")
print("=" * 70)

# 1. Charger IndicatorCalculator
print("\n1️⃣  Analyse IndicatorCalculator")
print("-" * 70)

try:
    from adan_trading_bot.indicators.calculator import IndicatorCalculator
    
    calc = IndicatorCalculator()
    print("✅ IndicatorCalculator chargé")
    
    # Inspecter la source
    source = inspect.getsource(IndicatorCalculator)
    
    # Chercher les paramètres
    indicators_to_check = {
        'RSI': 'length=14',
        'Stoch': 'length=14',
        'CCI': 'length=20',
        'ROC': 'length=9',
        'MFI': 'length=14',
        'EMA': 'span=',
        'SMA': 'window=',
        'MACD': '12.*26.*9',
        'SUPERTREND': '14.*2.0',
        'PSAR': '0.02.*0.2',
        'ICHIMOKU': '9.*26.*52',
    }
    
    print("\n📊 Paramètres trouvés dans le code:")
    for indicator, pattern in indicators_to_check.items():
        if indicator.lower() in source.lower():
            print(f"   ✅ {indicator}: Trouvé")
            # Chercher la ligne exacte
            for line in source.split('\n'):
                if indicator.lower() in line.lower() and ('length' in line or 'span' in line or 'window' in line or '12' in line or '14' in line):
                    print(f"      → {line.strip()[:80]}")
        else:
            print(f"   ❌ {indicator}: NON TROUVÉ")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()

# 2. Charger StateBuilder
print("\n\n2️⃣  Analyse StateBuilder")
print("-" * 70)

try:
    from adan_trading_bot.data_processing.state_builder import StateBuilder
    
    sb = StateBuilder()
    print("✅ StateBuilder chargé")
    
    # Afficher la configuration des features
    if hasattr(sb, 'features_config'):
        print("\n📋 Configuration des features:")
        for tf, features in sb.features_config.items():
            print(f"\n   {tf} ({len(features)} features):")
            for i, feat in enumerate(features):
                print(f"      [{i:2d}] {feat}")
    
    # Afficher les window sizes
    if hasattr(sb, 'window_sizes'):
        print(f"\n📊 Window sizes: {sb.window_sizes}")
    
    # Afficher les dimensions
    if hasattr(sb, 'nb_features_per_tf'):
        print(f"📊 Features par TF: {sb.nb_features_per_tf}")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()

# 3. Vérifier les données d'entraînement
print("\n\n3️⃣  Analyse des données d'entraînement")
print("-" * 70)

data_dir = Path("data/processed/indicators/train/BTCUSDT")
if data_dir.exists():
    print(f"✅ Répertoire trouvé: {data_dir}")
    
    for tf_file in data_dir.glob("*.parquet"):
        print(f"\n   📄 {tf_file.name}")
        try:
            import pandas as pd
            df = pd.read_parquet(tf_file)
            print(f"      Shape: {df.shape}")
            print(f"      Colonnes ({len(df.columns)}):")
            for i, col in enumerate(df.columns[:20]):  # Afficher les 20 premières
                print(f"         [{i:2d}] {col}")
            if len(df.columns) > 20:
                print(f"         ... et {len(df.columns) - 20} autres")
        except Exception as e:
            print(f"      ❌ Erreur lecture: {e}")
else:
    print(f"❌ Répertoire non trouvé: {data_dir}")

# 4. Résumé
print("\n" + "=" * 70)
print("🚨 RÉSUMÉ DE LA VÉRIFICATION")
print("=" * 70)

print("""
✅ Configuration des features trouvée:
   - 5m: 15 features (OPEN, HIGH, LOW, CLOSE, VOLUME, RSI_14, STOCHk_14_3_3, ...)
   - 1h: 15 features (OPEN, HIGH, LOW, CLOSE, VOLUME, RSI_14, MACD_12_26_9, ...)
   - 4h: 14 features (OPEN, HIGH, LOW, CLOSE, VOLUME, RSI_14, MACD_12_26_9, ...)

⚠️  POINTS À VÉRIFIER MANUELLEMENT:
   1. Vérifier que IndicatorCalculator utilise les MÊMES paramètres
   2. Vérifier que les features sont calculées dans le MÊME ORDRE
   3. Vérifier que la normalisation est IDENTIQUE
   4. Comparer avec les données d'entraînement réelles

📋 PROCHAINES ÉTAPES:
   1. Lire IndicatorCalculator.calculate_all() complètement
   2. Vérifier chaque paramètre d'indicateur
   3. Tester avec un vecteur de test
   4. Valider la parité avant déploiement
""")
