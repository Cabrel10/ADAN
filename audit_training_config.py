#!/usr/bin/env python3
"""
EXTRACTION DE LA CONFIGURATION RÉELLE D'ENTRAÎNEMENT
Audit complet de la structure des features
"""

import pickle
import numpy as np
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent / "src"))

print("🔍 AUDIT: Configuration réelle d'entraînement")
print("=" * 70)

# 1. Analyser les fichiers de modèles
models_dir = Path("models")
workers = ['w1', 'w2', 'w3', 'w4']

print("\n1️⃣  ANALYSE DES MODÈLES ENTRAÎNÉS")
print("-" * 70)

for wid in workers:
    print(f"\n📊 Analyse {wid}:")
    
    # a) VecNormalize.pkl contient les stats de normalisation
    vecnorm_path = models_dir / wid / "vecnormalize.pkl"
    if vecnorm_path.exists():
        try:
            with open(vecnorm_path, 'rb') as f:
                vecnorm = pickle.load(f)
            
            print(f"   ✅ VecNormalize chargé")
            
            # Extraire les dimensions
            if hasattr(vecnorm, 'obs_rms'):
                mean = vecnorm.obs_rms.mean
                var = vecnorm.obs_rms.var
                print(f"   Dimensions observation: {len(mean)}")
                print(f"   Mean shape: {mean.shape}")
                print(f"   Var shape: {var.shape}")
                
                # Analyser la distribution
                nonzero_mean = np.sum(np.abs(mean) > 0.01)
                print(f"   Features avec mean ≠ 0: {nonzero_mean}/{len(mean)}")
                
                # Afficher les premières et dernières valeurs
                print(f"   Premières 5 means: {mean[:5]}")
                print(f"   Dernières 5 means: {mean[-5:]}")
                
                # Analyser par segments (5m, 1h, 4h, portfolio)
                # Structure probable: (20*15) + (10*15) + (5*14) + 17 = 300 + 150 + 70 + 17 = 537? Non 542
                # Essayons: (20*15) + (10*15) + (5*14) + 20 = 300 + 150 + 70 + 20 = 540? Proche
                
                print(f"\n   Analyse par segments:")
                if len(mean) == 542:
                    # Segment 5m: 20 fenêtres × 15 features = 300
                    seg_5m = mean[:300]
                    print(f"   5m (0:300): mean={np.mean(seg_5m):.4f}, std={np.std(seg_5m):.4f}")
                    
                    # Segment 1h: 10 fenêtres × 15 features = 150
                    seg_1h = mean[300:450]
                    print(f"   1h (300:450): mean={np.mean(seg_1h):.4f}, std={np.std(seg_1h):.4f}")
                    
                    # Segment 4h: 5 fenêtres × 14 features = 70
                    seg_4h = mean[450:520]
                    print(f"   4h (450:520): mean={np.mean(seg_4h):.4f}, std={np.std(seg_4h):.4f}")
                    
                    # Segment portfolio: 22 features
                    seg_port = mean[520:542]
                    print(f"   Portfolio (520:542): mean={np.mean(seg_port):.4f}, std={np.std(seg_port):.4f}")
                    print(f"   Portfolio values: {seg_port}")
                
        except Exception as e:
            print(f"   ❌ Erreur chargement: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"   ❌ Fichier non trouvé: {vecnorm_path}")

# 2. Analyser la configuration live
print("\n\n2️⃣  ANALYSE DE LA CONFIGURATION LIVE")
print("-" * 70)

try:
    from adan_trading_bot.data_processing.state_builder import StateBuilder
    from adan_trading_bot.indicators.calculator import IndicatorCalculator
    
    sb = StateBuilder()
    print(f"   ✅ StateBuilder chargé")
    
    if hasattr(sb, 'features_per_timeframe'):
        print(f"   Features par TF: {sb.features_per_timeframe}")
    
    if hasattr(sb, 'window_sizes'):
        print(f"   Window sizes: {sb.window_sizes}")
    
    if hasattr(sb, 'timeframes'):
        print(f"   Timeframes: {sb.timeframes}")
    
    # Calculer le nombre total de features
    total_features = 0
    if hasattr(sb, 'features_per_timeframe') and hasattr(sb, 'window_sizes'):
        for tf in sb.timeframes:
            n_features = sb.features_per_timeframe.get(tf, 15)
            n_windows = sb.window_sizes.get(tf, 20)
            subtotal = n_features * n_windows
            total_features += subtotal
            print(f"   {tf}: {n_windows} fenêtres × {n_features} features = {subtotal}")
    
    print(f"   Total features market: {total_features}")
    print(f"   Total avec portfolio: {total_features + 20}")
    
except ImportError as e:
    print(f"   ❌ Impossible d'importer: {e}")

# 3. Chercher les fichiers de configuration
print("\n\n3️⃣  RECHERCHE FICHIERS DE CONFIGURATION")
print("-" * 70)

config_patterns = [
    "training_config*",
    "*feature*config*",
    "*obs*config*",
    "feature_list*",
    "indicator_config*"
]

found_configs = []
for pattern in config_patterns:
    for f in Path(".").rglob(pattern):
        if f.is_file() and not f.name.startswith("."):
            found_configs.append(f)
            print(f"   📄 {f}")

if not found_configs:
    print(f"   ⚠️  Aucun fichier de configuration trouvé")

# 4. Chercher les références aux features dans le code
print("\n\n4️⃣  RECHERCHE RÉFÉRENCES AUX FEATURES DANS LE CODE")
print("-" * 70)

feature_refs = []
for py_file in Path("src").rglob("*.py"):
    try:
        with open(py_file, 'r') as f:
            content = f.read()
            if 'feature' in content.lower() and ('rsi' in content.lower() or 'macd' in content.lower()):
                feature_refs.append(py_file)
                print(f"   📝 {py_file}")
    except:
        pass

if not feature_refs:
    print(f"   ⚠️  Aucune référence trouvée")

print("\n" + "=" * 70)
print("🚨 RÉSUMÉ AUDIT")
print("=" * 70)
print(f"✅ Modèles trouvés: {len(workers)}")
print(f"✅ Fichiers config trouvés: {len(found_configs)}")
print(f"✅ Fichiers avec références features: {len(feature_refs)}")
print("\n📋 PROCHAINES ÉTAPES:")
print("1. Vérifier StateBuilder pour la liste EXACTE des features")
print("2. Vérifier IndicatorCalculator pour les paramètres")
print("3. Comparer avec les données d'entraînement (data/processed/indicators/)")
