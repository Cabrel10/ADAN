#!/usr/bin/env python3
"""Debug des indicateurs techniques ADAN"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_indicators():
    """Debug complet des indicateurs"""
    print("🔧 DEBUG DES INDICATEURS ADAN")
    print("="*60)

    # 1. Vérifier si pandas_ta est installé
    try:
        import pandas_ta as ta
        print("✅ pandas_ta installé")
        
        # Tester un calcul simple
        df = pd.DataFrame({'close': np.random.randn(100) * 100 + 50000})
        rsi = ta.rsi(df['close'], length=14)
        print(f"✅ RSI calculé (shape: {rsi.shape})")
        print(f"   Valeurs: {rsi.dropna().iloc[:5].values}")
    except ImportError as e:
        print(f"❌ pandas_ta non installé: {e}")

    # 2. Vérifier la configuration des indicateurs
    config_path = "config/config.yaml"
    if os.path.exists(config_path):
        print(f"✅ Configuration trouvée: {config_path}")
        
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extraire les indicateurs par timeframe
            timeframes = config.get('data', {}).get('features_config', {}).get('timeframes', {})
            for tf, tf_config in timeframes.items():
                indicators = tf_config.get('indicators', [])
                print(f"\n📈 Timeframe {tf}:")
                print(f"   {len(indicators)} indicateurs configurés:")
                for i, indicator in enumerate(indicators[:10]):
                    print(f"     {i+1}. {indicator}")
                if len(indicators) > 10:
                    print(f"     ... et {len(indicators)-10} de plus")
        except Exception as e:
            print(f"⚠️  Erreur lecture config: {e}")
    else:
        print(f"❌ Configuration non trouvée: {config_path}")

    # 3. Tester une observation factice
    print("\n🧪 TEST D'OBSERVATION FACTICE:")
    fake_obs = np.random.randn(3, 20, 14)
    print(f"   Shape: {fake_obs.shape}")
    print(f"   Valeurs min/max: {fake_obs.min():.2f}/{fake_obs.max():.2f}")
    print(f"   Moyenne/Std: {fake_obs.mean():.2f}/{fake_obs.std():.2f}")

    # 4. Vérifier la normalisation
    try:
        from adan_trading_bot.normalization import ObservationNormalizer
        normalizer = ObservationNormalizer()
        if normalizer.is_loaded:
            normalized = normalizer.normalize(fake_obs.flatten())
            print(f"\n✅ Normalisation testée:")
            print(f"   Shape avant: {fake_obs.shape}")
            print(f"   Shape après: {normalized.shape}")
            print(f"   Normalisé min/max: {normalized.min():.2f}/{normalized.max():.2f}")
        else:
            print("\n⚠️  Normaliseur non chargé")
    except Exception as e:
        print(f"\n❌ Erreur normalisation: {e}")

    # 5. Recommandations
    print("\n🎯 RECOMMANDATIONS:")
    if not os.path.exists("config/config.yaml"):
        print("1. ❌ Config manquante - Vérifier le chemin")
    else:
        print("1. ✅ Config trouvée")
    print("2. Vérifier que paper_trading_monitor.py charge la bonne config")
    print("3. Vérifier que build_observation() calcule les indicateurs")
    print("4. Vérifier les logs pour 'Built observation' ou 'indicators'")

if __name__ == "__main__":
    debug_indicators()
