#!/usr/bin/env python3
"""
Script de test pour vérifier la construction dynamique de base_feature_names
et la configuration des indicateurs par timeframe
"""

import sys
import os
import pandas as pd
import numpy as np

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from adan_trading_bot.environment.multi_asset_env import MultiAssetEnv
from adan_trading_bot.environment.state_builder import StateBuilder
from adan_trading_bot.common.utils import get_logger, load_config

logger = get_logger()

def test_lot1_dynamic_features():
    """Test la construction dynamique pour le Lot 1."""
    
    print("🔧 Test Lot 1 - Construction dynamique des features")
    
    # Charger la configuration Lot 1
    data_config = load_config('config/data_config_cpu_lot1.yaml')
    env_config = load_config('config/environment_config.yaml')
    
    config = {
        'data': data_config,
        'environment': env_config
    }
    
    assets = data_config['assets']
    training_tf = data_config.get('training_timeframe', '1h')
    indicators_config = data_config.get('indicators_by_timeframe', {}).get(training_tf, [])
    
    print(f"   Assets: {assets}")
    print(f"   Training timeframe: {training_tf}")
    print(f"   Indicateurs configurés: {len(indicators_config)}")
    
    # Créer DataFrame mock avec colonnes attendues APRÈS process_data.py
    columns = []
    expected_base_features = ["open", "high", "low", "close", "volume"]
    
    # Ajouter les indicateurs avec suffixe timeframe comme généré par add_technical_indicators
    for indicator in indicators_config:
        output_col_name = indicator.get('output_col_name')
        output_col_names = indicator.get('output_col_names')
        
        if output_col_name:
            expected_base_features.append(f"{output_col_name}_{training_tf}")
        elif output_col_names:
            for col_name in output_col_names:
                expected_base_features.append(f"{col_name}_{training_tf}")
    
    # Créer colonnes fusionnées: {feature}_{asset}
    for asset in assets:
        for feature in expected_base_features:
            columns.append(f"{feature}_{asset}")
    
    print(f"   Features de base attendues: {expected_base_features}")
    print(f"   Total colonnes fusionnées: {len(columns)}")
    
    # Créer DataFrame mock
    df = pd.DataFrame(np.random.randn(100, len(columns)), columns=columns)
    df.index = pd.date_range('2024-01-01', periods=100, freq='1h')
    
    try:
        env = MultiAssetEnv(df_received=df, config=config, scaler=None, encoder=None)
        
        print("✅ MultiAssetEnv créé avec succès")
        print(f"   Features construites: {env.base_feature_names}")
        print(f"   Nombre de features: {len(env.base_feature_names)} (attendu: {len(expected_base_features)})")
        
        # Vérifier que les features correspondent
        if set(env.base_feature_names) == set(expected_base_features):
            print("✅ Features correspondent parfaitement!")
        else:
            print("⚠️  Différences dans les features:")
            missing = set(expected_base_features) - set(env.base_feature_names)
            extra = set(env.base_feature_names) - set(expected_base_features)
            if missing:
                print(f"   Manquantes: {missing}")
            if extra:
                print(f"   En plus: {extra}")
        
        # Test StateBuilder
        print("\n🔧 Test StateBuilder avec features dynamiques...")
        state_builder = StateBuilder(config, assets, base_feature_names=env.base_feature_names, cnn_input_window_size=20)
        
        # Test d'observation
        window = df.tail(20)  # Dernières 20 lignes
        observation = state_builder.build_observation(window, 100.0, {}, image_shape=(1, 20, len(env.base_feature_names) * len(assets)))
        
        expected_features_total = len(env.base_feature_names) * len(assets)
        actual_features = observation['image_features'].shape[-1]
        
        print(f"   Features dans observation: {actual_features} (attendu: {expected_features_total})")
        
        if actual_features == expected_features_total:
            print("✅ StateBuilder fonctionne correctement!")
            return True
        else:
            print("❌ Mismatch dans le nombre de features StateBuilder")
            return False
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lot2_precomputed_features():
    """Test la configuration pour le Lot 2 avec features pré-calculées."""
    
    print("\n🔧 Test Lot 2 - Features pré-calculées")
    
    # Charger la configuration Lot 2
    data_config = load_config('config/data_config_cpu_lot2.yaml')
    env_config = load_config('config/environment_config.yaml')
    
    config = {
        'data': data_config,
        'environment': env_config
    }
    
    assets = data_config['assets']
    base_market_features = data_config.get('base_market_features', [])
    
    print(f"   Assets: {assets}")
    print(f"   Features pré-calculées: {len(base_market_features)}")
    print(f"   Exemples: {base_market_features[:5]}")
    
    # Créer DataFrame mock avec colonnes fusionnées
    columns = []
    for asset in assets:
        for feature in base_market_features:
            columns.append(f"{feature}_{asset}")
    
    print(f"   Total colonnes fusionnées: {len(columns)}")
    
    # Créer DataFrame mock
    df = pd.DataFrame(np.random.randn(100, len(columns)), columns=columns)
    df.index = pd.date_range('2024-01-01', periods=100, freq='1m')
    
    try:
        env = MultiAssetEnv(df_received=df, config=config, scaler=None, encoder=None)
        
        print("✅ MultiAssetEnv créé avec succès")
        print(f"   Features utilisées: {len(env.base_feature_names)}")
        print(f"   Correspondent aux base_market_features: {set(env.base_feature_names) == set(base_market_features)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_feature_extraction():
    """Test l'extraction des features par StateBuilder."""
    
    print("\n🔧 Test extraction features par StateBuilder")
    
    # Configuration simple pour test
    config = {
        'data': {
            'training_timeframe': '1h',
            'cnn_input_window_size': 20
        },
        'environment': {
            'initial_capital': 100.0
        }
    }
    
    assets = ['BTCUSDT', 'ETHUSDT']
    base_features = ['open', 'high', 'low', 'close', 'volume', 'rsi_14_1h', 'ema_20_1h']
    
    # Créer DataFrame avec colonnes fusionnées
    columns = []
    for asset in assets:
        for feature in base_features:
            columns.append(f"{feature}_{asset}")
    
    df = pd.DataFrame(np.random.randn(50, len(columns)), columns=columns)
    
    state_builder = StateBuilder(config, assets, base_feature_names=base_features, cnn_input_window_size=20)
    
    try:
        # Test avec fenêtre complète
        window = df.tail(20)
        observation = state_builder.build_observation(window, 100.0, {})
        
        expected_shape = (1, 20, len(base_features) * len(assets))
        actual_shape = observation['image_features'].shape
        
        print(f"   Shape attendue: {expected_shape}")
        print(f"   Shape actuelle: {actual_shape}")
        
        if actual_shape == expected_shape:
            print("✅ Extraction features réussie!")
            return True
        else:
            print("❌ Shape incorrecte")
            return False
            
    except Exception as e:
        print(f"❌ Erreur extraction: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTS DE VALIDATION DES FEATURES DYNAMIQUES")
    print("=" * 60)
    
    success1 = test_lot1_dynamic_features()
    success2 = test_lot2_precomputed_features()
    success3 = test_feature_extraction()
    
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS:")
    print(f"   Lot 1 (dynamique): {'✅ SUCCÈS' if success1 else '❌ ÉCHEC'}")
    print(f"   Lot 2 (pré-calc): {'✅ SUCCÈS' if success2 else '❌ ÉCHEC'}")
    print(f"   Extraction features: {'✅ SUCCÈS' if success3 else '❌ ÉCHEC'}")
    
    overall_success = success1 and success2 and success3
    print(f"\n🎯 RÉSULTAT GLOBAL: {'✅ TOUS LES TESTS RÉUSSIS' if overall_success else '❌ CERTAINS TESTS ONT ÉCHOUÉ'}")
    print("=" * 60)
    
    exit(0 if overall_success else 1)