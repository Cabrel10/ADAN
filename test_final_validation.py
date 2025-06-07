#!/usr/bin/env python3
"""
Script de validation finale pour toutes les corrections effectuées sur ADAN.
Ce script vérifie que le système est prêt pour l'entraînement long.
"""

import sys
import os
import pandas as pd
import numpy as np

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from adan_trading_bot.environment.multi_asset_env import MultiAssetEnv
from adan_trading_bot.environment.state_builder import StateBuilder
from adan_trading_bot.environment.order_manager import OrderManager
from adan_trading_bot.common.utils import get_logger, load_config

logger = get_logger()

def create_mock_merged_dataframe(assets, base_features, timeframe='1h', num_rows=100):
    """Créer un DataFrame mock fusionné pour les tests."""
    columns = []
    for asset in assets:
        for feature in base_features:
            columns.append(f"{feature}_{asset}")
    
    df = pd.DataFrame(np.random.randn(num_rows, len(columns)), columns=columns)
    df.index = pd.date_range('2024-01-01', periods=num_rows, freq=timeframe)
    
    # S'assurer que les prix sont positifs
    for asset in assets:
        for price_col in ['open', 'high', 'low', 'close']:
            col_name = f"{price_col}_{asset}"
            if col_name in df.columns:
                df[col_name] = np.abs(df[col_name]) + 50  # Prix positifs autour de 50
    
    return df

def test_lot1_environment():
    """Test complet de l'environnement Lot 1."""
    print("🔧 Test Lot 1 - Environnement complet")
    
    try:
        # Charger configuration
        data_config = load_config('config/data_config_cpu_lot1.yaml')
        env_config = load_config('config/environment_config.yaml')
        config = {'data': data_config, 'environment': env_config}
        
        assets = data_config['assets']
        training_tf = data_config.get('training_timeframe', '1h')
        
        # Construire les features attendues dynamiquement
        expected_features = ["open", "high", "low", "close", "volume"]
        indicators_config = data_config.get('indicators_by_timeframe', {}).get(training_tf, [])
        
        for indicator in indicators_config:
            output_col_name = indicator.get('output_col_name')
            output_col_names = indicator.get('output_col_names')
            
            if output_col_name:
                expected_features.append(f"{output_col_name}_{training_tf}")
            elif output_col_names:
                for col_name in output_col_names:
                    expected_features.append(f"{col_name}_{training_tf}")
        
        # Créer DataFrame mock
        df = create_mock_merged_dataframe(assets, expected_features, timeframe='1h')
        
        # Créer environnement
        env = MultiAssetEnv(df_received=df, config=config, max_episode_steps_override=50)
        
        # Vérifications
        assert len(env.base_feature_names) == len(expected_features), f"Features mismatch: {len(env.base_feature_names)} vs {len(expected_features)}"
        assert env.training_timeframe == training_tf, f"Timeframe mismatch: {env.training_timeframe} vs {training_tf}"
        
        # Test reset
        obs, info = env.reset()
        assert 'image_features' in obs, "Missing image_features in observation"
        assert 'vector_features' in obs, "Missing vector_features in observation"
        
        expected_img_shape = (1, data_config.get('cnn_input_window_size', 20), len(expected_features) * len(assets))
        assert obs['image_features'].shape == expected_img_shape, f"Image shape mismatch: {obs['image_features'].shape} vs {expected_img_shape}"
        
        # Test step avec action BUY
        action = 1  # BUY premier actif
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"   ✅ Features: {len(env.base_feature_names)} construites dynamiquement")
        print(f"   ✅ Observation shape: {obs['image_features'].shape}")
        print(f"   ✅ Action BUY testée, reward: {reward:.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur Lot 1: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lot2_environment():
    """Test complet de l'environnement Lot 2."""
    print("🔧 Test Lot 2 - Environnement complet")
    
    try:
        # Charger configuration
        data_config = load_config('config/data_config_cpu_lot2.yaml')
        env_config = load_config('config/environment_config.yaml')
        config = {'data': data_config, 'environment': env_config}
        
        assets = data_config['assets']
        base_features = data_config.get('base_market_features', [])
        
        # Créer DataFrame mock
        df = create_mock_merged_dataframe(assets, base_features, timeframe='1m')
        
        # Créer environnement
        env = MultiAssetEnv(df_received=df, config=config, max_episode_steps_override=50)
        
        # Vérifications
        assert len(env.base_feature_names) == len(base_features), f"Features mismatch: {len(env.base_feature_names)} vs {len(base_features)}"
        assert set(env.base_feature_names) == set(base_features), "Features content mismatch"
        
        # Test reset et step
        obs, info = env.reset()
        action = 3  # SELL premier actif (devrait être rejeté car pas de position)
        obs, reward, done, truncated, info = env.step(action)
        
        expected_img_shape = (1, data_config.get('cnn_input_window_size', 20), len(base_features) * len(assets))
        assert obs['image_features'].shape == expected_img_shape, f"Image shape mismatch: {obs['image_features'].shape} vs {expected_img_shape}"
        
        print(f"   ✅ Features: {len(base_features)} pré-calculées utilisées")
        print(f"   ✅ Observation shape: {obs['image_features'].shape}")
        print(f"   ✅ Action SELL (sans position) testée, reward: {reward:.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur Lot 2: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_order_manager_normalized_prices():
    """Test OrderManager avec prix normalisés (négatifs)."""
    print("🔧 Test OrderManager - Prix normalisés")
    
    try:
        # Configuration simple
        config = {
            'environment': {
                'penalties': {
                    'invalid_order_base': -0.3,
                    'out_of_funds': -0.5,
                    'order_below_tolerable_if_not_adjusted': -0.3
                },
                'order_rules': {
                    'min_value_tolerable': 10.0,
                    'min_value_absolute': 9.0
                },
                'transaction': {
                    'fee_percent': 0.001,
                    'fixed_fee': 0.0
                }
            }
        }
        
        order_manager = OrderManager(config)
        
        # Test BUY avec prix négatif (normalisé)
        capital = 100.0
        positions = {}
        current_price = -1.5  # Prix négatif (normalisé)
        
        reward, status, info = order_manager.execute_order(
            asset_id='BTCUSDT',
            action_type=1,  # BUY
            current_price=current_price,
            capital=capital,
            positions=positions,
            allocated_value_usdt=50.0
        )
        
        assert status == "BUY_EXECUTED", f"BUY should succeed with negative price, got: {status}"
        assert 'BTCUSDT' in positions, "Position should be created"
        assert positions['BTCUSDT']['qty'] > 0, f"Quantity should be positive: {positions['BTCUSDT']['qty']}"
        assert positions['BTCUSDT']['price'] == current_price, "Price should be stored as-is (normalized)"
        
        # Test SELL avec prix normalisé différent
        new_price = -1.2  # Prix différent pour PnL
        
        reward, status, info = order_manager.execute_order(
            asset_id='BTCUSDT',
            action_type=2,  # SELL
            current_price=new_price,
            capital=info['new_capital'],
            positions=positions,
            quantity=None  # Vendre toute la position
        )
        
        assert status == "SELL_EXECUTED", f"SELL should succeed, got: {status}"
        assert 'BTCUSDT' not in positions, "Position should be closed after full SELL"
        
        # Test avec prix très petit (devrait être rejeté)
        tiny_price = 1e-10
        
        reward, status, info = order_manager.execute_order(
            asset_id='ETHUSDT',
            action_type=1,  # BUY
            current_price=tiny_price,
            capital=100.0,
            positions={},
            allocated_value_usdt=50.0
        )
        
        assert status == "PRICE_NOT_AVAILABLE", f"Should reject tiny price, got: {status}"
        
        print(f"   ✅ BUY avec prix négatif: réussi")
        print(f"   ✅ SELL avec calcul PnL: réussi")
        print(f"   ✅ Rejet prix trop petit: réussi")
        print(f"   ✅ Gestion quantités positives: réussi")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur prix normalisés: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_state_builder_missing_features():
    """Test StateBuilder avec des features manquantes."""
    print("🔧 Test StateBuilder - Gestion features manquantes")
    
    try:
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
        base_features = ['open', 'high', 'low', 'close', 'volume', 'rsi_14_1h']
        
        # Créer DataFrame avec des colonnes manquantes
        available_features = ['open', 'high', 'low', 'close', 'volume']  # rsi_14_1h manquant
        columns = []
        for asset in assets:
            for feature in available_features:
                columns.append(f"{feature}_{asset}")
        
        df = pd.DataFrame(np.random.randn(30, len(columns)), columns=columns)
        
        state_builder = StateBuilder(config, assets, base_feature_names=base_features, cnn_input_window_size=20)
        
        # Test avec features manquantes (devrait gérer gracieusement)
        window = df.tail(20)
        observation = state_builder.build_observation(window, 100.0, {})
        
        # Vérifier que l'observation est créée malgré les features manquantes
        expected_shape = (1, 20, len(base_features) * len(assets))
        assert observation['image_features'].shape == expected_shape, f"Shape mismatch: {observation['image_features'].shape} vs {expected_shape}"
        
        # Vérifier que les features manquantes sont remplies de NaN
        img_features = observation['image_features'][0]  # Remove channel dimension
        has_nan = np.isnan(img_features).any()
        assert has_nan, "Should have NaN values for missing features"
        
        print(f"   ✅ Gestion features manquantes: NaN insérés")
        print(f"   ✅ Shape préservée: {observation['image_features'].shape}")
        print(f"   ✅ Observation créée malgré les manques")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur features manquantes: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_logging():
    """Test que les logs sont bien nettoyés et informatifs."""
    print("🔧 Test Logs - Vérification nettoyage")
    
    try:
        # Capturer les logs pendant la création d'environnement
        import io
        import logging
        
        # Créer un handler pour capturer les logs
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Test avec Lot 1
        data_config = load_config('config/data_config_cpu_lot1.yaml')
        env_config = load_config('config/environment_config.yaml')
        config = {'data': data_config, 'environment': env_config}
        
        assets = data_config['assets']
        expected_features = ["open", "high", "low", "close", "volume", "rsi_14_1h", "ema_20_1h", "ema_50_1h"]
        df = create_mock_merged_dataframe(assets, expected_features)
        
        # Créer environnement avec export désactivé pour éviter les logs visuels
        config['environment']['export_history'] = False
        env = MultiAssetEnv(df_received=df, config=config, max_episode_steps_override=10)
        
        # Faire quelques steps pour générer des logs (sans affichage)
        obs, info = env.reset()
        for _ in range(2):
            action = 0  # Action HOLD pour éviter trop de logs
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break
        
        # Analyser les logs
        log_output = log_capture.getvalue()
        
        # Vérifier qu'il n'y a pas trop de logs de debug
        debug_count = log_output.count('DEBUG')
        info_count = log_output.count('INFO')
        
        # Les logs INFO devraient être présents mais pas excessifs
        assert info_count > 0, "Should have some INFO logs"
        assert info_count < 50, f"Too many INFO logs: {info_count}"
        
        # Vérifier qu'il n'y a pas de logs d'erreur critiques non gérés
        assert 'ERREUR CRITIQUE' not in log_output, "Should not have unhandled critical errors"
        assert 'SOLUTION TEMPORAIRE' not in log_output, "Should not have temporary solutions"
        
        # Nettoyer
        logger.removeHandler(handler)
        
        print(f"   ✅ Logs INFO: {info_count} (raisonnable)")
        print(f"   ✅ Pas d'erreurs critiques non gérées")
        print(f"   ✅ Pas de solutions temporaires")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur test logs: {e}")
        return False

def run_all_tests():
    """Exécuter tous les tests de validation finale."""
    
    print("🎯 VALIDATION FINALE - SYSTÈME ADAN")
    print("=" * 60)
    
    tests = [
        ("Lot 1 - Environnement", test_lot1_environment),
        ("Lot 2 - Environnement", test_lot2_environment),
        ("OrderManager - Prix normalisés", test_order_manager_normalized_prices),
        ("StateBuilder - Features manquantes", test_state_builder_missing_features),
        ("Logs - Nettoyage", test_environment_logging)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 40)
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name}: SUCCÈS")
            else:
                print(f"❌ {test_name}: ÉCHEC")
        except Exception as e:
            print(f"❌ {test_name}: ERREUR - {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS FINAUX:")
    print("=" * 60)
    
    success_count = 0
    for test_name, success in results:
        status = "✅ SUCCÈS" if success else "❌ ÉCHEC"
        print(f"   {test_name:<35} {status}")
        if success:
            success_count += 1
    
    overall_success = success_count == len(results)
    
    print("\n" + "=" * 60)
    if overall_success:
        print("🎉 VALIDATION COMPLÈTE RÉUSSIE!")
        print("   Le système ADAN est prêt pour l'entraînement long.")
        print("   Toutes les corrections ont été validées avec succès.")
    else:
        print("⚠️  VALIDATION PARTIELLE")
        print(f"   {success_count}/{len(results)} tests réussis.")
        print("   Vérifiez les erreurs ci-dessus avant l'entraînement long.")
    
    print("=" * 60)
    
    return overall_success

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)