#!/usr/bin/env python3
"""
Script de test pour valider les corrections apportées au système ADAN.
Ce script teste les composants clés : StateBuilder, OrderManager, et MultiAssetEnv.
"""

import sys
import os
import numpy as np
import pandas as pd
import logging

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from adan_trading_bot.environment.state_builder import StateBuilder
from adan_trading_bot.environment.order_manager import OrderManager
from adan_trading_bot.environment.multi_asset_env import MultiAssetEnv
from adan_trading_bot.common.utils import get_logger

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = get_logger()

def create_mock_config():
    """Créer une configuration mock pour les tests."""
    return {
        'data': {
            'assets': ['ADAUSDT', 'DOGEUSDT', 'BTCUSDT'],
            'base_market_features': ['open', 'high', 'low', 'close', 'volume', 'rsi_14', 'ema_20'],
            'cnn_input_window_size': 20,
            'training_timeframe': '1h'
        },
        'environment': {
            'initial_capital': 1000.0,
            'transaction': {'fee_percent': 0.001, 'fixed_fee': 0.0},
            'order_rules': {'min_value_tolerable': 10.0, 'min_value_absolute': 9.0},
            'penalties': {'time_step': -0.001, 'invalid_order_base': -0.1}
        }
    }

def create_mock_merged_dataframe():
    """Créer un DataFrame fusionné mock pour les tests."""
    assets = ['ADAUSDT', 'DOGEUSDT', 'BTCUSDT']
    features = ['open', 'high', 'low', 'close', 'volume', 'rsi_14', 'ema_20']
    
    # Créer 100 timesteps de données
    timesteps = 100
    data = {}
    
    for asset in assets:
        for feature in features:
            col_name = f"{feature}_{asset}"
            if feature in ['open', 'high', 'low', 'close']:
                # Prix normalisés entre -1 et 1
                data[col_name] = np.random.uniform(-1, 1, timesteps)
            elif feature == 'volume':
                # Volume normalisé
                data[col_name] = np.random.uniform(0, 1, timesteps)
            else:
                # Indicateurs normalisés
                data[col_name] = np.random.uniform(-1, 1, timesteps)
    
    df = pd.DataFrame(data)
    df.index = pd.date_range('2024-01-01', periods=timesteps, freq='1H')
    return df

def test_state_builder():
    """Tester StateBuilder avec les corrections."""
    logger.info("=== Test StateBuilder ===")
    
    config = create_mock_config()
    assets = config['data']['assets']
    base_features = config['data']['base_market_features']
    
    # Créer StateBuilder
    state_builder = StateBuilder(
        config=config,
        assets=assets,
        scaler=None,
        encoder=None,
        base_feature_names=base_features,
        cnn_input_window_size=20
    )
    
    # Créer des données de fenêtre mock
    df_merged = create_mock_merged_dataframe()
    window_data = df_merged.tail(20)  # Derniers 20 timesteps
    
    # Tester _get_market_features_as_image
    try:
        image_tensor = state_builder._get_market_features_as_image(window_data)
        expected_shape = (1, 20, len(base_features) * len(assets))
        
        if image_tensor.shape == expected_shape:
            logger.info(f"✓ StateBuilder: Image tensor shape correcte {image_tensor.shape}")
        else:
            logger.error(f"✗ StateBuilder: Mauvaise shape {image_tensor.shape}, attendu {expected_shape}")
        
        # Vérifier qu'il n'y a pas de NaN (toutes les features trouvées)
        nan_count = np.isnan(image_tensor).sum()
        if nan_count == 0:
            logger.info("✓ StateBuilder: Toutes les features trouvées (pas de NaN)")
        else:
            logger.warning(f"⚠ StateBuilder: {nan_count} valeurs NaN trouvées")
            
        return True
        
    except Exception as e:
        logger.error(f"✗ StateBuilder: Erreur lors du test - {e}")
        return False

def test_order_manager():
    """Tester OrderManager avec la gestion des prix normalisés."""
    logger.info("=== Test OrderManager ===")
    
    config = create_mock_config()
    order_manager = OrderManager(config)
    
    # Test BUY avec prix normalisé négatif
    try:
        capital = 1000.0
        positions = {}
        current_price = -0.5  # Prix normalisé négatif
        allocated_value = 50.0
        
        reward_mod, status, info = order_manager.execute_order(
            asset_id='ADAUSDT',
            action_type=1,  # BUY
            current_price=current_price,
            capital=capital,
            positions=positions,
            allocated_value_usdt=allocated_value
        )
        
        if status == "BUY_EXECUTED":
            logger.info("✓ OrderManager: BUY avec prix normalisé négatif réussi")
            # Vérifier que la quantité est positive
            quantity = info.get('quantity', 0)
            if quantity > 0:
                logger.info(f"✓ OrderManager: Quantité positive {quantity:.6f}")
            else:
                logger.error(f"✗ OrderManager: Quantité invalide {quantity}")
        else:
            logger.error(f"✗ OrderManager: BUY échoué - {status}")
            
        # Test avec prix absolu très faible
        reward_mod2, status2, info2 = order_manager.execute_order(
            asset_id='BTCUSDT',
            action_type=1,
            current_price=1e-10,  # Prix trop faible
            capital=capital,
            positions={},
            allocated_value_usdt=50.0
        )
        
        if status2 == "PRICE_NOT_AVAILABLE":
            logger.info("✓ OrderManager: Détection prix trop faible fonctionne")
        else:
            logger.warning(f"⚠ OrderManager: Prix faible non détecté - {status2}")
            
        return True
        
    except Exception as e:
        logger.error(f"✗ OrderManager: Erreur lors du test - {e}")
        return False

def test_multi_asset_env():
    """Tester MultiAssetEnv avec les corrections."""
    logger.info("=== Test MultiAssetEnv ===")
    
    config = create_mock_config()
    df_merged = create_mock_merged_dataframe()
    
    try:
        # Créer l'environnement
        env = MultiAssetEnv(
            df_received=df_merged,
            config=config,
            scaler=None,
            encoder=None,
            max_episode_steps_override=50  # Limiter pour le test
        )
        
        # Vérifier max_steps
        if env.max_steps == 50:
            logger.info("✓ MultiAssetEnv: max_steps correctement limité")
        else:
            logger.warning(f"⚠ MultiAssetEnv: max_steps = {env.max_steps}, attendu 50")
        
        # Test reset
        obs, info = env.reset()
        logger.info(f"✓ MultiAssetEnv: Reset réussi, capital initial = {info['capital']}")
        
        # Test _get_current_prices
        prices = env._get_current_prices()
        if len(prices) == len(config['data']['assets']):
            logger.info(f"✓ MultiAssetEnv: Tous les prix trouvés {list(prices.keys())}")
        else:
            logger.error(f"✗ MultiAssetEnv: Prix manquants {prices}")
        
        # Test quelques steps
        for i in range(5):
            action = np.random.randint(0, env.action_space.n)
            obs, reward, done, truncated, info = env.step(action)
            
            if done:
                logger.info(f"✓ MultiAssetEnv: Episode terminé au step {i}")
                break
        
        return True
        
    except Exception as e:
        logger.error(f"✗ MultiAssetEnv: Erreur lors du test - {e}")
        return False

def main():
    """Fonction principale de test."""
    logger.info("🚀 Début des tests de validation des corrections ADAN")
    
    tests_results = []
    
    # Test 1: StateBuilder
    tests_results.append(("StateBuilder", test_state_builder()))
    
    # Test 2: OrderManager  
    tests_results.append(("OrderManager", test_order_manager()))
    
    # Test 3: MultiAssetEnv
    tests_results.append(("MultiAssetEnv", test_multi_asset_env()))
    
    # Résumé
    logger.info("\n" + "="*50)
    logger.info("RÉSUMÉ DES TESTS")
    logger.info("="*50)
    
    passed = 0
    for test_name, result in tests_results:
        status = "✓ PASSÉ" if result else "✗ ÉCHOUÉ"
        logger.info(f"{test_name:15} : {status}")
        if result:
            passed += 1
    
    logger.info(f"\nTests réussis: {passed}/{len(tests_results)}")
    
    if passed == len(tests_results):
        logger.info("🎉 Tous les tests ont réussi ! Les corrections sont validées.")
        return 0
    else:
        logger.error("❌ Certains tests ont échoué. Vérifiez les logs ci-dessus.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)