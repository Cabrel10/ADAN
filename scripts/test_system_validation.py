#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de validation système pour tester les corrections OrderManager et la nouvelle configuration.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Assurer que le package src est dans le PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '..'))

from src.adan_trading_bot.common.utils import load_config, get_logger
from src.adan_trading_bot.environment.multi_asset_env import MultiAssetEnv
from src.adan_trading_bot.environment.order_manager import OrderManager

logger = get_logger()

def create_test_data(assets, timeframe="1m", num_rows=100):
    """Créer des données de test simulées."""
    logger.info(f"Création de données de test pour {len(assets)} actifs, {num_rows} lignes")
    
    # Générer des timestamps
    timestamps = pd.date_range(start='2024-01-01', periods=num_rows, freq='1min')
    
    # Créer un DataFrame avec des colonnes pour chaque actif
    data = {'timestamp': timestamps}
    
    base_features = ["open", "high", "low", "close", "volume", "SMA_short", "RSI", "MACD"]
    
    for asset in assets:
        for feature in base_features:
            col_name = f"{feature}_{asset}"
            if feature in ["open", "high", "low", "close"]:
                # Prix réalistes pour crypto
                base_price = np.random.uniform(100, 1000)
                data[col_name] = base_price + np.cumsum(np.random.normal(0, 0.1, num_rows))
            elif feature == "volume":
                data[col_name] = np.random.uniform(1000, 10000, num_rows)
            else:
                # Indicateurs techniques normalisés (peuvent être négatifs)
                data[col_name] = np.random.normal(0, 1, num_rows)
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    logger.info(f"Données de test créées: {df.shape}, colonnes: {df.columns.tolist()[:5]}...")
    return df

def test_order_manager():
    """Test des corrections OrderManager."""
    logger.info("=== Test OrderManager ===")
    
    # Configuration basique
    config = {
        'environment': {
            'initial_capital': 1000.0,
            'transaction': {'fee_percent': 0.001},
            'order_rules': {
                'min_value_tolerable': 10.0,
                'min_value_absolute': 5.0
            },
            'penalties': {
                'invalid_order_base': -0.3,
                'out_of_funds': -0.5
            }
        }
    }
    
    order_manager = OrderManager(config)
    
    # Test 1: BUY avec prix normalisé négatif
    logger.info("Test 1: BUY avec prix normalisé négatif")
    capital = 1000.0
    positions = {}
    current_price = -0.5  # Prix normalisé négatif
    allocated_value = 50.0
    
    reward, status, info = order_manager.execute_order(
        "BTCUSDT", 1, current_price, capital, positions,
        allocated_value_usdt=allocated_value
    )
    
    logger.info(f"Résultat BUY: status={status}, reward={reward:.3f}, nouveau_capital=${info.get('new_capital', 'N/A')}")
    
    # Test 2: SELL sans position
    logger.info("Test 2: SELL sans position")
    reward, status, info = order_manager.execute_order(
        "ETHUSDT", 2, 0.3, capital, positions
    )
    
    logger.info(f"Résultat SELL sans position: status={status}, reward={reward:.3f}")
    
    # Test 3: SELL avec position existante
    if status in ["BUY_EXECUTED"] and "BTCUSDT" in positions:
        logger.info("Test 3: SELL avec position existante")
        reward, status, info = order_manager.execute_order(
            "BTCUSDT", 2, -0.3, info['new_capital'], positions
        )
        logger.info(f"Résultat SELL: status={status}, reward={reward:.3f}, nouveau_capital=${info.get('new_capital', 'N/A')}")
    
    logger.info("Test OrderManager terminé\n")

def test_configuration_loading(profile):
    """Test du chargement de la configuration."""
    logger.info(f"=== Test Configuration ({profile}) ===")
    
    try:
        config_path = f'config/data_config_{profile}.yaml'
        config = load_config(config_path)
        
        logger.info(f"Configuration chargée: {config_path}")
        logger.info(f"Actifs: {config.get('assets', [])}")
        logger.info(f"Timeframes: {config.get('timeframes_to_process', [])}")
        logger.info(f"Training timeframe: {config.get('training_timeframe', 'N/A')}")
        logger.info(f"Data source type: {config.get('data_source_type', 'N/A')}")
        
        # Vérifier les indicateurs
        indicators = config.get('indicators_by_timeframe', {})
        for tf, inds in indicators.items():
            logger.info(f"Indicateurs {tf}: {len(inds)} configurés")
        
        logger.info("Test configuration réussi\n")
        return config
        
    except Exception as e:
        logger.error(f"Erreur chargement configuration: {e}")
        return None

def test_multiasset_env(config):
    """Test de MultiAssetEnv avec la nouvelle configuration."""
    logger.info("=== Test MultiAssetEnv ===")
    
    try:
        # Créer des données de test
        assets = config.get('assets', ['BTCUSDT', 'ETHUSDT'])[:3]  # Limiter à 3 pour test
        test_data = create_test_data(assets, config.get('training_timeframe', '1m'))
        
        # Créer une config de test limitée aux actifs testés
        test_config = config.copy()
        test_config['assets'] = assets
        
        # Wrapper la config dans le format attendu par MultiAssetEnv
        wrapped_config = {
            'data': test_config,
            'environment': {
                'initial_capital': 1000.0,
                'transaction': {'fee_percent': 0.001},
                'order_rules': {
                    'min_value_tolerable': 10.0,
                    'min_value_absolute': 5.0
                },
                'penalties': {
                    'invalid_order_base': -0.3,
                    'out_of_funds': -0.5
                }
            }
        }
        
        # Initialiser l'environnement
        logger.info("Initialisation MultiAssetEnv...")
        env = MultiAssetEnv(test_data, wrapped_config)
        
        logger.info(f"Environnement créé avec {len(env.assets)} actifs")
        logger.info(f"Features de base: {len(env.base_feature_names)} features")
        logger.info(f"Espace d'action: {env.action_space}")
        logger.info(f"Shape image CNN: {env.image_shape}")
        
        # Test reset
        logger.info("Test reset...")
        obs, info = env.reset()
        logger.info(f"Observation shape: image={obs['image_features'].shape}, vector={obs['vector_features'].shape}")
        
        # Test quelques actions
        logger.info("Test actions...")
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            logger.info(f"Step {i+1}: action={action}, reward={reward:.3f}, capital=${env.capital:.2f}")
            
            if terminated or truncated:
                break
        
        logger.info("Test MultiAssetEnv réussi\n")
        return True
        
    except Exception as e:
        logger.error(f"Erreur MultiAssetEnv: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Validation système ADAN")
    parser.add_argument('--profile', type=str, default='cpu', 
                       choices=['cpu', 'gpu'], 
                       help='Profil de configuration à tester')
    parser.add_argument('--skip-env', action='store_true', 
                       help='Ignorer le test MultiAssetEnv')
    
    args = parser.parse_args()
    
    logger.info("DÉMARRAGE VALIDATION SYSTÈME ADAN")
    logger.info("=" * 50)
    
    # Test 1: OrderManager
    test_order_manager()
    
    # Test 2: Configuration
    config = test_configuration_loading(args.profile)
    if not config:
        logger.error("Échec du test de configuration, arrêt")
        return 1
    
    # Test 3: MultiAssetEnv (si demandé)
    if not args.skip_env:
        if not test_multiasset_env(config):
            logger.error("Échec du test MultiAssetEnv")
            return 1
    else:
        logger.info("Test MultiAssetEnv ignoré (--skip-env)")
    
    logger.info("=" * 50)
    logger.info("VALIDATION SYSTÈME TERMINÉE AVEC SUCCÈS")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)