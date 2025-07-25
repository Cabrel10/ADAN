#!/usr/bin/env python3
"""
Test simple de l'environnement d'entraînement ADAN
"""

import sys
import os
import logging
import numpy as np
import yaml
from datetime import datetime

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

def setup_logging():
    """Configuration du logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_config():
    """Charger la configuration depuis les fichiers YAML"""
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
    
    # Charger toutes les configurations nécessaires
    config_files = [
        'main_config.yaml',
        'data_config_cpu.yaml', 
        'environment_config.yaml',
        'trading_config.yaml',
        'reward_config.yaml',
        'risk_config.yaml',
        'train_config.yaml'
    ]
    
    config = {}
    for config_file in config_files:
        config_path = os.path.join(config_dir, config_file)
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    config.update(file_config)
    
    # Configuration de base pour le chargement des données
    data_config = {
        'data': {
            'data_dir': 'data/final',
            'chunk_size': 10000,
            'assets': ['BTC', 'ETH', 'SOL', 'XRP', 'ADA'],
            'timeframes': ['5m', '1h', '4h']
        },
        'state': {
            'window_size': 100,
            'timeframes': ['5m', '1h', '4h']
        },
        'portfolio': {
            'initial_balance': 1000.0,
            'max_leverage': 3.0,
            'risk_per_trade': 0.01
        },
        'trading': {
            'commission': 0.001,
            'slippage': 0.0005
        },
        'trading_rules': {
            'min_order_value_usdt': 10.0,
            'commission_pct': 0.001,
            'slippage_pct': 0.0005,
            'stop_loss': 0.02,
            'take_profit': 0.04
        }
    }
    
    # Fusionner les configurations
    full_config = {**data_config, **config}
    
    # Ajouter les sections manquantes si elles n'existent pas
    if 'portfolio' not in full_config:
        full_config['portfolio'] = {
            'initial_capital': 1000.0,
            'max_position_size': 0.1,
            'transaction_cost': 0.001
        }
    
    if 'risk_management' not in full_config:
        full_config['risk_management'] = {
            'max_drawdown': 0.2,
            'var_confidence': 0.95,
            'position_size_limit': 0.1,
            'capital_tiers': [
                {'min_capital': 0, 'max_capital': 1000, 'max_position_size': 0.05},
                {'min_capital': 1000, 'max_capital': 10000, 'max_position_size': 0.1},
                {'min_capital': 10000, 'max_capital': float('inf'), 'max_position_size': 0.15}
            ]
        }
    
    return full_config

def test_training_behavior():
    """Test du comportement d'entraînement"""
    logger = setup_logging()
    logger.info("🚀 Testing ADAN training behavior...")
    
    try:
        # Charger la configuration
        config = load_config()
        
        # Créer l'environnement
        logger.info("Creating environment...")
        env = MultiAssetChunkedEnv(config=config)
        logger.info("✅ Environment created successfully")
        
        # Reset de l'environnement
        obs, info = env.reset()
        logger.info(f"✅ Environment reset. Observation keys: {list(obs.keys())}")
        logger.info(f"   Price features shape: {obs['price_features'].shape}")
        logger.info(f"   Portfolio state shape: {obs['portfolio_state'].shape}")
        
        # Simulation d'un épisode d'entraînement
        logger.info("🎯 Starting training simulation...")
        
        total_reward = 0
        episode_length = 0
        max_steps = 100  # Limiter pour le test
        
        for step in range(max_steps):
            # Action aléatoire (0=hold, 1=buy, 2=sell pour chaque asset)
            action = env.action_space.sample()
            
            # Exécuter l'action
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            episode_length += 1
            
            # Log périodique
            if step % 20 == 0:
                portfolio_value = info.get('portfolio_value', 0)
                cash = info.get('cash', 0)
                logger.info(f"Step {step}: Reward={reward:.4f}, Portfolio={portfolio_value:.2f}, Cash={cash:.2f}")
            
            # Vérifier si l'épisode est terminé
            if terminated or truncated:
                logger.info(f"Episode ended at step {step}")
                obs, info = env.reset()
                break
        
        # Résultats finaux
        logger.info("📊 Training simulation results:")
        logger.info(f"   Total reward: {total_reward:.4f}")
        logger.info(f"   Episode length: {episode_length}")
        logger.info(f"   Average reward per step: {total_reward/episode_length:.4f}")
        logger.info(f"   Final portfolio value: {info.get('portfolio_value', 0):.2f}")
        
        # Test de performance
        logger.info("⚡ Performance test...")
        import time
        start_time = time.time()
        
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
        
        end_time = time.time()
        steps_per_second = 50 / (end_time - start_time)
        logger.info(f"   Performance: {steps_per_second:.2f} steps/second")
        
        env.close()
        logger.info("🎉 Training behavior test completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale"""
    success = test_training_behavior()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)