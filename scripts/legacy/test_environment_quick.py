#!/usr/bin/env python3
"""
Test rapide de l'environnement ADAN Trading Bot
"""

import sys
import os
import logging
import numpy as np
import torch

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
import yaml

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
        'risk_config.yaml'
    ]
    
    config = {}
    for config_file in config_files:
        config_path = os.path.join(config_dir, config_file)
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    config.update(file_config)
    
    # Ajouter les sections manquantes si elles n'existent pas
    if 'portfolio' not in config:
        config['portfolio'] = {
            'initial_capital': 1000.0,
            'max_position_size': 0.1,
            'transaction_cost': 0.001
        }
    
    if 'risk_management' not in config:
        config['risk_management'] = {
            'max_drawdown': 0.2,
            'var_confidence': 0.95,
            'position_size_limit': 0.1,
            'capital_tiers': [
                {'min_capital': 0, 'max_capital': 1000, 'max_position_size': 0.05},
                {'min_capital': 1000, 'max_capital': 10000, 'max_position_size': 0.1},
                {'min_capital': 10000, 'max_capital': float('inf'), 'max_position_size': 0.15}
            ]
        }
    
    return config

def test_environment_initialization():
    """Test d'initialisation de l'environnement"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Testing environment initialization...")
    
    try:
        # Charger la configuration
        config = load_config()
        
        # Créer l'environnement
        env = MultiAssetChunkedEnv(config=config)
        logger.info("✅ Environment initialized successfully")
        
        # Test reset
        obs, info = env.reset()
        if isinstance(obs, dict):
            logger.info(f"✅ Environment reset successful. Obs keys: {list(obs.keys())}")
            if 'observation' in obs:
                logger.info(f"   Observation shape: {obs['observation'].shape}")
        else:
            logger.info(f"✅ Environment reset successful. Obs shape: {obs.shape}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        logger.info(f"✅ Environment step successful. Reward: {reward:.4f}")
        
        # Test quelques steps supplémentaires
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            logger.info(f"Step {i+1}: Reward: {reward:.4f}, Done: {terminated or truncated}")
            
            if terminated or truncated:
                obs, info = env.reset()
                logger.info("Environment reset after episode end")
        
        env.close()
        logger.info("✅ Environment test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_performance():
    """Test de performance de l'environnement"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Testing environment performance...")
    
    try:
        # Charger la configuration
        config = load_config()
        
        # Créer l'environnement
        env = MultiAssetChunkedEnv(config=config)
        
        # Test de performance
        import time
        start_time = time.time()
        
        obs, info = env.reset()
        
        num_steps = 100
        total_reward = 0
        
        for i in range(num_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                obs, info = env.reset()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        steps_per_second = num_steps / elapsed_time
        
        logger.info(f"✅ Performance test completed:")
        logger.info(f"   - Steps: {num_steps}")
        logger.info(f"   - Time: {elapsed_time:.2f}s")
        logger.info(f"   - Steps/sec: {steps_per_second:.2f}")
        logger.info(f"   - Total reward: {total_reward:.4f}")
        logger.info(f"   - Avg reward: {total_reward/num_steps:.4f}")
        
        env.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale"""
    logger = setup_logging()
    logger.info("🚀 Starting ADAN environment quick tests...")
    
    # Test d'initialisation
    if not test_environment_initialization():
        logger.error("❌ Environment initialization test failed")
        return False
    
    # Test de performance
    if not test_environment_performance():
        logger.error("❌ Environment performance test failed")
        return False
    
    logger.info("🎉 All environment tests passed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)