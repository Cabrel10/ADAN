import pytest
import yaml
import os
import numpy as np

# Assuming MultiAssetChunkedEnv is importable from src.adan_trading_bot.environment
from src.adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

# Helper function to load YAML configuration
def load_config(config_path="/home/morningstar/Documents/trading/bot/config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def test_workers_step_synchronization():
    """Vérifie que tous les workers progressent de manière synchronisée"""
    main_config = load_config()
    workers_config_data = load_config("/home/morningstar/Documents/trading/bot/config/workers.yaml")

    envs = []
    for i in range(4):
        worker_key = f"w{i+1}"
        worker_specific_config = workers_config_data['workers'][worker_key]
        
        # Override initial_balance in main_config for the test to ensure sufficient capital
        if i == 0: # w1
            main_config['portfolio']['initial_balance'] = 20.5
        elif i == 1: # w2
            main_config['portfolio']['initial_balance'] = 500.0
        elif i == 2: # w3
            main_config['portfolio']['initial_balance'] = 1000.0
        elif i == 3: # w4
            main_config['portfolio']['initial_balance'] = 750.0

        env = MultiAssetChunkedEnv(worker_id=i, config=main_config, worker_config=worker_specific_config)
        envs.append(env)
    
    # Reset tous les envs
    for env in envs:
        env.reset()
    
    # Exécuter 100 steps synchronisés
    for step in range(100):
        actions = [env.action_space.sample() for env in envs]
        
        # Step en parallèle
        results = []
        for env, action in zip(envs, actions):
            obs, reward, done, truncated, info = env.step(action) # Added truncated
            results.append((obs, reward, done, truncated, info)) # Added truncated
        
        # Vérifier que les current_step sont identiques
        steps = [env.current_step for env in envs]
        assert len(set(steps)) == 1, f"Steps désynchronisés: {steps}"
