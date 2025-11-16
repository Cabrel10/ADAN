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

def test_observation_shapes_consistency():
    """Vérifie la cohérence des shapes d'observation."""
    main_config = load_config()
    workers_config_data = load_config("/home/morningstar/Documents/trading/bot/config/workers.yaml")

    # --- CORRECTION CENTRALE ---
    # Aligner la configuration du test avec les données disponibles
    assets_disponibles_pour_test = ["BTCUSDT", "XRPUSDT"]
    timeframes_disponibles_pour_test = ["5m", "1h", "4h"] # <--- NOUVEAU

    main_config['data']['assets'] = assets_disponibles_pour_test
    main_config['environment']['assets'] = assets_disponibles_pour_test
    
    envs = []
    for i in range(4):
        worker_key = f"w{i+1}"
        worker_specific_config = workers_config_data['workers'][worker_key]
        
        # --- CORRECTION CENTRALE ---
        # S'assurer que le worker de test n'utilise que les actifs ET timeframes disponibles
        worker_specific_config['assets'] = assets_disponibles_pour_test
        worker_specific_config['timeframes'] = timeframes_disponibles_pour_test # <--- CORRECTION
        
        # Override initial_balance in main_config for the test to ensure sufficient capital
        if i == 0: # w1
            main_config['portfolio']['initial_balance'] = 20.5
        elif i == 1: # w2
            main_config['portfolio']['initial_balance'] = 500.0
        elif i == 2: # w3
            main_config['portfolio']['initial_balance'] = 1000.0
        elif i == 3: # w4
            main_config['portfolio']['initial_balance'] = 750.0

        env = MultiAssetChunkedEnv(
            worker_id=i, 
            config=main_config, 
            worker_config=worker_specific_config
        )
        envs.append(env)
        
    for env in envs:
        obs, _ = env.reset()
        
        # --- Les assertions devraient maintenant passer ---
        assert '5m' in obs
        assert '1h' in obs
        assert '4h' in obs
        
        # Vérifier types
        assert isinstance(obs['5m'], np.ndarray)
        assert obs['5m'].dtype in [np.float32, np.float64]
        
        # Vérifier pas de NaN
        assert not np.any(np.isnan(obs['5m']))
        assert not np.any(np.isnan(obs['1h']))
        assert not np.any(np.isnan(obs['4h']))

        # Dynamically check shapes based on config
        config = load_config()
        features_config = config['data']['features_config']['timeframes']
        
        for tf in ['5m', '1h', '4h']:
            if tf in obs:
                expected_features = (
                    len(features_config[tf].get('price', [])) +
                    len(features_config[tf].get('volume', [])) +
                    len(features_config[tf].get('indicators', []))
                )
                # The shape is (window_size, num_features)
                assert obs[tf].shape[1] == expected_features, f"Mismatch for {tf}"
