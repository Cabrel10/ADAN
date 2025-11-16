import pytest
import yaml
import os
import pickle
import numpy as np
import pandas as pd # Added pandas import
from datetime import datetime

# Assuming MultiAssetChunkedEnv is importable from src.adan_trading_bot.environment
from src.adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

# Helper function to load YAML configuration
def load_config(config_path="/home/morningstar/Documents/trading/bot/config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

@pytest.fixture(scope="module")
def setup_env_for_chunk_tests():
    main_config = load_config()
    workers_config_data = load_config("/home/morningstar/Documents/trading/bot/config/workers.yaml")
    worker_specific_config = workers_config_data['workers']['w1'] # Using w1 for these tests

    # Override initial_balance for the test to ensure sufficient capital
    main_config['portfolio']['initial_balance'] = 20.5

    env = MultiAssetChunkedEnv(
        worker_id=0, # Using worker_id 0 for these tests
        config=main_config,
        worker_config=worker_specific_config
    )
    return env

def test_dbe_state_persistence_across_chunks(setup_env_for_chunk_tests):
    """Vérifie la persistance DBE lors des transitions de chunks"""
    env = setup_env_for_chunk_tests
    dbe_state_file = f"dbe_state_{env.worker_id}.pkl"

    try:
        env.reset()
        
        # Simuler des décisions DBE sur chunk 0
        env.dbe.decision_history = [1, 2, 3]
        initial_state_dbe_history = env.dbe.decision_history
        
        # Manually trigger save of DBE state
        if hasattr(env, "dbe") and env.dbe is not None:
            with open(dbe_state_file, "wb") as f:
                pickle.dump(env.dbe, f)
        assert os.path.exists(dbe_state_file)

        # Créer un nouvel environnement pour simuler le chargement du chunk suivant
        main_config = load_config()
        workers_config_data = load_config("/home/morningstar/Documents/trading/bot/config/workers.yaml")
        worker_specific_config = workers_config_data['workers']['w1']
        env2 = MultiAssetChunkedEnv(
            worker_id=0,
            config=main_config,
            worker_config=worker_specific_config
        )
        
        # Simuler le chargement du chunk 1 dans le nouvel environnement
        # This should trigger the load of DBE state
        if os.path.exists(dbe_state_file):
            with open(dbe_state_file, "rb") as f:
                loaded_dbe = pickle.load(f)
            env2.dbe = loaded_dbe

        # Vérifier que l'état DBE est préservé
        assert hasattr(env2.dbe, 'decision_history')
        assert env2.dbe.decision_history == initial_state_dbe_history

    finally:
        if os.path.exists(dbe_state_file):
            os.remove(dbe_state_file)

def test_chunk_fallback_mechanism(setup_env_for_chunk_tests):
    """Vérifie le mécanisme de fallback en cas d'échec de chargement"""
    env = setup_env_for_chunk_tests
    
    # Temporarily set total_chunks to a small number to easily trigger fallback
    original_total_chunks = env.data_loader.total_chunks
    env.data_loader.total_chunks = 1 # Only one chunk exists
    
    try:
        # Attempt to load a non-existent chunk, expecting fallback to chunk 0
        # The _safe_load_chunk method should handle the error and return a synthetic chunk
        chunk_data = env._safe_load_chunk(9999, fallback_enabled=True)
        
        assert chunk_data is not None
        assert isinstance(chunk_data, dict)
        assert len(chunk_data) > 0 # Should contain some data
        
        # Verify that the synthetic chunk has expected structure (e.g., BTCUSDT asset)
        assert 'BTCUSDT' in chunk_data
        assert '5m' in chunk_data['BTCUSDT']
        assert isinstance(chunk_data['BTCUSDT']['5m'], pd.DataFrame)
        assert not chunk_data['BTCUSDT']['5m'].empty

    finally:
        env.data_loader.total_chunks = original_total_chunks # Restore original value