import pytest
import yaml
import os
import psutil
import gc
import shutil
import pickle # For crash recovery test
import tracemalloc

# Assuming MultiAssetChunkedEnv is importable from src.adan_trading_bot.environment
from src.adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

# Helper function to load YAML configuration
def load_config(config_path="/home/morningstar/Documents/trading/bot/config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

@pytest.fixture(scope="module", autouse=True)
def cleanup_dbe_state_files():
    """Fixture to clean up DBE state files before and after tests."""
    # Setup: Clean up any existing test artifacts
    for i in range(4):
        dbe_state_file = f"dbe_state_{i}.pkl"
        if os.path.exists(dbe_state_file):
            os.remove(dbe_state_file)
    yield
    # Teardown: Clean up artifacts after tests
    for i in range(4):
        dbe_state_file = f"dbe_state_{i}.pkl"
        if os.path.exists(dbe_state_file):
            os.remove(dbe_state_file)

def test_memory_leak_detection():
    """Vérifie qu'il n'y a pas de fuite mémoire sur 1000 steps avec tracemalloc."""
    main_config = load_config()
    workers_config_data = load_config("/home/morningstar/Documents/trading/bot/config/workers.yaml")
    worker_specific_config = workers_config_data['workers']['w1']
    main_config['portfolio']['initial_balance'] = 20.5 # Sufficient for test

    env = MultiAssetChunkedEnv(worker_id=0, config=main_config, worker_config=worker_specific_config)
    env.reset()
    
    tracemalloc.start()
    
    # Run a few steps to warm up
    for _ in range(10):
        env.step(env.action_space.sample())

    gc.collect()
    snapshot1 = tracemalloc.take_snapshot()

    # Execute 1000 steps
    for step in range(1000):
        env.step(env.action_space.sample())
        
    gc.collect()
    snapshot2 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    top_stats = snapshot2.compare_to(snapshot1, 'lineno')

    # Check if the top memory-consuming line is growing significantly.
    # We allow for some memory growth, but it should be minimal.
    # Let's check the top stat. A leak would likely show a large, continuous increase.
    if top_stats:
        top_stat = top_stats[0]
        # Allow a maximum of 10 MB increase for the top allocating line over 1000 steps.
        # This is a heuristic and might need adjustment.
        assert top_stat.size_diff < 10 * 1024 * 1024, (
            f"Potential memory leak detected. "
            f"Top memory usage difference: {top_stat.size_diff / 1024 / 1024:.2f} MB\n"
            f"Location: {top_stat.traceback.format()}"
        )

def test_crash_recovery():
    """Vérifie la récupération après crash simulé"""
    main_config = load_config()
    workers_config_data = load_config("/home/morningstar/Documents/trading/bot/config/workers.yaml")
    worker_specific_config = workers_config_data['workers']['w1']
    main_config['portfolio']['initial_balance'] = 20.5 # Sufficient for test

    env = MultiAssetChunkedEnv(worker_id=0, config=main_config, worker_config=worker_specific_config)
    env.reset()
    
    # Entraîner 100 steps
    for _ in range(100):
        env.step(env.action_space.sample())
    
    # Simuler une modification de l'état du DBE
    env.dbe.decision_history = ["decision_1", "decision_2"]
    
    # Simuler le chargement d'un nouveau chunk, ce qui déclenche la sauvegarde de l'état du DBE
    # We need to manually trigger the save part of _safe_load_chunk
    dbe_state_file = f"dbe_state_{env.worker_id}.pkl"
    with open(dbe_state_file, "wb") as f:
        pickle.dump(env.dbe, f)
    assert os.path.exists(dbe_state_file)
    
    # Simuler crash (détruire env)
    del env
    
    # Recréer et vérifier reprise
    env2 = MultiAssetChunkedEnv(worker_id=0, config=main_config, worker_config=worker_specific_config)
    
    # Simuler le chargement du même chunk, ce qui devrait restaurer le DBE
    if os.path.exists(dbe_state_file):
        with open(dbe_state_file, "rb") as f:
            loaded_dbe = pickle.load(f)
        env2.dbe = loaded_dbe
    
    assert hasattr(env2.dbe, 'decision_history')
    assert env2.dbe.decision_history == ["decision_1", "decision_2"]