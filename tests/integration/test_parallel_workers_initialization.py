import pytest
import yaml
import os
from multiprocessing import Pool

# Assuming MultiAssetChunkedEnv is importable from src.adan_trading_bot.environment
from src.adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

# Helper function to load YAML configuration
def load_config(config_path="/home/morningstar/Documents/trading/bot/config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Helper function for worker initialization in a separate process
def init_worker(worker_id_and_config):
    worker_id, worker_specific_config, main_config = worker_id_and_config
    try:
        # Override initial_balance in main_config for the test to ensure sufficient capital
        # This is a simplified approach for integration testing, actual capital tiers are more complex
        if worker_id == 0: # w1
            main_config['portfolio']['initial_balance'] = 20.5
        elif worker_id == 1: # w2
            main_config['portfolio']['initial_balance'] = 500.0
        elif worker_id == 2: # w3
            main_config['portfolio']['initial_balance'] = 1000.0
        elif worker_id == 3: # w4
            main_config['portfolio']['initial_balance'] = 750.0

        env = MultiAssetChunkedEnv(
            worker_id=worker_id,
            config=main_config,
            worker_config=worker_specific_config
        )
        return {
            'worker_id': worker_id,
            'success': True,
            'tier': env.tier,
            'max_positions': env.max_positions,
            'force_trade': env.force_trade_steps_by_tf,
            'initial_equity': env.portfolio.initial_equity
        }
    except Exception as e:
        return {'worker_id': worker_id, 'success': False, 'error': str(e)}

def test_all_workers_initialize_successfully():
    """Vérifie que les 4 workers s'initialisent en parallèle"""
    main_config = load_config()
    workers_config_data = load_config("/home/morningstar/Documents/trading/bot/config/workers.yaml")

    worker_configs_for_pool = []
    for i in range(4):
        worker_key = f"w{i+1}"
        worker_specific_config = workers_config_data['workers'][worker_key]
        worker_configs_for_pool.append((i, worker_specific_config, main_config))

    with Pool(4) as pool:
        results = pool.map(init_worker, worker_configs_for_pool)
    
    # Vérifier tous succès
    assert all(r['success'] for r in results), f"Some workers failed to initialize: {[r['error'] for r in results if not r['success']]}"
    
    # Vérifier configurations différentes
    tiers = sorted([r['tier'] for r in results])
    assert tiers == [1, 3, 4, 5] # Sorted tiers from w1, w2, w4, w3

    # Verify max_positions
    max_positions = sorted([r['max_positions'] for r in results])
    assert max_positions == [1, 3, 4, 5] # Sorted max_positions from w1, w2, w4, w3

    # Verify initial_equity
    initial_equities = sorted([r['initial_equity'] for r in results])
    assert initial_equities == [20.5, 500.0, 750.0, 1000.0] # Sorted initial_equities

def test_independent_force_trade_states():
    """Vérifie que les force trade states sont indépendants"""
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
    
    # Modifier force trade count de W1 uniquement
    envs[0].daily_forced_trades_count = 5
    
    # Vérifier que les autres restent à 0
    assert envs[1].daily_forced_trades_count == 0
    assert envs[2].daily_forced_trades_count == 0
    assert envs[3].daily_forced_trades_count == 0
