import pytest
import yaml
import os
import numpy as np
import pickle
from datetime import datetime # Added this import

# Assuming MultiAssetChunkedEnv and DBE are importable from src.adan_trading_bot.environment
# You might need to adjust these imports based on your actual project structure
from src.adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
# from src.adan_trading_bot.environment.dynamic_behavior_engine import DynamicBehaviorEngine # Assuming DBE is part of MultiAssetChunkedEnv or can be accessed

# Helper function to load YAML configuration
def load_config(config_path="/home/morningstar/Documents/trading/bot/config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

@pytest.fixture(scope="module")
def worker1_env():
    main_config = load_config() # This loads config/config.yaml
    workers_config_data = load_config("/home/morningstar/Documents/trading/bot/config/workers.yaml") # Load workers.yaml
    
    # Get the specific worker's config (e.g., w1)
    worker_specific_config = workers_config_data['workers']['w1']
    
    # Pass the main_config as 'config' and the worker_specific_config as 'worker_config'
    env = MultiAssetChunkedEnv(worker_id=0, config=main_config, worker_config=worker_specific_config)
    return env

# Test 2.1.1: Initialisation Worker 1
def test_worker1_initialization(worker1_env):
    """Vérifie l'initialisation complète du Worker 1"""
    env = worker1_env
    
    # Validations critiques
    assert env.worker_id == 0
    # These values should come from config/workers.yaml for w1
    assert env.tier == 1 
    assert env.max_positions == 1
    assert env.force_trade_steps_by_tf == {'5m': 144, '1h': 12, '4h': 3}
    assert env.daily_max_forced_trades == 10 # This is a global cap, not worker specific from config/workers.yaml
    assert env.daily_forced_trades_count == 0
    assert hasattr(env, 'dbe')
    assert env.portfolio.initial_equity == 20.5 # Corrected to initial_equity

# Test 2.1.2: Force Trade avec Daily Cap
def test_worker1_force_trade_daily_cap(worker1_env):
    """Vérifie que le daily cap est respecté"""
    env = worker1_env
    env.daily_forced_trades_count = 10  # Simuler cap atteint
    
    # Assuming _force_trade returns 0.0 if cap is reached and no trade is forced
    result = env._force_trade('5m')
    assert result == 0.0  # Doit retourner 0 si cap atteint
    # assert "Daily forced trade limit" in env.last_log_message # This would require capturing logs

# Test 2.1.3: Vérification DBE Persistence
def test_worker1_dbe_persistence(worker1_env):
    """Vérifie que l'état de l'objet DBE est sauvegardé/restauré correctement."""
    env = worker1_env
    dbe_state_file = f"dbe_state_{env.worker_id}.pkl"

    try:
        # 1. Simuler une modification de l'état du DBE
        env.dbe.decision_history = [10, 20, 30]

        # 2. Sauvegarder l'objet DBE directement
        with open(dbe_state_file, 'wb') as f:
            pickle.dump(env.dbe, f) 
        assert os.path.exists(dbe_state_file)

        # 3. Charger l'objet dans une nouvelle instance d'environnement
        main_config = load_config()
        workers_config_data = load_config("/home/morningstar/Documents/trading/bot/config/workers.yaml")
        worker_specific_config = workers_config_data['workers']['w1']
        env2 = MultiAssetChunkedEnv(worker_id=0, config=main_config, worker_config=worker_specific_config)
        
        with open(dbe_state_file, 'rb') as f:
            loaded_dbe = pickle.load(f)
        
        # Remplacer le DBE par défaut par celui qui a été chargé
        env2.dbe = loaded_dbe

        # 4. Vérifier que l'état a été correctement restauré
        assert hasattr(env2.dbe, 'decision_history')
        assert env2.dbe.decision_history == [10, 20, 30]

    finally:
        # Nettoyer le fichier de test
        if os.path.exists(dbe_state_file):
            os.remove(dbe_state_file)

# Test 2.1.4: Validation CNN Input Shapes
def test_worker1_cnn_shapes(worker1_env):
    """Vérifie que les shapes CNN correspondent aux données de la configuration."""
    env = worker1_env
    obs, _ = env.reset()

    # Charger la configuration pour vérifier le nombre de features attendu
    config = load_config()
    features_config = config['data']['features_config']['timeframes']

    # Vérifier les shapes pour chaque timeframe
    for tf in ['5m', '1h', '4h']:
        if tf in obs:
            expected_features = (
                len(features_config[tf].get('price', [])) +
                len(features_config[tf].get('volume', [])) +
                len(features_config[tf].get('indicators', []))
            )
            # La forme est (window_size, num_features)
            assert obs[tf].shape[1] == expected_features, f"Mismatch for {tf}"

# Test 2.1.5: Position Limit Enforcement
def test_worker1_position_limit(worker1_env):
    """Vérifie que max_positions=1 est respecté via l'API publique."""
    env = worker1_env
    env.portfolio.reset()

    # S'assurer que le portefeuille est vide au départ
    assert len(env.portfolio._get_open_positions()) == 0

    # 1. Ouvrir une première position (valide et réalisable)
    # Notional : 15 USDT (15 / 50000 = 0.0003 BTC)
    receipt1 = env.portfolio.open_position(
        asset='BTCUSDT',
        price=50000,
        size=0.0003, # <-- Taille de position corrigée
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        timestamp=datetime.now(),
        current_step=1
    )
    
    assert receipt1 is not None, "La première position (valide) aurait dû être ouverte."
    assert len(env.portfolio._get_open_positions()) == 1, "Il devrait y avoir une position ouverte."

    # 2. Tenter d'ouvrir une deuxième position (doit échouer à cause de la limite)
    receipt2 = env.portfolio.open_position(
        asset='ETHUSDT',
        price=3000,
        size=0.005, # Notional de 15 USDT
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        timestamp=datetime.now(),
        current_step=2
    )
    
    assert receipt2 is None, "La deuxième ouverture de position aurait dû échouer (limite de 1 position)."
    assert len(env.portfolio._get_open_positions()) == 1, "Le nombre de positions ouvertes ne doit pas avoir changé."