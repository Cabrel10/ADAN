import pytest
import yaml
import os
import numpy as np
import pickle
from datetime import datetime

# Assuming MultiAssetChunkedEnv and DBE are importable from src.adan_trading_bot.environment
from src.adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

# Helper function to load YAML configuration
def load_config(config_path="/home/morningstar/Documents/trading/bot/config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

@pytest.fixture(scope="module")
def worker2_env():
    """Crée une instance de l'environnement pour le Worker 2 avec la configuration correcte."""
    main_config = load_config()
    workers_config_data = load_config("/home/morningstar/Documents/trading/bot/config/workers.yaml")
    worker_specific_config = workers_config_data['workers']['w2']

    # Surcharger le capital initial pour correspondre au tier du worker (ex: 500$ pour Tier 3)
    main_config['portfolio']['initial_balance'] = 500.0
    
    # Passer le worker_id et la configuration spécifique
    env = MultiAssetChunkedEnv(
        worker_id=1,  # ID pour le Worker 2
        config=main_config, 
        worker_config=worker_specific_config # Passer la config spécifique
    )
    return env

# Test 2.1.1: Initialisation Worker 2
def test_worker2_initialization(worker2_env):
    """Vérifie l'initialisation complète du Worker 2"""
    env = worker2_env
    
    # Validations critiques
    assert env.worker_id == 1
    assert env.tier == 3 
    assert env.max_positions == 3
    assert env.force_trade_steps_by_tf == {'5m': 96, '1h': 8, '4h': 3}
    assert env.daily_max_forced_trades == 10 # Global cap
    assert env.daily_forced_trades_count == 0
    assert hasattr(env, 'dbe')
    # --- CORRECTION DE L'ASSERTION ---
    assert env.portfolio.initial_equity == 500.0 # Doit correspondre au capital du Tier 3

# Test 2.1.2: Force Trade avec Daily Cap
def test_worker2_force_trade_daily_cap(worker2_env):
    """Vérifie que le daily cap est respecté"""
    env = worker2_env
    env.daily_forced_trades_count = 10  # Simuler cap atteint
    
    result = env._force_trade('5m')
    assert result == 0.0  # Doit retourner 0 si cap atteint

# Test 2.1.3: Vérification DBE Persistence
def test_worker2_dbe_persistence(worker2_env):
    """Vérifie que l'état de l'objet DBE est sauvegardé/restauré correctement."""
    env = worker2_env
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
        worker_specific_config = workers_config_data['workers']['w2']
        env2 = MultiAssetChunkedEnv(worker_id=1, config=main_config, worker_config=worker_specific_config)
        
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
def test_worker2_cnn_shapes(worker2_env):
    """Vérifie que les shapes CNN correspondent aux données de la configuration."""
    env = worker2_env
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
def test_worker2_position_limit(worker2_env):
    """Vérifie que max_positions=3 est respecté."""
    env = worker2_env
    env.portfolio.reset()
    assert len(env.portfolio._get_open_positions()) == 0

    # Utiliser une valeur notionnelle fixe et valide pour les tests
    trade_notional = 12.0  # > 11.0 USDT minimum

    # 1. Ouvrir 3 positions valides
    # Trade 1
    receipt1 = env.portfolio.open_position(
        asset='BTCUSDT', price=50000, size=trade_notional / 50000,
        stop_loss_pct=0.02, take_profit_pct=0.04, timestamp=datetime.now(), current_step=1
    )
    assert receipt1 is not None, "La première position aurait dû s'ouvrir."
    
    # Trade 2
    receipt2 = env.portfolio.open_position(
        asset='ETHUSDT', price=3000, size=trade_notional / 3000,
        stop_loss_pct=0.02, take_profit_pct=0.04, timestamp=datetime.now(), current_step=2
    )
    assert receipt2 is not None, "La deuxième position aurait dû s'ouvrir."

    # Trade 3
    receipt3 = env.portfolio.open_position(
        asset='SOLUSDT', price=100, size=trade_notional / 100,
        stop_loss_pct=0.02, take_profit_pct=0.04, timestamp=datetime.now(), current_step=3
    )
    assert receipt3 is not None, "La troisième position aurait dû s'ouvrir."
    assert len(env.portfolio._get_open_positions()) == 3

    # 2. Tenter d'ouvrir une 4ème position
    receipt4 = env.portfolio.open_position(
        asset='XRPUSDT', price=0.5, size=trade_notional / 0.5,
        stop_loss_pct=0.02, take_profit_pct=0.04, timestamp=datetime.now(), current_step=4
    )
    assert receipt4 is None, "La quatrième position aurait dû être refusée."
    assert len(env.portfolio._get_open_positions()) == 3
