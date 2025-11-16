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
def worker3_env():
    """Crée une instance de l'environnement pour le Worker 3 avec la configuration correcte."""
    main_config = load_config()
    workers_config_data = load_config("/home/morningstar/Documents/trading/bot/config/workers.yaml")
    worker_specific_config = workers_config_data['workers']['w3']

    # Surcharger le capital initial pour correspondre au tier du worker (ex: 1000$ pour Tier 5)
    main_config['portfolio']['initial_balance'] = 1000.0
    
    # Passer le worker_id et la configuration spécifique
    env = MultiAssetChunkedEnv(
        worker_id=2,  # ID pour le Worker 3
        config=main_config, 
        worker_config=worker_specific_config # Passer la config spécifique
    )
    return env

# Test 2.1.5: Position Limit Enforcement
def test_worker3_position_limit(worker3_env):
    """Vérifie que max_positions=5 est respecté via l'API publique."""
    env = worker3_env
    env.portfolio.reset()

    # S'assurer que le portefeuille est vide au départ
    assert len(env.portfolio._get_open_positions()) == 0

    # Utiliser une valeur notionnelle fixe et valide pour les tests
    trade_notional = 12.0  # > 11.0 USDT minimum

    # 1. Ouvrir cinq positions (valides)
    receipt1 = env.portfolio.open_position(
        asset='BTCUSDT', 
        price=50000, 
        size=trade_notional / 50000,  # Notional de 15 USDT
        stop_loss_pct=0.02, 
        take_profit_pct=0.04, 
        timestamp=datetime.now(),
        current_step=1
    )
    assert receipt1 is not None, "La première position aurait dû s'ouvrir"
    assert len(env.portfolio._get_open_positions()) == 1

    receipt2 = env.portfolio.open_position(
        asset='ETHUSDT', 
        price=3000, 
        size=trade_notional / 3000, # Notional de 15 USDT
        stop_loss_pct=0.02, 
        take_profit_pct=0.04, 
        timestamp=datetime.now(),
        current_step=2
    )
    assert receipt2 is not None, "La deuxième position aurait dû s'ouvrir"
    assert len(env.portfolio._get_open_positions()) == 2

    receipt3 = env.portfolio.open_position(
        asset='SOLUSDT', 
        price=100, 
        size=trade_notional / 100, # Notional de 15 USDT
        stop_loss_pct=0.02, 
        take_profit_pct=0.04, 
        timestamp=datetime.now(),
        current_step=3
    )
    assert receipt3 is not None, "La troisième position aurait dû s'ouvrir"
    assert len(env.portfolio._get_open_positions()) == 3

    receipt4 = env.portfolio.open_position(
        asset='XRPUSDT', 
        price=0.5, 
        size=trade_notional / 0.5, # Notional de 15 USDT
        stop_loss_pct=0.02, 
        take_profit_pct=0.04, 
        timestamp=datetime.now(),
        current_step=4
    )
    assert receipt4 is not None, "La quatrième position aurait dû s'ouvrir"
    assert len(env.portfolio._get_open_positions()) == 4

    receipt5 = env.portfolio.open_position(
        asset='ADAUSDT', 
        price=0.3, 
        size=trade_notional / 0.3, # Notional de 15 USDT
        stop_loss_pct=0.02, 
        take_profit_pct=0.04, 
        timestamp=datetime.now(),
        current_step=5
    )
    assert receipt5 is not None, "La cinquième position aurait dû s'ouvrir"
    assert len(env.portfolio._get_open_positions()) == 5

    # 2. Tenter d'ouvrir une sixième position (doit être refusée)
    receipt6 = env.portfolio.open_position(
        asset='BNBUSDT', 
        price=300, 
        size=trade_notional / 300, # Notional de 15 USDT
        stop_loss_pct=0.02, 
        take_profit_pct=0.04, 
        timestamp=datetime.now(),
        current_step=6
    )
    assert receipt6 is None, "La sixième position aurait dû être refusée (limite de 5)"
    assert len(env.portfolio._get_open_positions()) == 5, "Le nombre de positions ouvertes ne doit pas changer"
