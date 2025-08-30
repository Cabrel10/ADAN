"""
Configuration globale pour les tests unitaires.
Ce fichier est automatiquement chargé par pytest avant l'exécution des tests.
"""
import os
import sys
from pathlib import Path
from typing import Generator

import pytest
from _pytest.monkeypatch import MonkeyPatch

# Ajouter le répertoire racine au PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Charger les variables d'environnement de test
os.environ["ENVIRONMENT"] = "test"
os.environ["LOG_LEVEL"] = "INFO"
os.environ["DEBUG"] = "True"

# Désactiver les appels réseau dans les tests
os.environ["PYTHONHTTPSVERIFY"] = "0"


@pytest.fixture(autouse=True)
def set_test_environment() -> Generator[None, None, None]:
    """
    Configure l'environnement de test.
    """
    # Sauvegarder l'état original des variables d'environnement
    original_environ = os.environ.copy()
    
    # Configuration spécifique aux tests
    os.environ["TESTING"] = "True"
    
    yield  # Les tests s'exécutent ici
    
    # Restaurer l'environnement original
    os.environ.clear()
    os.environ.update(original_environ)


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """
    Crée un répertoire temporaire pour les données de test.
    """
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    
    # Créer des sous-répertoires
    (data_dir / "raw").mkdir()
    (data_dir / "processed").mkdir()
    (data_dir / "models").mkdir()
    
    return data_dir


@pytest.fixture
def mock_config(tmp_path: Path, monkeypatch: MonkeyPatch) -> dict:
    """
    Retourne une configuration de test.
    """
    config = {
        "general": {
            "project_name": "adan_trading_bot_test",
            "log_level": "INFO",
            "random_seed": 42,
            "device": "cpu"
        },
        "data": {
            "data_dir": str(tmp_path / "data"),
            "raw_data_dir": "${data.data_dir}/raw",
            "processed_data_dir": "${data.data_dir}/processed",
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "timeframes": ["1h", "4h"]
        },
        "environment": {
            "initial_balance": 10000.0,
            "commission": 0.001,
            "max_position_size": 0.1,
            "max_drawdown": 0.2,
            "window_size": 50,
            "use_indicators": True
        },
        "model": {
            "name": "PPO",
            "network": {
                "hidden_sizes": [64, 64],
                "activation": "ReLU"
            },
            "training": {
                "total_timesteps": 1000,
                "batch_size": 32,
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5
            },
            "buffer": {
                "buffer_size": 1000,
                "batch_size": 32,
                "alpha": 0.6,
                "beta": 0.4,
                "beta_increment": 0.001
            }
        },
        "training": {
            "num_workers": 2,
            "save_freq": 100,
            "eval_freq": 50,
            "log_interval": 10
        }
    }
    
    # Appliquer la configuration comme variable d'environnement
    monkeypatch.setenv("CONFIG", str(config))
    
    return config


@pytest.fixture
def sample_experience() -> dict:
    """
    Retourne un échantillon d'expérience pour les tests.
    """
    return {
        "state": [0.1, 0.2, 0.3, 0.4, 0.5],
        "action": 1,
        "reward": 0.5,
        "next_state": [0.2, 0.3, 0.4, 0.5, 0.6],
        "done": False,
        "info": {}
    }
