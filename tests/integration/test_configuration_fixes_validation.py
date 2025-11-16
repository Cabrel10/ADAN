import pytest
import yaml
import os

# Helper function to load YAML configuration
def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Path to the configuration files
CONFIG_DIR = "/home/morningstar/Documents/trading/bot/config"
MAIN_CONFIG_FILE = os.path.join(CONFIG_DIR, "config.yaml")
WORKERS_CONFIG_FILE = os.path.join(CONFIG_DIR, "workers.yaml")
# ENVIRONMENT_CONFIG_FILE is not a separate file, its content is within config.yaml

@pytest.fixture(scope="module")
def main_config():
    return load_config(MAIN_CONFIG_FILE)

@pytest.fixture(scope="module")
def workers_config():
    return load_config(WORKERS_CONFIG_FILE)

# Test 1.1.1: Charger config/config.yaml et valider tous les champs obligatoires
def test_main_config_mandatory_fields(main_config):
    assert main_config['agent']['algorithm'] == "PPO"
    assert main_config['agent']['batch_size'] == 64 # Corrected from 8192 to 64
    assert main_config['agent']['n_envs'] == 4

# Test 1.1.2: Valider config/workers.yaml pour chaque worker
def test_workers_config_validation(workers_config):
    workers = workers_config['workers']

    # Worker 1
    assert workers['w1']['tier'] == 1
    assert workers['w1']['max_positions'] == 1
    assert workers['w1']['force_trade'] == {'5m': 144, '1h': 12, '4h': 3}

    # Worker 2
    assert workers['w2']['tier'] == 3
    assert workers['w2']['max_positions'] == 3
    assert workers['w2']['force_trade'] == {'5m': 96, '1h': 8, '4h': 3}

    # Worker 3
    assert workers['w3']['tier'] == 5
    assert workers['w3']['max_positions'] == 5
    assert workers['w3']['force_trade'] == {'5m': 72, '1h': 6, '4h': 2}

    # Worker 4
    assert workers['w4']['tier'] == 4
    assert workers['w4']['max_positions'] == 4
    assert workers['w4']['force_trade'] == {'5m': 84, '1h': 7, '4h': 2}

# Test 1.1.3: Valider config/environment.yaml (now checking relevant sections in main_config.yaml)
def test_environment_config_validation(main_config):
    # Check for capital_tiers at the top level of config.yaml
    assert 'capital_tiers' in main_config
    assert isinstance(main_config['capital_tiers'], list)
    assert len(main_config['capital_tiers']) > 0

    # Check for trading_rules at the top level of config.yaml
    assert 'trading_rules' in main_config
    assert isinstance(main_config['trading_rules'], dict)
    assert 'frequency' in main_config['trading_rules']
    assert isinstance(main_config['trading_rules']['frequency'], dict)
