"""
Tests unitaires pour le module config_watcher.py
"""
import os
import tempfile
import time
import yaml
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from adan_trading_bot.common.config_watcher import ConfigWatcher, config_reactive

# Configuration de test
TEST_CONFIG_DIR = Path(tempfile.mkdtemp())
TEST_CONFIG = {
    'training': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 100
    },
    'environment': {
        'name': 'TradingEnv',
        'params': {
            'window_size': 50,
            'trading_fee': 0.001
        }
    }
}

# Création des fichiers de configuration de test
for filename, config in {
    'config.yaml': TEST_CONFIG,
    'train_config.yaml': TEST_CONFIG['training'],
    'environment_config.yaml': TEST_CONFIG['environment']
}.items():
    with open(TEST_CONFIG_DIR / filename, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)

class TestConfigWatcher:
    """Tests pour la classe ConfigWatcher"""

    def test_initialization(self):
        """Teste l'initialisation du ConfigWatcher"""
        watcher = ConfigWatcher(config_dir=TEST_CONFIG_DIR, validate=False)
        assert watcher is not None
        assert watcher.config_dir == TEST_CONFIG_DIR
        assert watcher.enabled is True
        assert watcher.validate_schemas is False
        watcher.stop()

    def test_load_initial_configs(self):
        """Teste le chargement initial des configurations"""
        watcher = ConfigWatcher(config_dir=TEST_CONFIG_DIR, validate=False)
        
        # Vérification du chargement des configurations
        assert 'training' in watcher.current_configs
        assert 'environment' in watcher.current_configs
        assert watcher.current_configs['training']['batch_size'] == 32
        assert watcher.current_configs['environment']['name'] == 'TradingEnv'
        
        watcher.stop()

    def test_register_callback(self):
        """Teste l'enregistrement d'un callback"""
        watcher = ConfigWatcher(config_dir=TEST_CONFIG_DIR, validate=False)
        
        # Création d'un mock pour le callback
        mock_callback = MagicMock()
        
        # Enregistrement du callback
        watcher.register_callback('training', mock_callback)
        
        # Vérification que le callback est bien enregistré
        assert mock_callback in watcher.callbacks['training']
        
        watcher.stop()

    def test_config_change_detection(self):
        """Teste la détection des changements de configuration"""
        watcher = ConfigWatcher(config_dir=TEST_CONFIG_DIR, validate=False)
        
        # Création d'un mock pour le callback
        mock_callback = MagicMock()
        watcher.register_callback('training', mock_callback)
        
        # Modification du fichier de configuration
        config_path = TEST_CONFIG_DIR / 'train_config.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Modification d'une valeur
        config['batch_size'] = 64
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        # Attente pour la détection du changement
        time.sleep(0.5)
        
        # Vérification que le callback a été appelé
        mock_callback.assert_called_once()
        
        # Vérification que la configuration a été mise à jour
        assert watcher.current_configs['training']['batch_size'] == 64
        
        watcher.stop()

    @patch('adan_trading_bot.common.config_watcher.config_validator.ConfigValidator')
    def test_config_validation(self, mock_validator):
        """Teste la validation des configurations"""
        # Configuration du mock du validateur
        mock_instance = MagicMock()
        mock_validator.return_value = mock_instance
        mock_instance.validate_train_config.return_value = True
        
        # Création du watcher avec validation activée
        watcher = ConfigWatcher(config_dir=TEST_CONFIG_DIR, validate=True)
        
        # Réinitialisation du mock pour ignorer les appels initiaux
        mock_instance.validate_train_config.reset_mock()
        
        # Modification du fichier de configuration
        config_path = TEST_CONFIG_DIR / 'train_config.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        config['batch_size'] = 128
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        # Attente pour la détection du changement
        time.sleep(0.5)
        
        # Vérification que la validation a été appelée
        mock_instance.validate_train_config.assert_called_once()
        
        # Vérification que la configuration a été mise à jour
        assert watcher.current_configs['training']['batch_size'] == 128
        
        watcher.stop()

    def test_config_reactive_decorator(self):
        """Teste le décorateur config_reactive"""
        # Création d'une classe de test avec une méthode décorée
        class TestClass:
            def __init__(self):
                self.called = False
                self.last_config = None
                self.last_changes = None
            
            @config_reactive('training')
            def on_training_config_change(self, config_type, new_config, changes):
                self.called = True
                self.last_config = new_config
                self.last_changes = changes
        
        # Création d'une instance de la classe de test
        test_obj = TestClass()
        
        # Création du watcher et enregistrement du callback
        watcher = ConfigWatcher(config_dir=TEST_CONFIG_DIR, validate=False)
        watcher.register_callback('training', test_obj.on_training_config_change)
        
        # Modification du fichier de configuration
        config_path = TEST_CONFIG_DIR / 'train_config.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        config['batch_size'] = 256
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        # Attente pour la détection du changement
        time.sleep(0.5)
        
        # Vérification que la méthode décorée a été appelée
        assert test_obj.called is True
        assert test_obj.last_config['batch_size'] == 256
        assert 'batch_size' in test_obj.last_changes
        
        watcher.stop()

    @classmethod
    def teardown_class(cls):
        """Nettoyage après les tests"""
        # Suppression des fichiers de test
        for filename in ['config.yaml', 'train_config.yaml', 'environment_config.yaml']:
            path = TEST_CONFIG_DIR / filename
            if path.exists():
                os.remove(path)
        
        # Suppression du répertoire de test
        if TEST_CONFIG_DIR.exists():
            os.rmdir(TEST_CONFIG_DIR)
