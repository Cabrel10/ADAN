"""
Unit tests for EnhancedConfigManager.
Tests hot-reload capabilities, validation, and configuration management.
"""

import unittest
import tempfile
import os
import json
import yaml
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from adan_trading_bot.common.enhanced_config_manager import EnhancedConfigManager


class TestEnhancedConfigManager(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Create test configuration files
        self._create_test_configs()

        # Initialize config manager
        self.config_manager = EnhancedConfigManager(
            config_dir=str(self.config_dir),
            enable_hot_reload=False  # Disable for unit tests
        )

    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'config_manager'):
            self.config_manager.shutdown()

        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_configs(self):
        """Create test configuration files."""
        # Model configuration
        model_config = {
            "architecture": {
                "layers": [64, 32, 16],
                "activation": "relu",
                "dropout": 0.2
            },
            "diagnostics": {
                "use_grad_cam": True,
                "save_attention_maps": False
            }
        }

        with open(self.config_dir / "model.yaml", 'w') as f:
            yaml.dump(model_config, f)

        # Environment configuration
        env_config = {
            "initial_balance": 1000.0,
            "trading_fees": 0.001,
            "max_steps": 1000,
            "assets": ["BTCUSDT", "ETHUSDT"],
            "observation": {
                "shape": [3, 20, 10],
                "timeframes": ["5m", "1h"]
            },
            "memory": {
                "chunk_size": 1000,
                "max_chunks": 5
            }
        }

        with open(self.config_dir / "environment.yaml", 'w') as f:
            yaml.dump(env_config, f)

        # Trading configuration
        trading_config = {
            "futures_enabled": False,
            "leverage": 1.0,
            "commission_pct": 0.001,
            "min_order_value_usdt": 10.0
        }

        with open(self.config_dir / "trading.yaml", 'w') as f:
            yaml.dump(trading_config, f)

    def test_initialization(self):
        """Test configuration manager initialization."""
        self.assertIsNotNone(self.config_manager)
        self.assertEqual(str(self.config_manager.config_dir), str(self.config_dir))
        self.assertFalse(self.config_manager.enable_hot_reload)

    def test_load_configurations(self):
        """Test loading of configuration files."""
        # Check that configurations were loaded
        model_config = self.config_manager.get_config("model")
        self.assertIsNotNone(model_config)
        self.assertIn("architecture", model_config)
        self.assertEqual(model_config["architecture"]["activation"], "relu")

        env_config = self.config_manager.get_config("environment")
        self.assertIsNotNone(env_config)
        self.assertEqual(env_config["initial_balance"], 1000.0)
        self.assertEqual(len(env_config["assets"]), 2)

        trading_config = self.config_manager.get_config("trading")
        self.assertIsNotNone(trading_config)
        self.assertFalse(trading_config["futures_enabled"])

    def test_get_config_value(self):
        """Test getting specific configuration values using dot notation."""
        # Test nested value access
        activation = self.config_manager.get_config_value("model", "architecture.activation")
        self.assertEqual(activation, "relu")

        # Test array access
        assets = self.config_manager.get_config_value("environment", "assets")
        self.assertEqual(len(assets), 2)
        self.assertIn("BTCUSDT", assets)

        # Test default value
        missing_value = self.config_manager.get_config_value("model", "missing.key", "default")
        self.assertEqual(missing_value, "default")

    def test_set_config_value(self):
        """Test setting configuration values."""
        # Set a new value
        success = self.config_manager.set_config_value("model", "architecture.new_param", "test_value")
        self.assertTrue(success)

        # Verify the value was set
        new_value = self.config_manager.get_config_value("model", "architecture.new_param")
        self.assertEqual(new_value, "test_value")

        # Set nested value
        success = self.config_manager.set_config_value("model", "new_section.param", 42)
        self.assertTrue(success)

        nested_value = self.config_manager.get_config_value("model", "new_section.param")
        self.assertEqual(nested_value, 42)

    def test_configuration_validation(self):
        """Test configuration validation against schemas."""
        # Valid configuration should pass
        valid_config = {
            "architecture": {"layers": [32, 16]},
            "diagnostics": {"use_grad_cam": False}
        }

        is_valid = self.config_manager.validate_config("model", valid_config)
        self.assertTrue(is_valid)

        # Invalid configuration should fail
        invalid_config = {
            "invalid_key": "invalid_value"
        }

        is_valid = self.config_manager.validate_config("model", invalid_config)
        self.assertFalse(is_valid)

    def test_change_callbacks(self):
        """Test configuration change callbacks."""
        callback_called = threading.Event()
        callback_data = {}

        def test_callback(config_type, new_config, changes):
            callback_data['config_type'] = config_type
            callback_data['new_config'] = new_config
            callback_data['changes'] = changes
            callback_called.set()

        # Register callback
        self.config_manager.register_change_callback("model", test_callback)

        # Trigger a change
        self.config_manager.set_config_value("model", "test_param", "test_value")

        # Wait for callback
        callback_called.wait(timeout=1.0)

        # Verify callback was called
        self.assertTrue(callback_called.is_set())
        self.assertEqual(callback_data['config_type'], "model")
        self.assertIn("test_param", callback_data['changes'])

    def test_reload_config(self):
        """Test configuration reloading."""
        # Modify configuration file
        new_config = {
            "architecture": {
                "layers": [128, 64, 32],
                "activation": "tanh",
                "dropout": 0.3
            }
        }

        with open(self.config_dir / "model.yaml", 'w') as f:
            yaml.dump(new_config, f)

        # Reload configuration
        success = self.config_manager.reload_config("model")
        self.assertTrue(success)

        # Verify changes
        activation = self.config_manager.get_config_value("model", "architecture.activation")
        self.assertEqual(activation, "tanh")

        layers = self.config_manager.get_config_value("model", "architecture.layers")
        self.assertEqual(len(layers), 3)
        self.assertEqual(layers[0], 128)

    def test_status_information(self):
        """Test getting status information."""
        status = self.config_manager.get_status()

        self.assertIn("config_dir", status)
        self.assertIn("hot_reload_enabled", status)
        self.assertIn("loaded_configs", status)
        self.assertIn("last_loaded", status)

        # Check loaded configurations
        self.assertIn("model", status["loaded_configs"])
        self.assertIn("environment", status["loaded_configs"])
        self.assertIn("trading", status["loaded_configs"])

        # Check hot-reload status
        self.assertFalse(status["hot_reload_enabled"])

    def test_environment_variable_resolution(self):
        """Test environment variable resolution in configurations."""
        # Set environment variable
        os.environ['TEST_VALUE'] = 'test_env_value'

        try:
            # Create config with environment variable
            config_with_env = {
                "test_param": "${TEST_VALUE}",
                "nested": {
                    "env_param": "${TEST_VALUE}"
                }
            }

            with open(self.config_dir / "test_env.yaml", 'w') as f:
                yaml.dump(config_with_env, f)

            # Reload to pick up new config
            self.config_manager._load_single_config("test_env", self.config_dir / "test_env.yaml")

            # Verify environment variable was resolved
            test_value = self.config_manager.get_config_value("test_env", "test_param")
            self.assertEqual(test_value, "test_env_value")

            nested_value = self.config_manager.get_config_value("test_env", "nested.env_param")
            self.assertEqual(nested_value, "test_env_value")

        finally:
            # Clean up environment variable
            os.environ.pop('TEST_VALUE', None)

    def test_thread_safety(self):
        """Test thread safety of configuration access."""
        results = []
        errors = []

        def config_reader():
            try:
                for i in range(100):
                    config = self.config_manager.get_config("model")
                    if config:
                        results.append(config["architecture"]["activation"])
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def config_writer():
            try:
                for i in range(50):
                    self.config_manager.set_config_value("model", f"test_param_{i}", f"value_{i}")
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=config_reader))
        for _ in range(2):
            threads.append(threading.Thread(target=config_writer))

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)

        # Check results
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        self.assertGreater(len(results), 0)

    def test_missing_configuration(self):
        """Test handling of missing configurations."""
        # Try to get non-existent configuration
        missing_config = self.config_manager.get_config("non_existent")
        self.assertEqual(missing_config, {})

        # Try to get value from non-existent configuration
        missing_value = self.config_manager.get_config_value("non_existent", "some.key", "default")
        self.assertEqual(missing_value, "default")

        # Try to reload non-existent configuration
        success = self.config_manager.reload_config("non_existent")
        self.assertFalse(success)


class TestEnhancedConfigManagerHotReload(unittest.TestCase):
    """Test hot-reload functionality separately to avoid conflicts."""

    def setUp(self):
        """Set up test environment for hot-reload tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Create initial configuration
        self.test_config = {
            "test_param": "initial_value",
            "nested": {"param": 42}
        }

        self.config_file = self.config_dir / "test.yaml"
        with open(self.config_file, 'w') as f:
            yaml.dump(self.test_config, f)

    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'config_manager'):
            self.config_manager.shutdown()

        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('adan_trading_bot.common.enhanced_config_manager.ConfigWatcher')
    def test_hot_reload_initialization(self, mock_watcher):
        """Test hot-reload watcher initialization."""
        # Create config manager with hot-reload enabled
        self.config_manager = EnhancedConfigManager(
            config_dir=str(self.config_dir),
            enable_hot_reload=True
        )

        # Verify watcher was created
        mock_watcher.assert_called_once()

    def test_hot_reload_disabled(self):
        """Test behavior when hot-reload is disabled."""
        self.config_manager = EnhancedConfigManager(
            config_dir=str(self.config_dir),
            enable_hot_reload=False
        )

        # Verify no watcher was created
        self.assertIsNone(self.config_manager.config_watcher)


if __name__ == '__main__':
    unittest.main()
