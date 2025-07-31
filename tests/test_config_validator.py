
import unittest
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from adan_trading_bot.common.config_validator import ConfigValidator

class TestConfigValidator(unittest.TestCase):

    def setUp(self):
        """Set up the test case."""
        self.validator = ConfigValidator()
        self.valid_config = {
            'general': {
                'project_name': 'ADAN',
                'random_seed': 42,
                'timezone': 'UTC',
                'debug_mode': True,
                'n_jobs': -1
            },
            'paths': {
                'base_dir': '/path/to/ADAN'
            },
            'data': {
                'data_dir': '/path/to/data'
            },
            'environment': {
                'assets': ['BTC', 'ETH'],
                'observation': {
                    'timeframes': ['5m', '1h']
                }
            },
            'agent': {},
            'training': {},
            'logging': {}
        }

    def test_validate_main_config_valid(self):
        """Test that a valid main configuration passes."""
        result = self.validator.validate_main_config(self.valid_config, 'config.yaml')
        self.assertTrue(result)
        self.assertEqual(len(self.validator.get_validation_errors()), 0)

    def test_validate_main_config_missing_section(self):
        """Test that a missing section in main config is detected."""
        invalid_config = self.valid_config.copy()
        del invalid_config['paths']
        result = self.validator.validate_main_config(invalid_config, 'config.yaml')
        self.assertFalse(result)
        self.assertIn("config.yaml: Missing required section 'paths'", self.validator.get_validation_errors())

if __name__ == '__main__':
    unittest.main()
