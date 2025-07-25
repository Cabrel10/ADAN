
import unittest
import yaml
import os

class TestConfigLoading(unittest.TestCase):

    def test_stop_loss_exists_in_config(self):
        config_path = os.path.join(os.path.dirname(__file__), '../../config/config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.assertIn('environment', config)
        self.assertIn('trading_rules', config['environment'])
        self.assertIn('stop_loss', config['environment']['trading_rules'])

if __name__ == '__main__':
    unittest.main()
