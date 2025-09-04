
import unittest
import yaml
import os

class TestConfigLoading(unittest.TestCase):

    def test_workers_config(self):
        config_path = os.path.join(os.path.dirname(__file__), '../../config/config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Vérifier que nous avons 4 workers configurés
        self.assertIn('dbe', config)
        self.assertIn('workers', config['dbe'])
        self.assertEqual(len(config['dbe']['workers']), 4)

        # Vérifier que chaque worker a une configuration complète
        for i in range(1, 5):
            worker_key = f'w{i}'
            worker_config = config['dbe']['workers'][worker_key]
            self.assertIn('regime', worker_config)
            if worker_key != 'w4':
                self.assertIn('bias', worker_config)
            else:
                self.assertIn('adaptive', worker_config)
            self.assertIn('indicators', worker_config)
            self.assertIn('filters', worker_config)

            # Vérifier la configuration PPO
            self.assertIn('ppo', config)
            self.assertIn('workers', config['ppo'])
            ppo_config = config['ppo']['workers'][worker_key]
            self.assertIn('learning_rate', ppo_config)
            self.assertIn('batch_size', ppo_config)
            self.assertIn('ent_coef', ppo_config)

            # Vérifier la configuration trading
            self.assertIn('trading', config)
            self.assertIn('workers', config['trading'])
            trading_config = config['trading']['workers'][worker_key]
            self.assertIn('stop_loss_pct', trading_config)
            self.assertIn('take_profit_pct', trading_config)
            self.assertIn('position_size_pct', trading_config)

if __name__ == '__main__':
    unittest.main()
