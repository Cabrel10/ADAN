import unittest
import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
import yaml

from adan_trading_bot.data_processing.data_loader import ComprehensiveDataLoader

class TestComprehensiveDataLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path('./test_processed_data')
        cls.test_dir.mkdir(exist_ok=True)

        cls.assets = ["BTC/USDT", "ETH/USDT"]
        cls.timeframes = ["1m", "1h"]
        cls.chunk_size = 5

        # Create dummy processed data files
        for asset in cls.assets:
            asset_file_name = asset.replace('/', '') # Convert to BTCUSDT format for filename
            for tf in cls.timeframes:
                tf_dir = cls.test_dir / tf
                tf_dir.mkdir(parents=True, exist_ok=True)
                
                num_rows = 20 # Enough rows for multiple chunks
                data = {
                    'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_rows, freq=tf)),
                    'open': np.random.rand(num_rows) * 100,
                    'high': np.random.rand(num_rows) * 100 + 10,
                    'low': np.random.rand(num_rows) * 100 - 10,
                    'close': np.random.rand(num_rows) * 100,
                    'volume': np.random.rand(num_rows) * 1000,
                    'minutes_since_update': np.random.rand(num_rows) * 5
                }
                # Add some dummy indicators to match the new structure
                for i in range(1, 10): # Add 9 dummy indicators
                    data[f'indicator_{i}'] = np.random.rand(num_rows)

                df = pd.DataFrame(data)
                df.set_index('timestamp', inplace=True)
                df.to_parquet(tf_dir / f"{asset_file_name}.parquet")

        # Create a dummy config file
        cls.config_path = cls.test_dir / 'test_data_config.yaml'
        config_content = {
            'data_sources': [
                {'name': 'binance', 'type': 'exchange', 'assets': cls.assets} # Use original format for config
            ],
            'feature_engineering': {
                'timeframes': cls.timeframes,
                'indicators_by_timeframe': {
                    '1m': [f'indicator_{i}' for i in range(1, 10)] + ['open', 'high', 'low', 'close', 'volume', 'minutes_since_update'],
                    '1h': [f'indicator_{i}' for i in range(1, 10)] + ['open', 'high', 'low', 'close', 'volume', 'minutes_since_update']
                }
            },
            'chunk_size': cls.chunk_size
        }
        with open(cls.config_path, 'w') as f:
            yaml.dump(config_content, f)

        # Load the config for the data loader
        with open(cls.config_path, 'r') as f:
            cls.data_config = yaml.safe_load(f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        self.data_loader = ComprehensiveDataLoader(self.data_config, processed_data_dir=str(self.test_dir))

    def test_initialization(self):
        self.assertIsNotNone(self.data_loader)
        self.assertEqual(self.data_loader.chunk_size, self.chunk_size)
        self.assertEqual(self.data_loader.timeframes, self.timeframes)
        self.assertEqual(self.data_loader.assets, self.assets)

    def test_load_asset_paths(self):
        self.data_loader.load_asset_paths()
        self.assertGreater(len(self.data_loader.asset_data_paths), 0)
        self.assertGreater(len(self.data_loader.asset_total_rows), 0)
        for asset in self.assets:
            self.assertIn(asset, self.data_loader.asset_data_paths)
            self.assertIn(asset, self.data_loader.asset_total_rows)
            self.assertGreater(self.data_loader.asset_total_rows[asset], 0)

    def test_get_next_chunk(self):
        self.data_loader.load_asset_paths()
        
        # Test first chunk of first asset
        chunk = self.data_loader.get_next_chunk()
        self.assertIsNotNone(chunk)
        self.assertEqual(len(chunk), self.chunk_size)
        self.assertIn('close_1m', chunk.columns) # Check for merged columns
        self.assertIn('close_1h', chunk.columns)

        # Consume all chunks for the first asset
        total_rows_first_asset = self.data_loader.asset_total_rows[self.assets[0]]
        num_chunks_first_asset = (total_rows_first_asset + self.chunk_size - 1) // self.chunk_size
        
        for _ in range(num_chunks_first_asset - 1):
            chunk = self.data_loader.get_next_chunk()
            self.assertIsNotNone(chunk)
        
        # Next chunk should be for the second asset
        chunk = self.data_loader.get_next_chunk()
        self.assertIsNotNone(chunk)
        self.assertEqual(self.data_loader.current_asset, self.assets[1])

    def test_get_next_observation(self):
        self.data_loader.load_asset_paths()
        
        # Get observations until end of data
        obs_count = 0
        while True:
            observation = self.data_loader.get_next_observation()
            if observation is None:
                break
            obs_count += 1
            self.assertIsInstance(observation, pd.Series)
            # Check that observation contains features from all timeframes
            self.assertIn('close_1m', observation.index)
            self.assertIn('close_1h', observation.index)
        
        # Total observations should be sum of rows for all assets
        expected_obs_count = sum(self.data_loader.asset_total_rows.values())
        self.assertEqual(obs_count, expected_obs_count)

    def test_reset(self):
        self.data_loader.load_asset_paths()
        # Consume some data
        for _ in range(self.chunk_size + 1): # Go into second chunk of first asset
            self.data_loader.get_next_observation()
        
        self.data_loader.reset()
        self.assertEqual(self.data_loader.current_asset_index, 0)
        self.assertIsNone(self.data_loader.current_asset)
        self.assertEqual(self.data_loader.current_chunk_start_index, 0)
        self.assertEqual(self.data_loader.current_chunk_id, 0)
        self.assertIsNone(self.data_loader.current_chunk)
        self.assertEqual(self.data_loader.current_chunk_index, 0)

    def test_get_progress(self):
        self.data_loader.load_asset_paths()
        self.data_loader.get_next_chunk() # Load first chunk
        progress = self.data_loader.get_progress()
        self.assertGreater(progress, 0)
        self.assertLess(progress, 100)

        # Consume all data
        while self.data_loader.get_next_observation() is not None:
            pass
        
        self.assertAlmostEqual(self.data_loader.get_overall_progress(), 100.0)

if __name__ == '__main__':
    unittest.main()