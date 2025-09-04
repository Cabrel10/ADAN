import unittest
import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
import yaml

from adan_trading_bot.data_processing.feature_engineer import FeatureEngineer

class TestFeatureEngineer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path('./test_feature_engineer_data')
        cls.test_dir.mkdir(exist_ok=True)
        cls.models_dir = cls.test_dir / 'models'
        cls.models_dir.mkdir(exist_ok=True)

        # Create dummy data
        num_rows = 100
        cls.dummy_df = pd.DataFrame({
            'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_rows, freq='1min')),
            'open_1m': np.random.rand(num_rows) * 100,
            'high_1m': np.random.rand(num_rows) * 100 + 10,
            'low_1m': np.random.rand(num_rows) * 100 - 10,
            'close_1m': np.random.rand(num_rows) * 100,
            'volume_1m': np.random.rand(num_rows) * 1000,
            'open_1h': np.random.rand(num_rows) * 100,
            'high_1h': np.random.rand(num_rows) * 100 + 10,
            'low_1h': np.random.rand(num_rows) * 100 - 10,
            'close_1h': np.random.rand(num_rows) * 100,
            'volume_1h': np.random.rand(num_rows) * 1000,
        })
        cls.dummy_df.set_index('timestamp', inplace=True)

        # Create a dummy config for FeatureEngineer
        cls.feature_engineer_config = {
            'feature_engineering': {
                'timeframes': ["1m", "1h"],
                'indicators': {
                    'rsi': {'length': 14},
                    'macd': {},
                    'bbands': {'length': 20},
                },
                'columns_to_normalize': ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'bbands']
            }
        }

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        self.fe = FeatureEngineer(self.feature_engineer_config, models_dir=str(self.models_dir))

    def test_initialization(self):
        self.assertIsNotNone(self.fe)
        self.assertEqual(self.fe.timeframes, ["1m", "1h"])

    def test_process_data(self):
        processed_df = self.fe.process_data(self.dummy_df.copy(), fit_scaler=True)
        self.assertFalse(processed_df.empty)
        # Check if new indicator columns are added and normalized
        self.assertIn('RSI_14_1m', processed_df.columns)
        self.assertIn('MACD_12_26_9_1m', processed_df.columns) # Default MACD columns
        self.assertIn('BBL_20_2.0_1m', processed_df.columns) # Bollinger Bands Lower

        # Check if data is normalized (mean close to 0, std close to 1)
        for col in ['close_1m', 'RSI_14_1m']:
            if col in processed_df.columns:
                self.assertAlmostEqual(processed_df[col].mean(), 0, delta=0.1)
                self.assertAlmostEqual(processed_df[col].std(), 1, delta=0.1)

    def test_save_load_scaler(self):
        # Process data to fit and save scaler
        self.fe.process_data(self.dummy_df.copy(), fit_scaler=True)
        self.assertTrue(self.fe.scaler_path.exists())

        # Create a new FeatureEngineer instance and load the scaler
        new_fe = FeatureEngineer(self.feature_engineer_config, models_dir=str(self.models_dir))
        new_fe.load_scaler()
        self.assertTrue(new_fe.fitted)
        # Ensure the loaded scaler is the same as the saved one (by comparing attributes)
        self.assertEqual(new_fe.scaler.mean_.tolist(), self.fe.scaler.mean_.tolist())
        self.assertEqual(new_fe.scaler.scale_.tolist(), self.fe.scaler.scale_.tolist())

if __name__ == '__main__':
    unittest.main()
