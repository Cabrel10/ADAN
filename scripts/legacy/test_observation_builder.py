#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify the observation builder with multi-timeframe data.
This script loads sample data, builds observations, and verifies their structure.
"""

import unittest
import logging
import numpy as np
import pandas as pd
import torch as th
from pathlib import Path
from unittest.mock import Mock

# Add src to PYTHONPATH
SCRIPT_DIR = Path(__file__).resolve().parent
project_root = SCRIPT_DIR.parent.parent # Adjust based on your project structure
sys.path.append(str(project_root / "src"))

from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.common.utils import load_config, get_logger

# Configure logger
logger = get_logger()
logger.setLevel(logging.INFO)

# Configuration
CONFIG_PATH = "config/main_config.yaml" # Assuming main_config.yaml is in project_root/config

class TestObservationBuilderIntegration(unittest.TestCase):
    def setUp(self):
        # Load a sample config for the environment
        self.config = load_config(str(project_root / CONFIG_PATH))
        
        # Mock the data_loader for MultiAssetChunkedEnv
        # In a real scenario, you'd have actual data files
        self.mock_data_loader = Mock()
        self.mock_data_loader.assets_list = ['BTCUSDT', 'ETHUSDT']
        self.mock_data_loader.total_chunks = 1 # Simulate one chunk of data

        # Create dummy data for the mock data loader
        # This data needs to match the expected structure for StateBuilder
        # (timeframes as columns, e.g., '5m_close', '1h_open')
        num_steps = 100 # Number of steps in the dummy chunk
        timeframes = self.config['feature_engineering']['timeframes']
        features_by_timeframe = self.config['feature_engineering']['features']
        
        dummy_data = {}
        for asset in self.mock_data_loader.assets_list:
            asset_data = {}
            for tf in timeframes:
                num_features = len(features_by_timeframe.get(tf, ['open', 'high', 'low', 'close', 'volume']))
                # Create columns with timeframe prefix
                cols = [f"{tf}_{feat}" for feat in features_by_timeframe.get(tf, ['open', 'high', 'low', 'close', 'volume'])]
                df = pd.DataFrame(np.random.rand(num_steps, num_features), columns=cols)
                df.index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_steps, freq='1min'))
                asset_data[tf] = df
            dummy_data[asset] = asset_data

        self.mock_data_loader.load_chunk.return_value = dummy_data
        self.mock_data_loader.get_chunk_optimal_pnl.return_value = {'BTCUSDT': 0.1, 'ETHUSDT': 0.05} # Dummy PnL

        # Initialize the environment
        self.env = MultiAssetChunkedEnv(config=self.config)
        self.env.data_loader = self.mock_data_loader # Inject the mock data loader

    def test_observation_structure_and_content(self):
        # Reset the environment to get initial observation
        obs, info = self.env.reset()

        # Verify observation is a dictionary
        self.assertIsInstance(obs, dict)
        self.assertIn('BTCUSDT', obs)
        self.assertIn('ETHUSDT', obs)

        # Verify observation for each asset is a numpy array
        for asset, asset_obs in obs.items():
            self.assertIsInstance(asset_obs, np.ndarray)
            # Check shape: (channels, window_size, num_features)
            # The actual shape depends on StateBuilder's internal logic and config
            # For now, just check if it's a 3D array
            self.assertEqual(asset_obs.ndim, 3)
            # Check for NaN or Inf values
            self.assertFalse(np.isnan(asset_obs).any(), f"NaNs found in observation for {asset}")
            self.assertFalse(np.isinf(asset_obs).any(), f"Infs found in observation for {asset}")

        # Perform a step and verify next observation
        action = np.array([0, 0]) # Hold for both assets
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        self.assertIsInstance(next_obs, dict)
        self.assertIn('BTCUSDT', next_obs)
        self.assertIn('ETHUSDT', next_obs)

        for asset, asset_obs in next_obs.items():
            self.assertIsInstance(asset_obs, np.ndarray)
            self.assertEqual(asset_obs.ndim, 3)
            self.assertFalse(np.isnan(asset_obs).any(), f"NaNs found in next observation for {asset}")
            self.assertFalse(np.isinf(asset_obs).any(), f"Infs found in next observation for {asset}")

if __name__ == "__main__":
    unittest.main()

