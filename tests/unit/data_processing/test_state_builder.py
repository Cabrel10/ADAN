#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for the state_builder module.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add the project root to the Python path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from adan_trading_bot.data_processing.state_builder import StateBuilder


class TestStateBuilder(unittest.TestCase):
    """Test cases for the StateBuilder class."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample configuration
        self.features_config = {
            '5m': ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'],
            '1h': ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
        }
        
        # Sample market data
        self.sample_data = {
            '5m': {
                'BTC/USDT': pd.DataFrame({
                    'OPEN': np.linspace(100, 200, 1000),
                    'HIGH': np.linspace(105, 205, 1000),
                    'LOW': np.linspace(95, 195, 1000),
                    'CLOSE': np.linspace(102, 202, 1000),
                    'VOLUME': np.random.poisson(1000, 1000)
                }),
                'ETH/USDT': pd.DataFrame({
                    'OPEN': np.linspace(50, 150, 1000),
                    'HIGH': np.linspace(55, 155, 1000),
                    'LOW': np.linspace(45, 145, 1000),
                    'CLOSE': np.linspace(52, 152, 1000),
                    'VOLUME': np.random.poisson(800, 1000)
                })
            },
            '1h': {
                'BTC/USDT': pd.DataFrame({
                    'OPEN': np.linspace(100, 200, 1000),
                    'HIGH': np.linspace(105, 205, 1000),
                    'LOW': np.linspace(95, 195, 1000),
                    'CLOSE': np.linspace(102, 202, 1000),
                    'VOLUME': np.random.poisson(5000, 1000)
                })
            }
        }
        
        # Initialize StateBuilder
        self.state_builder = StateBuilder(
            features_config=self.features_config,
            window_size=100,
            include_portfolio_state=True,
            normalize=True
        )

    def test_initialization(self):
        """Test StateBuilder initialization."""
        self.assertEqual(self.state_builder.window_size, 100)
        self.assertTrue(self.state_builder.include_portfolio_state)
        self.assertTrue(self.state_builder.normalize)
        self.assertEqual(self.state_builder.timeframes, ['5m', '1h'])

    def test_build_observation(self):
        """Test building an observation from market data."""
        # Get a sample index
        current_idx = 200
        
        # Build observation
        observation = self.state_builder.build_observation(current_idx, self.sample_data)
        
        # Verify observation structure
        self.assertIsInstance(observation, dict)
        self.assertIn('observation', observation)
        self.assertIn('portfolio_state', observation)
        
        # Verify observation shape
        obs_array = observation['observation']
        self.assertEqual(obs_array.shape, (2, 100, 5))  # 2 timeframes, 100 window, 5 features

    def test_validation_checks(self):
        """Test data validation in state building."""
        # Create invalid data with NaN values
        invalid_data = self.sample_data.copy()
        invalid_data['5m']['BTC/USDT'].iloc[100:110, 0] = np.nan
        
        # Test with validation
        with self.assertRaises(ValueError):
            self.state_builder.build_observation(200, invalid_data)

    def test_window_adaptation(self):
        """Test adaptive window sizing based on market conditions."""
        # Test with high volatility
        volatile_data = self.sample_data.copy()
        volatile_data['5m']['BTC/USDT'].loc[100:200, 'CLOSE'] *= 1.1  # 10% increase
        
        # Build observation with adaptive window
        observation = self.state_builder.build_observation(200, volatile_data)
        obs_array = observation['observation']
        
        # Verify observation shape
        self.assertEqual(obs_array.shape, (2, 100, 5))

    def test_portfolio_state_inclusion(self):
        """Test inclusion of portfolio state in observations."""
        # Test with portfolio state included
        observation = self.state_builder.build_observation(200, self.sample_data)
        self.assertIn('portfolio_state', observation)
        self.assertIsInstance(observation['portfolio_state'], np.ndarray)
        
        # Test without portfolio state
        sb_no_portfolio = StateBuilder(
            features_config=self.features_config,
            window_size=100,
            include_portfolio_state=False,
            normalize=True
        )
        observation = sb_no_portfolio.build_observation(200, self.sample_data)
        self.assertNotIn('portfolio_state', observation)


if __name__ == '__main__':
    unittest.main()
