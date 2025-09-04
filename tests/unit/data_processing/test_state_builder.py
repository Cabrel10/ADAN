#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for the state_builder module.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from adan_trading_bot.data_processing.state_builder import StateBuilder


class TestStateBuilder(unittest.TestCase):
    """Test cases for the StateBuilder class."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample configuration with actual indicators
        self.features_config = {
            '5m': [
                'RSI_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
                'MACD_HIST', 'ATR_14', 'EMA_5', 'EMA_12',
                'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER'
            ],
            '1h': [
                'RSI_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
                'MACD_HIST', 'ADX_14', 'BB_UPPER', 'BB_MIDDLE',
                'BB_LOWER', 'EMA_26', 'EMA_50'
            ],
            '4h': [
                'RSI_14', 'MACD_HIST', 'ADX_14', 'ATR_14',
                'OBV', 'EMA_50', 'EMA_200', 'SMA_200',
                'STOCHk_14_3_3', 'STOCHd_14_3_3'
            ]
        }

        def create_sample_data(size=1000, base_value=100, spread=0.1):
            """Create sample market data with technical indicators."""
            np.random.seed(42)
            base = np.linspace(base_value, base_value * 2, size)
            noise = np.random.normal(0, spread * base_value, size)
            close = base + noise
            high = close * (1 + np.abs(np.random.normal(0, 0.01, size)))
            low = close * (1 - np.abs(np.random.normal(0, 0.01, size)))
            
            return {
                'OPEN': close * (1 + np.random.normal(0, 0.005, size)),
                'HIGH': high,
                'LOW': low,
                'CLOSE': close,
                'VOLUME': np.random.poisson(1000, size),
                'RSI_14': np.random.uniform(30, 70, size),
                'STOCHk_14_3_3': np.random.uniform(20, 80, size),
                'STOCHd_14_3_3': np.random.uniform(20, 80, size),
                'MACD_HIST': np.random.normal(0, 1, size),
                'ATR_14': np.random.uniform(1, 5, size),
                'ADX_14': np.random.uniform(20, 60, size),
                'BB_UPPER': close * 1.02,
                'BB_MIDDLE': close,
                'BB_LOWER': close * 0.98,
                'EMA_5': close * (1 + np.random.normal(0, 0.001, size)),
                'EMA_12': close * (1 + np.random.normal(0, 0.001, size)),
                'EMA_26': close * (1 + np.random.normal(0, 0.001, size)),
                'EMA_50': close * (1 + np.random.normal(0, 0.0005, size)),
                'EMA_200': close * (1 + np.random.normal(0, 0.0002, size)),
                'SMA_200': close * (1 + np.random.normal(0, 0.0002, size)),
                'OBV': np.cumsum(np.random.normal(0, 1000, size))
            }

        # Sample market data with all indicators
        self.sample_data = {
            '5m': {
                'BTC/USDT': pd.DataFrame(create_sample_data(1000, 50000)),
                'ETH/USDT': pd.DataFrame(create_sample_data(1000, 3000))
            },
            '1h': {
                'BTC/USDT': pd.DataFrame(create_sample_data(1000, 50000)),
                'ETH/USDT': pd.DataFrame(create_sample_data(1000, 3000))
            },
            '4h': {
                'BTC/USDT': pd.DataFrame(create_sample_data(1000, 50000)),
                'ETH/USDT': pd.DataFrame(create_sample_data(1000, 3000))
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
        # Vérification des timeframes attendus
        expected_timeframes = sorted(['5m', '1h', '4h'])
        self.assertEqual(sorted(self.state_builder.timeframes), expected_timeframes)

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
        # Expected shape: (num_timeframes, window_size, num_features)
        expected_shape = (3, 100, 10)  # 3 timeframes, 100 window, 10 features
        self.assertEqual(obs_array.shape, expected_shape)

    def test_validation_checks(self):
        """Test data validation in state building."""
        # Create invalid data with NaN values in indicators
        invalid_data = self.sample_data.copy()
        timeframes = ['15m', '1h', '4h']  # Mise à jour des timeframes
        assets = ['BTC/USDT', 'ETH/USDT']
        
        for tf in timeframes:
            for asset in assets:
                if tf in invalid_data and asset in invalid_data[tf]:
                    # Set some indicator values to NaN
                    invalid_data[tf][asset].loc[100:110, 'RSI_14'] = np.nan
                    invalid_data[tf][asset].loc[200:210, 'MACD_HIST'] = np.nan

        # Test avec validation - gère les NaN en les remplaçant par des zéros
        try:
            observation = self.state_builder.build_observation(300, invalid_data)
            # Vérifier que l'observation est valide malgré les NaN
            self.assertIn('observation', observation)
            self.assertFalse(np.isnan(observation['observation']).any())
        except Exception as e:
            msg = f"La construction de l'observation a échoué: {str(e)}"
            self.fail(msg)

    def test_window_adaptation(self):
        """Test adaptive window sizing based on market conditions."""
        # Test avec haute volatilité
        volatile_data = self.sample_data.copy()
        volatile_data['5m']['BTC/USDT'].loc[100:200, 'CLOSE'] *= 1.1  # 10% increase

        # Build observation with adaptive window
        observation = self.state_builder.build_observation(200, volatile_data)
        obs_array = observation['observation']

        # Vérification de la forme de l'observation
        expected_shape = (3, 100, 10)  # 3 timeframes, 100 fenêtre, 10 features
        self.assertEqual(obs_array.shape, expected_shape)

    def test_portfolio_state_inclusion(self):
        """Test inclusion of portfolio state in observations."""
        # Test with portfolio state included
        observation = self.state_builder.build_observation(200, self.sample_data)
        self.assertIn('portfolio_state', observation)
        self.assertIsInstance(observation['portfolio_state'], np.ndarray)

        # Test sans état du portefeuille
        sb_no_portfolio = StateBuilder(
            features_config=self.features_config,
            window_size=100,
            include_portfolio_state=False,
            normalize=True
        )
        observation = sb_no_portfolio.build_observation(200, self.sample_data)
        # Le portefeuille est toujours présent mais peut être vide
        self.assertIn('portfolio_state', observation)
        self.assertTrue(np.all(observation['portfolio_state'] == 0))


if __name__ == '__main__':
    unittest.main()
