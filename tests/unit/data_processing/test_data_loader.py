#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for the data_loader module.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os
import sys
import psutil

# Add the project root to the Python path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader


class TestChunkedDataLoader(unittest.TestCase):
    """Test cases for the ChunkedDataLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a sample config with features configuration
        self.config = {
            'data': {
                'base_path': '/fake/data/path',
                'timeframes': ['5m', '1h'],
                'default_assets': ['BTC/USDT', 'ETH/USDT'],
                'features_per_timeframe': {
                    '5m': ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'],
                    '1h': ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
                },
                'data_split': 'train',
                'chunk_sizes': {'5m': 100, '1h': 100}
            },
            'environment': {
                'max_episode_steps': 1000,
                'assets': ['BTC/USDT']
            }
        }

        # Worker config - prioritaire sur la config principale
        self.worker_config = {
            'timeframes': ['5m', '1h'],
            'data_split_override': 'train',
            'assets': ['BTC/USDT'],
            'chunk_sizes': {'5m': 100, '1h': 100}
        }

        # Sample market data
        self.sample_data = pd.DataFrame({
            'OPEN': np.random.rand(1000) * 100 + 1000,
            'HIGH': np.random.rand(1000) * 105 + 1000,
            'LOW': np.random.rand(1000) * 95 + 1000,
            'CLOSE': np.random.rand(1000) * 100 + 1000,
            'VOLUME': np.random.poisson(1000, 1000),
            'timestamp': pd.date_range(start='2023-01-01', periods=1000, freq='5min')
        })

    @patch('pandas.read_parquet')
    @patch('pathlib.Path.glob')
    def test_init_with_defaults(self, mock_glob, mock_read_parquet):
        """Test initialization with default parameters."""
        # Setup mocks
        mock_glob.return_value = [Path('BTC_USDT_5m.parquet')]
        mock_read_parquet.return_value = self.sample_data

        # Initialize with default config
        loader = ChunkedDataLoader(self.config, self.worker_config)

        # Verify attributes
        self.assertEqual(loader.timeframes, self.worker_config['timeframes'])
        self.assertEqual(
            loader.data_split,
            self.worker_config['data_split_override']
        )
        self.assertEqual(loader.assets_list, self.worker_config['assets'])
        self.assertEqual(loader.chunk_sizes['5m'], 100)
        self.assertEqual(loader.chunk_sizes['1h'], 100)

    @patch('pandas.read_parquet')
    @patch('pathlib.Path.glob')
    def test_load_chunk(self, mock_glob, mock_read_parquet):
        """Test loading a chunk of data."""
        # Setup mocks
        mock_glob.return_value = [Path('BTC_USDT_5m.parquet')]
        mock_read_parquet.return_value = self.sample_data

        # Initialize loader
        loader = ChunkedDataLoader(self.config, self.worker_config)

        # Mock the _load_asset_timeframe_parallel method to return sample data
        def mock_load_side_effect(asset, tf, max_retries=3):
            return (asset, tf, self.sample_data)

        with patch.object(loader, '_load_asset_timeframe_parallel',
                        side_effect=mock_load_side_effect):
            # Load chunk
            chunk = loader.load_chunk()

            # Verify chunk properties
            self.assertIsInstance(chunk, dict)
            self.assertIn('BTC/USDT', chunk)  # Check asset is a key
            self.assertIn('5m', chunk['BTC/USDT'])  # Check timeframe exists for asset
            self.assertIn('1h', chunk['BTC/USDT'])  # Check both timeframes are present

            # Verify data was loaded correctly
            self.assertEqual(
                len(chunk['BTC/USDT']['5m']),
                100  # Vérifie que la taille du chunk est de 100 lignes comme défini dans la configuration
            )
            self.assertEqual(
                len(chunk['BTC/USDT']['1h']),
                100  # Vérifie que la taille du chunk est de 100 lignes comme défini dans la configuration
            )

    @patch('pathlib.Path.glob')
    def test_lazy_loading(self, mock_glob):
        """Test lazy loading of data."""
        """Test lazy loading of data chunks."""
        # Configure mock to return sample data
        mock_glob.return_value = ['/fake/path/BTC_USDT_5m.parquet']

        # Initialize loader with lazy loading
        loader = ChunkedDataLoader(self.config, self.worker_config)

        # Mock the _load_asset_timeframe_parallel method to return chunks
        chunk1 = pd.DataFrame({'OPEN': [1, 2], 'HIGH': [1.1, 2.2], 'LOW': [0.9, 1.8],
                              'CLOSE': [1.05, 2.1], 'VOLUME': [100, 200],
                              'timestamp': pd.date_range('2023-01-01', periods=2, freq='5min')})
        chunk2 = pd.DataFrame({'OPEN': [3, 4], 'HIGH': [3.1, 4.2], 'LOW': [2.9, 3.8],
                              'CLOSE': [3.05, 4.1], 'VOLUME': [300, 400],
                              'timestamp': pd.date_range('2023-01-01 00:10', periods=2, freq='5min')})

        def mock_load_side_effect(asset, tf, max_retries=3):
            if not hasattr(mock_load_side_effect, 'count'):
                mock_load_side_effect.count = 0
            mock_load_side_effect.count += 1
            if mock_load_side_effect.count == 1:
                return ('BTC/USDT', '5m', chunk1)
            else:
                return ('BTC/USDT', '5m', chunk2)

        with patch.object(loader, '_load_asset_timeframe_parallel',
                         side_effect=mock_load_side_effect):
            # Load first chunk
            result1 = loader.load_chunk()
            self.assertIn('BTC/USDT', result1)
            self.assertIn('5m', result1['BTC/USDT'])

            # Load second chunk
            result2 = loader.load_chunk()
            self.assertIn('BTC/USDT', result2)
            self.assertIn('5m', result2['BTC/USDT'])

            # Verify different chunks were loaded
            self.assertNotEqual(id(result1['BTC/USDT']['5m']),
                              id(result2['BTC/USDT']['5m']))

    @patch('pandas.read_parquet')
    @patch('pathlib.Path.glob')
    def test_memory_efficiency(self, mock_glob, mock_read_parquet):
        # Configure mock to return sample data
        mock_glob.return_value = ['/fake/path/BTC_USDT_5m.parquet']
        mock_read_parquet.return_value = self.sample_data

        # Initialize loader with small chunk size
        config = self.config.copy()
        config['data']['chunk_sizes'] = {'5m': 5}  # Small chunk size for testing
        loader = ChunkedDataLoader(config, self.worker_config)

        # Create test chunks
        chunks = [
            pd.DataFrame({
                'OPEN': np.random.rand(5) * 1000 + 1000,
                'HIGH': np.random.rand(5) * 1000 + 1000,
                'LOW': np.random.rand(5) * 1000 + 1000,
                'CLOSE': np.random.rand(5) * 1000 + 1000,
                'VOLUME': np.random.randint(100, 1000, 5),
                'timestamp': pd.date_range('2023-01-01', periods=5, freq='5min')
            }) for _ in range(3)  # 3 chunks of 5 rows each
        ]

        # Track which chunk to return next
        chunk_index = 0

        def mock_load_side_effect(asset, tf, max_retries=3):
            nonlocal chunk_index
            chunk = chunks[chunk_index % len(chunks)]
            chunk_index += 1
            return (asset, tf, chunk)

        with patch.object(loader, '_load_asset_timeframe_parallel',
                         side_effect=mock_load_side_effect):
            # Load data in chunks and track memory
            chunks_loaded = 0
            memory_usage = []

            # Load all chunks
            for _ in range(3):  # We know we have 3 chunks
                chunk = loader.load_chunk()
                chunks_loaded += 1
                # Verify chunk size is as expected
                self.assertIn('BTC/USDT', chunk)
                self.assertIn('5m', chunk['BTC/USDT'])
                self.assertLessEqual(len(chunk['BTC/USDT']['5m']), 5)  # Verify chunk size

                # Track memory usage
                process = psutil.Process()
                memory_usage.append(process.memory_info().rss / 1024 / 1024)  # in MB

            # Verify we loaded all chunks
            self.assertEqual(chunks_loaded, 3)

            # Verify memory usage didn't grow too much
            # Allow for some variation in memory usage
            self.assertLess(max(memory_usage) - min(memory_usage), 10.0)  # Less than 10MB variation


if __name__ == '__main__':
    unittest.main()
