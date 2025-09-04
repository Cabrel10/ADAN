#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Performance tests for data loading and state building.
"""

import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import psutil

# Add the project root to the Python path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from src.adan_trading_bot.data_processing.state_builder import StateBuilder

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDataLoadingPerformance(unittest.TestCase):
    """Performance tests for data loading and state building."""

    @classmethod
    def setUpClass(cls):
        """Set up test data for all tests."""
        # Sample configuration
        cls.config = {
            'data': {
                'base_path': '/fake/data/path',
                'timeframes': ['5m', '1h'],
                'default_assets': ['BTC/USDT']
            },
            'environment': {
                'max_episode_steps': 1000
            }
        }

        # Worker config
        cls.worker_config = {
            'timeframes': ['5m', '1h'],
            'data_split': 'train',
            'assets': ['BTC/USDT'],
            'chunk_size': 1000
        }

        # Generate large sample data
        cls.large_sample_size = 100000  # 100k samples
        cls.sample_data = {
            '5m': {
                'BTC/USDT': pd.DataFrame({
                    'OPEN': np.linspace(100, 200, cls.large_sample_size),
                    'HIGH': np.linspace(105, 205, cls.large_sample_size),
                    'LOW': np.linspace(95, 195, cls.large_sample_size),
                    'CLOSE': np.linspace(102, 202, cls.large_sample_size),
                    'VOLUME': np.random.poisson(1000, cls.large_sample_size)
                })
            },
            '1h': {
                'BTC/USDT': pd.DataFrame({
                    'OPEN': np.linspace(100, 200, cls.large_sample_size // 12),
                    'HIGH': np.linspace(105, 205, cls.large_sample_size // 12),
                    'LOW': np.linspace(95, 195, cls.large_sample_size // 12),
                    'CLOSE': np.linspace(102, 202, cls.large_sample_size // 12),
                    'VOLUME': np.random.poisson(5000, cls.large_sample_size // 12)
                })
            }
        }

    def get_memory_usage_mb(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    @patch('pandas.read_parquet')
    @patch('pathlib.Path.glob')
    def test_data_loading_speed(self, mock_glob, mock_read_parquet):
        """Test the speed of data loading with different chunk sizes."""
        # Setup mocks
        mock_glob.return_value = [Path('BTC_USDT_5m.parquet'), Path('BTC_USDT_1h.parquet')]
        mock_read_parquet.side_effect = [
            self.sample_data['5m']['BTC/USDT'],
            self.sample_data['1h']['BTC/USDT']
        ]

        # Test different chunk sizes
        chunk_sizes = [100, 1000, 5000, 10000]
        results = {}

        for chunk_size in chunk_sizes:
            self.worker_config['chunk_size'] = chunk_size

            # Measure memory before
            start_mem = self.get_memory_usage_mb()
            start_time = time.time()

            # Initialize and load data
            loader = ChunkedDataLoader(self.config, self.worker_config)
            chunks = list(loader.load_chunk())

            # Measure performance
            elapsed = time.time() - start_time
            end_mem = self.get_memory_usage_mb()
            mem_used = end_mem - start_mem

            # Store results
            results[chunk_size] = {
                'time': elapsed,
                'memory_mb': mem_used,
                'num_chunks': len(chunks)
            }

            logger.info(f"Chunk size: {chunk_size}, "
                      f"Time: {elapsed:.2f}s, "
                      f"Memory: {mem_used:.2f}MB, "
                      f"Chunks: {len(chunks)}")

        # Log performance comparison
        logger.info("\nPerformance comparison:")
        for chunk_size, metrics in results.items():
            logger.info(f"Chunk {chunk_size}: {metrics['time']:.2f}s, {metrics['memory_mb']:.2f}MB")

    def test_state_building_performance(self):
        """Test the performance of state building."""
        # Initialize StateBuilder
        features_config = {
            '5m': ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'],
            '1h': ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
        }

        # Test with different window sizes
        window_sizes = [50, 100, 200]

        for window_size in window_sizes:
            state_builder = StateBuilder(
                features_config=features_config,
                window_size=window_size,
                include_portfolio_state=True,
                normalize=True
            )

            # Test indices to evaluate
            test_indices = [1000, 5000, 10000]

            logger.info(f"\nTesting window size: {window_size}")

            for idx in test_indices:
                # Measure memory before
                start_mem = self.get_memory_usage_mb()
                start_time = time.time()

                # Build observation
                observation = state_builder.build_observation(idx, self.sample_data)

                # Measure performance
                elapsed = (time.time() - start_time) * 1000  # in ms
                end_mem = self.get_memory_usage_mb()
                mem_used = end_mem - start_mem

                # Log results
                logger.info(f"  Index {idx}: {elapsed:.2f}ms, Memory: {mem_used:.2f}MB")

                # Verify observation shape
                obs_shape = observation['observation'].shape
                self.assertEqual(obs_shape, (2, window_size, 5))  # 2 timeframes, window_size, 5 features


if __name__ == '__main__':
    unittest.main()
