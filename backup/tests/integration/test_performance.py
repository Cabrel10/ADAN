#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Performance tests for the ADAN Trading Bot.

These tests evaluate the bot's performance over extended periods and under various conditions.
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd
import yaml
import time
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager
from adan_trading_bot.trading.order_manager import OrderManager
from adan_trading_bot.trading.safety_manager import SafetyManager
from adan_trading_bot.trading import OrderSide, OrderStatus

# Load test configuration
CONFIG_PATH = Path(__file__).parent.parent.parent / 'config' / 'environment_config.yaml'

class TestPerformance(unittest.TestCase):
    """Performance tests for the ADAN Trading Bot."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before all tests."""
        # Load configuration
        with open(CONFIG_PATH, 'r') as f:
            cls.config = yaml.safe_load(f)

        # Create a larger sample market data for testing performance
        num_data_points = 1000 # Increased data points for longer runs
        dates = pd.date_range(start='2023-01-01', periods=num_data_points, freq='H')

        # Generate more realistic-looking price data
        np.random.seed(42) # for reproducibility

        btc_prices = 50000 + np.cumsum(np.random.randn(num_data_points) * 50)
        eth_prices = 3000 + np.cumsum(np.random.randn(num_data_points) * 10)

        cls.sample_data = {
            'BTC/USDT': pd.DataFrame({
                'open': btc_prices,
                'high': btc_prices + np.random.rand(num_data_points) * 100,
                'low': btc_prices - np.random.rand(num_data_points) * 100,
                'close': btc_prices + np.random.randn(num_data_points) * 50,
                'volume': 100 + np.random.rand(num_data_points) * 50
            }, index=dates),
            'ETH/USDT': pd.DataFrame({
                'open': eth_prices,
                'high': eth_prices + np.random.rand(num_data_points) * 20,
                'low': eth_prices - np.random.rand(num_data_points) * 20,
                'close': eth_prices + np.random.randn(num_data_points) * 10,
                'volume': 1000 + np.random.rand(num_data_points) * 200
            }, index=dates)
        }

        # Ensure close prices are within high/low range
        for asset_data in cls.sample_data.values():
            asset_data['close'] = np.clip(asset_data['close'], asset_data['low'], asset_data['high'])

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before all tests."""
        # Load configuration
        with open(CONFIG_PATH, 'r') as f:
            cls.config = yaml.safe_load(f)

        # Create a larger sample market data for testing performance
        num_data_points = 1000 # Increased data points for longer runs
        dates = pd.date_range(start='2023-01-01', periods=num_data_points, freq='H')

        # Generate more realistic-looking price data
        np.random.seed(42) # for reproducibility

        btc_prices = 50000 + np.cumsum(np.random.randn(num_data_points) * 50)
        eth_prices = 3000 + np.cumsum(np.random.randn(num_data_points) * 10)

        cls.sample_data = {
            'BTC/USDT': pd.DataFrame({
                'open': btc_prices,
                'high': btc_prices + np.random.rand(num_data_points) * 100,
                'low': btc_prices - np.random.rand(num_data_points) * 100,
                'close': btc_prices + np.random.randn(num_data_points) * 50,
                'volume': 100 + np.random.rand(num_data_points) * 50
            }, index=dates),
            'ETH/USDT': pd.DataFrame({
                'open': eth_prices,
                'high': eth_prices + np.random.rand(num_data_points) * 20,
                'low': eth_prices - np.random.rand(num_data_points) * 20,
                'close': eth_prices + np.random.randn(num_data_points) * 10,
                'volume': 1000 + np.random.rand(num_data_points) * 200
            }, index=dates)
        }

        # Ensure close prices are within high/low range
        for asset_data in cls.sample_data.values():
            asset_data['close'] = np.clip(asset_data['close'], asset_data['low'], asset_data['high'])

        # Create a dummy ChunkedDataLoader that uses the sample_data directly
        class DummyChunkedDataLoader:
            def __init__(self, sample_data, assets_list):
                self.sample_data = sample_data
                self.assets_list = assets_list
                self.chunk_size = 10000 # Dummy chunk size

            def load_chunk(self, chunk_id):
                # For simplicity, return the entire sample_data as a single chunk
                # In a real scenario, this would load a specific chunk from disk
                if chunk_id == 0:
                    return {asset: {'1m': df} for asset, df in self.sample_data.items()} # Wrap in timeframe dict
                return None

            def __len__(self):
                return 1 # Only one chunk for this dummy loader

        cls.dummy_data_loader = DummyChunkedDataLoader(cls.sample_data, list(cls.sample_data.keys()))

        cls.env = MultiAssetChunkedEnv(
            config={
                **cls.config,
                "data": {"assets": list(cls.sample_data.keys())},
                "portfolio": {"initial_capital": cls.config['initial_capital']}
            },
            data_loader_instance=cls.dummy_data_loader # Pass the dummy data loader
        )

    def setUp(self):
        # Reset the environment for each test method
        self.env.reset()

    def test_long_run_performance(self):
        """Test the bot's performance over a long simulation run."""
        print("\n--- Running Long Run Performance Test ---")

        state, info = self.env.reset()
        done = False
        total_steps = 0

        start_time = time.time()

        while not done and total_steps < len(self.sample_data['BTC/USDT']) - 1: # Run through most of the data
            # Simple action: alternate between buying BTC and ETH, then holding
            if total_steps % 3 == 0:
                action = np.array([1.0, 0.0]) # Buy BTC
            elif total_steps % 3 == 1:
                action = np.array([0.0, 1.0]) # Buy ETH
            else:
                action = np.array([0.0, 0.0]) # Hold

            next_state, reward, done, _, info = self.env.step(action)
            state = next_state
            total_steps += 1

            if total_steps % 100 == 0:
                print(f"Step {total_steps}: Portfolio Value = {self.env.portfolio_manager.portfolio_value:.2f}")

        end_time = time.time()
        duration = end_time - start_time

        final_metrics = self.env.portfolio_manager.get_metrics()

        print(f"--- Long Run Performance Test Results ---")
        print(f"Total Steps: {total_steps}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Steps per second: {total_steps / duration:.2f}")
        print(f"Final Portfolio Value: {final_metrics['total_capital']:.2f}")
        print(f"Total PnL (%): {final_metrics['total_pnl_pct']:.2f}%")
        print(f"Max Drawdown: {final_metrics['drawdown']:.2f}")
        print(f"Sharpe Ratio: {final_metrics['sharpe_ratio']:.2f}")

        # Assertions for performance criteria
        self.assertGreater(final_metrics['total_capital'], self.env.portfolio_manager.initial_capital * 0.9) # Should not lose too much
        self.assertGreater(total_steps / duration, 10) # Should process at least 10 steps per second

    def test_speed_and_efficiency(self):
        """Test the speed and efficiency of the environment's step function."""
        print("\n--- Running Speed and Efficiency Test ---")

        state, info = self.env.reset()
        num_steps_to_test = 500

        step_times = []

        for i in range(num_steps_to_test):
            action = np.array([0.0, 0.0]) # Hold action for simplicity

            start_step_time = time.time()
            next_state, reward, done, _, info = self.env.step(action)
            end_step_time = time.time()

            step_times.append(end_step_time - start_step_time)
            state = next_state

            if done:
                break

        avg_step_time = np.mean(step_times)
        total_duration = np.sum(step_times)

        print(f"--- Speed and Efficiency Test Results ---")
        print(f"Total Steps Tested: {len(step_times)}")
        print(f"Total Duration: {total_duration:.4f} seconds")
        print(f"Average Step Time: {avg_step_time:.6f} seconds")
        print(f"Steps per second: {1 / avg_step_time:.2f}")

        # Assertions for speed criteria
        self.assertLess(avg_step_time, 0.1) # Each step should take less than 0.1 seconds
        self.assertGreater(1 / avg_step_time, 10) # Should process at least 10 steps per second

    def test_stress_conditions(self):
        """Test the bot's stability under extreme market conditions."""
        print("\n--- Running Stress Conditions Test ---")

        # Generate stress market data: sudden drop and high volatility
        num_stress_points = 500
        stress_dates = pd.date_date_range(start='2024-01-01', periods=num_stress_points, freq='H')

        # Simulate a sudden price drop
        btc_stress_prices = 60000 * np.ones(num_stress_points)
        btc_stress_prices[50:100] = np.linspace(60000, 30000, 50) # 50% drop
        btc_stress_prices[100:] = 30000 + np.cumsum(np.random.randn(num_stress_points - 100) * 200) # High volatility after drop

        eth_stress_prices = 4000 * np.ones(num_stress_points)
        eth_stress_prices[50:100] = np.linspace(4000, 1500, 50) # ~60% drop
        eth_stress_prices[100:] = 1500 + np.cumsum(np.random.randn(num_stress_points - 100) * 50) # High volatility after drop

        stress_data = {
            'BTC/USDT': pd.DataFrame({
                'open': btc_stress_prices,
                'high': btc_stress_prices + np.random.rand(num_stress_points) * 500,
                'low': btc_stress_prices - np.random.rand(num_stress_points) * 500,
                'close': btc_stress_prices + np.random.randn(num_stress_points) * 200,
                'volume': 200 + np.random.rand(num_stress_points) * 100
            }, index=stress_dates),
            'ETH/USDT': pd.DataFrame({
                'open': eth_stress_prices,
                'high': eth_stress_prices + np.random.rand(num_stress_points) * 100,
                'low': eth_stress_prices - np.random.rand(num_stress_points) * 100,
                'close': eth_stress_prices + np.random.randn(num_stress_points) * 50,
                'volume': 1500 + np.random.rand(num_stress_points) * 300
            }, index=stress_dates)
        }

        # Ensure close prices are within high/low range
        for asset_data in stress_data.values():
            asset_data['close'] = np.clip(asset_data['close'], asset_data['low'], asset_data['high'])

        # Initialize a new environment with stress data
        stress_env = MultiAssetEnv(
            data=stress_data,
            config=self.config,
            mode='paper'
        )

        state, info = stress_env.reset()
        done = False
        total_steps = 0

        start_time = time.time()

        try:
            while not done and total_steps < len(stress_data['BTC/USDT']) - 1:
                action = np.array([0.0, 0.0]) # Hold action for simplicity
                next_state, reward, done, _, info = stress_env.step(action)
                state = next_state
                total_steps += 1

                if stress_env.portfolio_manager.is_bankrupt():
                    print(f"Bot went bankrupt at step {total_steps}!")
                    break

            end_time = time.time()
            duration = end_time - start_time

            final_metrics = stress_env.portfolio_manager.get_metrics()

            print(f"--- Stress Conditions Test Results ---")
            print(f"Total Steps: {total_steps}")
            print(f"Duration: {duration:.2f} seconds")
            print(f"Final Portfolio Value: {final_metrics['total_capital']:.2f}")
            print(f"Total PnL (%): {final_metrics['total_pnl_pct']:.2f}%")
            print(f"Max Drawdown: {final_metrics['drawdown']:.2f}")

            # Assertions for stability under stress
            self.assertFalse(stress_env.portfolio_manager.is_bankrupt(), "Bot should not go bankrupt under stress")
            self.assertGreater(final_metrics['total_capital'], self.config['initial_capital'] * 0.1) # Should retain at least 10% of initial capital
            self.assertFalse(np.isnan(final_metrics['total_capital']), "Final capital should not be NaN")
            self.assertFalse(np.isinf(final_metrics['total_capital']), "Final capital should not be infinite")

        except Exception as e:
            self.fail(f"An unexpected error occurred during stress test: {e}")

if __name__ == '__main__':
    unittest.main()
