#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integration tests for the ADAN Trading Bot.

These tests verify that the main components work together correctly.
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from adan_trading_bot.environment.multi_asset_env import MultiAssetEnv
from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager
from adan_trading_bot.trading.order_manager import OrderManager
from adan_trading_bot.trading.safety_manager import SafetyManager
from adan_trading_bot.trading import OrderSide, OrderStatus

# Load test configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'environment_config.yaml')

class TestIntegration(unittest.TestCase):
    """Integration tests for the ADAN Trading Bot."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before all tests."""
        # Load configuration
        with open(CONFIG_PATH, 'r') as f:
            cls.config = yaml.safe_load(f)
        
        # Create sample market data for testing
        cls.sample_data = {
            'BTC/USDT': pd.DataFrame({
                'open': [50000, 50500, 51000, 50800, 51200],
                'high': [50500, 51000, 51500, 51300, 51500],
                'low': [49800, 50300, 50800, 50600, 51000],
                'close': [50400, 50800, 51200, 51000, 51300],
                'volume': [100, 120, 150, 130, 140]
            }),
            'ETH/USDT': pd.DataFrame({
                'open': [3000, 3050, 3100, 3080, 3120],
                'high': [3050, 3100, 3150, 3130, 3150],
                'low': [2980, 3030, 3080, 3060, 3100],
                'close': [3040, 3080, 3120, 3100, 3130],
                'volume': [1000, 1200, 1500, 1300, 1400]
            })
        }
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.env = MultiAssetEnv(
            data=self.sample_data,
            config=self.config,
            mode='paper'
        )
    
    def test_environment_initialization(self):
        """Test that the environment initializes correctly."""
        self.assertIsNotNone(self.env)
        self.assertEqual(len(self.env.assets), 2)
        self.assertIn('BTC/USDT', self.env.assets)
        self.assertIn('ETH/USDT', self.env.assets)
    
    def test_portfolio_initialization(self):
        """Test that the portfolio manager initializes correctly."""
        portfolio = self.env.portfolio_manager
        self.assertIsNotNone(portfolio)
        self.assertEqual(portfolio.equity, self.config['initial_capital'])
        self.assertEqual(len(portfolio.get_open_positions()), 0)
    
    def test_order_execution(self):
        """Test that orders are executed correctly."""
        # Reset environment
        state, _ = self.env.reset()
        
        # Test buying BTC
        action = np.array([1.0, 0.0])  # Full long on BTC, no action on ETH
        next_state, reward, done, _, info = self.env.step(action)
        
        # Check that a position was opened
        positions = self.env.portfolio_manager.get_open_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0].symbol, 'BTC/USDT')
        self.assertEqual(positions[0].side, OrderSide.BUY)
        
        # Check that safety orders were placed
        self.assertEqual(len(self.env.safety_manager.active_orders), 2)  # Stop-loss and take-profit
    
    def test_risk_management(self):
        """Test that risk management rules are enforced."""
        # Reset environment
        state, _ = self.env.reset()
        
        # Get current price and calculate position size that would exceed max position size
        current_price = self.sample_data['BTC/USDT']['close'].iloc[0]
        max_position_size = self.config['trading']['max_position_size']
        max_allowed = self.env.portfolio_manager.equity * max_position_size
        
        # Try to open a position that's too large
        oversized_position = (max_allowed * 1.5) / current_price  # 50% larger than allowed
        
        # The position should be capped at the maximum allowed size
        action = np.array([1.0, 0.0])  # Full long on BTC
        next_state, reward, done, _, info = self.env.step(action)
        
        positions = self.env.portfolio_manager.get_open_positions()
        self.assertLessEqual(positions[0].size * positions[0].entry_price, max_allowed * 1.01)  # Allow 1% tolerance for rounding
    
    def test_portfolio_metrics(self):
        """Test that portfolio metrics are calculated correctly."""
        # Reset environment
        state, _ = self.env.reset()
        
        # Take some actions
        actions = [
            np.array([1.0, 0.0]),  # Long BTC
            np.array([0.0, 0.0]),  # Hold
            np.array([0.0, 0.0]),  # Hold
            np.array([0.0, -1.0]), # Short ETH
            np.array([0.0, 0.0])   # Hold
        ]
        
        for action in actions:
            state, reward, done, _, info = self.env.step(action)
            if done:
                break
        
        # Check portfolio metrics
        metrics = self.env.portfolio_manager.get_portfolio_metrics()
        self.assertGreaterEqual(metrics['equity'], 0)  # Equity should never be negative
        self.assertLessEqual(metrics['max_drawdown_pct'], 100.0)  # Drawdown can't exceed 100%
        self.assertGreaterEqual(metrics['win_rate_pct'], 0)
        self.assertLessEqual(metrics['win_rate_pct'], 100)
    
    def test_safety_orders(self):
        """Test that safety orders (stop-loss/take-profit) work correctly."""
        # Reset environment
        state, _ = self.env.reset()
        
        # Open a position
        action = np.array([1.0, 0.0])  # Long BTC
        state, _, _, _, _ = self.env.step(action)
        
        # Get the position
        positions = self.env.portfolio_manager.get_open_positions()
        self.assertEqual(len(positions), 1)
        position = positions[0]
        
        # Check that safety orders were placed
        self.assertEqual(len(self.env.safety_manager.active_orders), 2)
        
        # Simulate price moving down to trigger stop-loss
        stop_price = position.entry_price * (1 - self.config['risk_management']['stop_loss']['value'])
        self.env.current_prices['BTC/USDT'] = stop_price * 0.99  # Just below stop price
        
        # Process safety orders
        self.env.safety_manager.check_and_execute_orders(self.env.current_prices)
        
        # Position should be closed
        positions = self.env.portfolio_manager.get_open_positions()
        self.assertEqual(len(positions), 0)
        
        # Check that the position was closed due to stop-loss
        closed_positions = self.env.portfolio_manager.closed_positions
        self.assertEqual(len(closed_positions), 1)
        self.assertEqual(closed_positions[0].tags.get('close_reason'), 'stop_loss')

if __name__ == '__main__':
    unittest.main()
