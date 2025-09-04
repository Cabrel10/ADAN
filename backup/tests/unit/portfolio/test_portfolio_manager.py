import unittest
import yaml
from unittest.mock import MagicMock, patch, call
import numpy as np
import logging
from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager, Position

# Charger la configuration globale pour les tests
with open("config/config.yaml", 'r') as f:
    full_config = yaml.safe_load(f)

class TestPortfolioManager(unittest.TestCase):
    def setUp(self):
        """Set up test parameters using the global config."""
        self.env_config = full_config
        self.portfolio_manager = PortfolioManager(full_config)

        # Set logging level for the portfolio manager to DEBUG for testing
        logging.getLogger('adan_trading_bot.portfolio.portfolio_manager').setLevel(logging.DEBUG)

    def test_initialization(self):
        """Test that the portfolio manager initializes correctly based on config."""
        self.assertEqual(self.portfolio_manager.initial_capital, 20.0)
        self.assertEqual(self.portfolio_manager.get_portfolio_value(), 20.0)
        self.assertIsInstance(self.portfolio_manager.positions, dict)

    def test_get_current_tier(self):
        """Test the logic for determining the correct capital tier."""
        # Test Micro Capital
        self.portfolio_manager.portfolio_value = 15.0
        tier = self.portfolio_manager.get_current_tier()
        self.assertEqual(tier['name'], 'Micro Capital')

        # Test Small Capital (exact lower bound)
        self.portfolio_manager.portfolio_value = 30.0
        tier = self.portfolio_manager.get_current_tier()
        self.assertEqual(tier['name'], 'Small Capital')

        # Test Enterprise Capital
        self.portfolio_manager.portfolio_value = 5000.0
        tier = self.portfolio_manager.get_current_tier()
        self.assertEqual(tier['name'], 'Enterprise')

    def test_calculate_position_size_with_confidence(self):
        """
        Tests that the position size is correctly modulated by the confidence score.
        """
        self.portfolio_manager.portfolio_value = 1000.0  # Force Enterprise Capital Tier

        # --- Test avec une confiance maximale (1.0) ---
        size_full_confidence = self.portfolio_manager.calculate_position_size(
            action_type='buy',
            asset='BTC/USDT',
            current_price=50000,
            confidence=1.0,
            stop_loss_pct=1.0 # 1% stop loss
        )
        # Expected size: 0.004
        self.assertAlmostEqual(size_full_confidence, 0.004, places=4)

        # --- Test avec une confiance modérée (0.5) ---
        size_half_confidence = self.portfolio_manager.calculate_position_size(
            action_type='buy',
            asset='BTC/USDT',
            current_price=50000,
            confidence=0.5,
            stop_loss_pct=1.0 # 1% stop loss
        )
        # Expected size: 0.004 * 0.5 = 0.002
        self.assertAlmostEqual(size_half_confidence, 0.004, places=4)

    def test_check_liquidation(self):
        """Test the liquidation logic based on max_drawdown_pct."""
        self.portfolio_manager.reset()
        initial_equity = self.portfolio_manager.initial_equity # Should be 20.0 from config
        self.portfolio_manager.portfolio_value = initial_equity

        # Simulate drawdown just below the threshold (Small Capital: 4.0%)
        # 3% drawdown: 20.0 - (20.0 * 0.03) = 19.4
        self.portfolio_manager.portfolio_value = initial_equity * (1 - 0.03) # 3% drawdown
        liquidated = self.portfolio_manager.check_liquidation(current_prices={'BTC/USDT': 50000})
        self.assertFalse(liquidated)

        # Simulate drawdown just above the threshold
        # 5.1% drawdown: 20.0 - (20.0 * 0.051) = 18.98
        self.portfolio_manager.portfolio_value = initial_equity * (1 - 0.051) # 5.1% drawdown
        liquidated = self.portfolio_manager.check_liquidation(current_prices={'BTC/USDT': 50000})
        self.assertTrue(liquidated)

if __name__ == '__main__':
    unittest.main()
