#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for the RiskAssessor class.
"""
import unittest

from adan_trading_bot.risk_management.risk_assessor import (
    RiskAssessor,
    RiskLevel,
)


class TestRiskAssessor(unittest.TestCase):
    """Test cases for RiskAssessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "risk_management": {
                "max_position_size": 0.1,
                "max_portfolio_risk": 0.02,
                "var_confidence": 0.95,
                "var_horizon": 1,
            }
        }
        self.risk_assessor = RiskAssessor(self.config)
        self.portfolio = {
            "total_value": 100000,
            "equity": 100000,
            "used_margin": 0,
            "positions": {
                "BTC": {
                    "size": 1,
                    "entry_price": 50000,
                    "current_price": 50000,
                },
                "ETH": {
                    "size": 10,
                    "entry_price": 3000,
                    "current_price": 3000,
                },
            },
        }

        # Sample market data
        self.market_data = {
            "prices": {"BTC": 50000, "ETH": 3000},
            "volumes": {"BTC": 1000, "ETH": 5000},
            "spreads": {"BTC": 0.0005, "ETH": 0.001},
        }

    def test_initialization(self):
        """Test RiskAssessor initialization."""
        self.assertEqual(self.risk_assessor.max_position_size, 0.1)
        self.assertEqual(self.risk_assessor.max_portfolio_risk, 0.02)
        self.assertEqual(self.risk_assessor.var_confidence, 0.95)

        # Test avec configuration vide
        empty_risk = RiskAssessor({})
        self.assertEqual(
            empty_risk.max_position_size, 0.1
        )  # Valeur par défaut
        self.assertEqual(empty_risk.var_horizon, 1)  # Valeur par défaut

    def test_update_market_data(self):
        """Test updating market data."""
        # Test initial update
        self.risk_assessor.update_market_data(self.market_data)
        self.assertEqual(self.risk_assessor.current_prices["BTC"], 50000)
        self.assertEqual(self.risk_assessor.volumes["ETH"], 5000)

        # Test update with new data
        new_market_data = {
            "prices": {"BTC": 51000, "ETH": 3100, "XRP": 0.5},
            "volumes": {"BTC": 1200, "ETH": 6000},
            "spreads": {"BTC": 12, "ETH": 6},
        }
        self.risk_assessor.update_market_data(new_market_data)
        self.assertEqual(self.risk_assessor.current_prices["BTC"], 51000)
        self.assertEqual(self.risk_assessor.volumes["ETH"], 6000)
        self.assertEqual(self.risk_assessor.spreads["BTC"], 12)

        # Verify historical returns are being tracked
        self.assertGreater(
            len(self.risk_assessor.historical_returns.get("BTC", [])), 0
        )

    def test_calculate_var(self):
        """Test VaR and CVaR calculation."""
        # Test with normal returns
        returns = [0.01, 0.02, -0.01, -0.02, -0.03]
        var, cvar = self.risk_assessor.calculate_var(returns, 0.95)
        self.assertAlmostEqual(var, 0.03, places=2)
        self.assertAlmostEqual(cvar, 0.03, places=3)

        # Test with empty returns
        var, cvar = self.risk_assessor.calculate_var([], 0.95)
        self.assertEqual(var, 0.0)
        self.assertEqual(cvar, 0.0)

        # Test with invalid confidence level
        with self.assertRaises(ValueError):
            self.risk_assessor.calculate_var(returns, 1.1)

        # Test with single return value
        var, cvar = self.risk_assessor.calculate_var([0.01], 0.95)
        self.assertEqual(var, -0.01)
        self.assertEqual(cvar, -0.01)

        # Test with different confidence level
        var, cvar = self.risk_assessor.calculate_var(returns, 0.99)
        self.assertGreaterEqual(var, 0.0)
        self.assertGreaterEqual(cvar, 0.0)

    def test_update_position_risk(self):
        """Test position risk calculation."""
        position = {
            "asset": "BTC",
            "size": 1,
            "entry_price": 49000,
            "stop_loss": 48000,
            "trailing_stop_pct": 0.05,
        }

        # First update
        self.risk_assessor.update_market_data(
            {"prices": {"BTC": 50000}, "volumes": {}, "spreads": {}}
        )
        pos_risk = self.risk_assessor.update_position_risk(position)

        self.assertEqual(pos_risk.asset, "BTC")
        self.assertEqual(pos_risk.current_price, 50000)
        self.assertAlmostEqual(pos_risk.pnl, 1000)  # (50000 - 49000) * 1
        self.assertAlmostEqual(pos_risk.trailing_stop, 47500)  # 50000 * 0.95

        # Test trailing stop update
        position["highest_price"] = 51000
        self.risk_assessor.update_market_data(
            {"prices": {"BTC": 51000}, "volumes": {}, "spreads": {}}
        )
        pos_risk = self.risk_assessor.update_position_risk(position)
        self.assertAlmostEqual(pos_risk.trailing_stop, 48450)  # 51000 * 0.95

    def test_calculate_position_size(self):
        """Test position size calculation."""
        # Set up account equity in the risk assessor
        self.risk_assessor.positions["total_equity"] = 100000

        # Test with valid parameters
        # Default max_position_size is 0.1 (10% of equity)
        # Max position value = 100000 * 10% = 10000
        # With price at 50000, max position size = 10000 / 50000 = 0.2
        size = self.risk_assessor.calculate_position_size(
            "BTC", 50000, 49000, 1.0
        )
        self.assertAlmostEqual(size, 0.2, places=2)

        # Test with invalid stop loss (above current price)
        size = self.risk_assessor.calculate_position_size(
            "BTC", 50000, 51000, 1.0
        )
        self.assertEqual(size, 0.0)

        # Test position size limit with larger max position size
        self.risk_assessor.max_position_size = 0.5  # 50% of equity
        # Risk per share = 50000 - 40000 = 10000
        # Risk amount = 100000 * 1% = 1000
        # Position size based on risk = 1000 / 10000 = 0.1
        # But limited by max position value = 100000 * 50% = 50000
        # Max position size = 50000 / 50000 = 1.0
        # So result should be min(0.1, 1.0) = 0.1
        size = self.risk_assessor.calculate_position_size(
            "BTC", 50000, 40000, 1.0
        )
        self.assertAlmostEqual(size, 0.1, places=2)

        # Test with higher risk per trade (2% instead of 1%)
        # Risk amount = 100000 * 2% = 2000
        # Position size based on risk = 2000 / 1000 = 2.0
        # Limited by max position size = 1.0 (50% of 100k is 50k)
        size = self.risk_assessor.calculate_position_size(
            "BTC", 50000, 48000, 2.0
        )
        self.assertAlmostEqual(size, 1.0, places=2)

    def test_check_stop_loss(self):
        """Test stop loss checking."""
        position = {
            "asset": "BTC",
            "size": 1,
            "entry_price": 50000,
            "stop_loss": 49000,
            "trailing_stop_pct": 0.05,
            "highest_price": 50000,  # Initialize highest price
        }

        # Initial update
        self.risk_assessor.update_market_data(
            {"prices": {"BTC": 50000}, "volumes": {}, "spreads": {}}
        )
        self.risk_assessor.update_position_risk(position)

        # Price above stop loss
        self.assertFalse(self.risk_assessor.check_stop_loss("BTC", 49500))

        # Price hits stop loss
        self.assertTrue(self.risk_assessor.check_stop_loss("BTC", 48500))

        # Test trailing stop
        position["highest_price"] = 52000
        self.risk_assessor.update_market_data({"prices": {"BTC": 52000}})
        self.risk_assessor.update_position_risk(position)

        # Calculate expected trailing stop (52000 * 0.95 = 49400)
        expected_trailing_stop = 52000 * 0.95

        # Price above trailing stop
        self.assertFalse(self.risk_assessor.check_stop_loss("BTC", 50000))

        # Price hits trailing stop
        self.assertTrue(
            self.risk_assessor.check_stop_loss(
                "BTC", expected_trailing_stop - 1
            )
        )

        # Test with non-existent position
        self.assertFalse(
            self.risk_assessor.check_stop_loss("NON_EXISTENT", 100)
        )

        # Test with None price
        self.assertFalse(self.risk_assessor.check_stop_loss("BTC", None))

    def test_assess_portfolio_risk(self):
        """Test portfolio risk assessment."""
        # Set up historical returns
        self.risk_assessor.historical_returns = {
            "BTC": [0.01, 0.02, -0.01, -0.02],
            "ETH": [0.02, -0.01, 0.01, -0.01],
        }

        # Test with empty portfolio
        empty_portfolio = {
            "total_value": 0,
            "equity": 0,
            "used_margin": 0,
            "positions": {},
        }
        empty_risk = self.risk_assessor.assess_portfolio_risk(empty_portfolio)
        self.assertEqual(empty_risk["total_value"], 0)
        self.assertEqual(empty_risk["leverage"], 0.0)

        # Test with normal portfolio
        risk_metrics = self.risk_assessor.assess_portfolio_risk(self.portfolio)

        # Check basic metrics
        self.assertEqual(risk_metrics["total_value"], 100000)
        self.assertEqual(risk_metrics["equity"], 100000)
        self.assertEqual(risk_metrics["leverage"], 1.0)

        # Check risk level
        expected_levels = [
            RiskLevel.LOW,
            RiskLevel.MODERATE,
            RiskLevel.HIGH,
            RiskLevel.EXTREME,
        ]
        self.assertIn(risk_metrics["risk_level"], expected_levels)

        # Test drawdown calculation
        self.portfolio["equity"] = 90000  # 10% drawdown
        risk_metrics = self.risk_assessor.assess_portfolio_risk(self.portfolio)
        self.assertEqual(risk_metrics["drawdown"], 0.1)

        # Test with leverage
        self.portfolio["used_margin"] = 50000
        risk_metrics = self.risk_assessor.assess_portfolio_risk(self.portfolio)
        self.assertGreater(risk_metrics["leverage"], 1.0)

        # Test with None portfolio
        self.assertEqual(self.risk_assessor.assess_portfolio_risk(None), {})

    def test_risk_limits(self):
        """Test risk limits management."""
        # Get current limits
        limits = self.risk_assessor.get_risk_limits()
        self.assertEqual(limits["max_position_size"], 0.1)

        # Update limits
        self.risk_assessor.update_risk_parameters(
            {
                "max_position_size": 0.2,
                "max_portfolio_risk": 0.05,
                "var_confidence": 0.99,
                "var_horizon": 5,
            }
        )

        # Verify updates
        self.assertEqual(self.risk_assessor.max_position_size, 0.2)
        self.assertEqual(self.risk_assessor.max_portfolio_risk, 0.05)
        self.assertEqual(self.risk_assessor.var_confidence, 0.99)
        self.assertEqual(self.risk_assessor.var_horizon, 5)

        # Test with invalid parameters
        with self.assertRaises(ValueError):
            self.risk_assessor.update_risk_parameters(
                {"max_position_size": -0.1}
            )

        with self.assertRaises(ValueError):
            self.risk_assessor.update_risk_parameters(
                {"var_confidence": 1.5}  # Must be between 0 and 1
            )

    def test_position_management(self):
        """Test position management methods."""
        # Test adding a position
        position = {
            "asset": "BTC",
            "size": 1,
            "entry_price": 50000,
            "stop_loss": 49000,
            "trailing_stop_pct": 0.05,
            "highest_price": 50000,  # Initialize highest price
        }

        # Initial update with market data
        self.risk_assessor.update_market_data(
            {
                "prices": {"BTC": 50000},
                "volumes": {"BTC": 1000},
                "spreads": {"BTC": 10},
            }
        )

        # Update position risk
        pos_risk = self.risk_assessor.update_position_risk(position)

        # Test get_position_risk
        pos_risk = self.risk_assessor.get_position_risk("BTC")
        self.assertEqual(pos_risk.asset, "BTC")
        self.assertEqual(pos_risk.size, 1)
        self.assertEqual(pos_risk.entry_price, 50000)
        self.assertEqual(pos_risk.current_price, 50000)
        self.assertEqual(pos_risk.pnl, 0.0)  # No PnL yet

        # Test trailing stop calculation
        self.assertIsNotNone(pos_risk.trailing_stop)

        # Test get_all_positions_risk
        all_positions = self.risk_assessor.get_all_positions_risk()
        self.assertIn("BTC", all_positions)
        self.assertEqual(len(all_positions), 1)

        # Test updating position with higher price
        initial_trailing_stop = pos_risk.trailing_stop
        position["highest_price"] = 52000
        self.risk_assessor.update_market_data({"prices": {"BTC": 52000}})
        updated_risk = self.risk_assessor.update_position_risk(position)
        self.assertGreater(updated_risk.trailing_stop, initial_trailing_stop)

        # Test clear_positions
        self.risk_assessor.clear_positions()
        self.assertEqual(len(self.risk_assessor.positions), 0)

        # Test non-existent position
        non_existent = self.risk_assessor.get_position_risk("NON_EXISTENT")
        self.assertIsNone(non_existent)

        # Test invalid position data
        with self.assertRaises(KeyError):
            self.risk_assessor.update_position_risk(
                {"asset": "BTC"}
            )  # Missing required fields


if __name__ == "__main__":
    unittest.main()
    unittest.main()
