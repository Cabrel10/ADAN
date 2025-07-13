#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from unittest.mock import Mock
from src.adan_trading_bot.environment.order_manager import OrderManager

class TestOrderManager(unittest.TestCase):
    def setUp(self):
        self.trading_rules = {'min_order_value': 10.0}
        self.penalties = {
            'invalid_action': -0.1,
            'insufficient_funds': -0.5,
            'max_positions_reached': -0.2,
            'position_not_found': -0.3
        }
        self.order_manager = OrderManager(self.trading_rules, self.penalties)
        
        self.mock_portfolio_manager = Mock()
        self.mock_portfolio_manager.capital = 1000.0
        self.mock_portfolio_manager.positions = {'BTCUSDT': {'units': 0.1, 'avg_price': 10000}}
        self.mock_portfolio_manager.get_current_tier.return_value = {'max_positions': 1}

    def test_valid_buy_order(self):
        order = {'asset': 'ETHUSDT', 'units': 2, 'price': 500}
        is_valid, penalty = self.order_manager.validate_order(order, self.mock_portfolio_manager)
        self.assertTrue(is_valid)
        self.assertEqual(penalty, 0.0)

    def test_invalid_buy_order_insufficient_funds(self):
        order = {'asset': 'ETHUSDT', 'units': 20, 'price': 500}
        is_valid, penalty = self.order_manager.validate_order(order, self.mock_portfolio_manager)
        self.assertFalse(is_valid)
        self.assertEqual(penalty, self.penalties['insufficient_funds'])

    def test_invalid_buy_order_max_positions_reached(self):
        self.mock_portfolio_manager.positions = {'BTCUSDT': {'units': 0.1, 'avg_price': 10000}}
        self.mock_portfolio_manager.get_current_tier.return_value = {'max_positions': 1}
        order = {'asset': 'ETHUSDT', 'units': 1, 'price': 100}
        is_valid, penalty = self.order_manager.validate_order(order, self.mock_portfolio_manager)
        self.assertFalse(is_valid)
        self.assertEqual(penalty, self.penalties['max_positions_reached'])

    def test_valid_sell_order(self):
        order = {'asset': 'BTCUSDT', 'units': -0.05, 'price': 10000}
        is_valid, penalty = self.order_manager.validate_order(order, self.mock_portfolio_manager)
        self.assertTrue(is_valid)
        self.assertEqual(penalty, 0.0)

    def test_invalid_sell_order_not_owned(self):
        order = {'asset': 'LTCUSDT', 'units': -1, 'price': 100}
        is_valid, penalty = self.order_manager.validate_order(order, self.mock_portfolio_manager)
        self.assertFalse(is_valid)
        self.assertEqual(penalty, self.penalties['position_not_found'])

    def test_invalid_order_too_small(self):
        order = {'asset': 'BTCUSDT', 'units': 0.0001, 'price': 10000}
        is_valid, penalty = self.order_manager.validate_order(order, self.mock_portfolio_manager)
        self.assertFalse(is_valid)
        self.assertEqual(penalty, self.penalties['invalid_action'])

if __name__ == '__main__':
    unittest.main()
