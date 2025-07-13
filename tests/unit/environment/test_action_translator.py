#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from src.adan_trading_bot.environment.action_translator import ActionTranslator
from src.adan_trading_bot.environment.portfolio_manager import PortfolioManager

class TestActionTranslator(unittest.TestCase):
    def setUp(self):
        self.assets = ['BTCUSDT', 'ETHUSDT']
        self.translator = ActionTranslator(self.assets)
        self.capital_tiers = [
            {'threshold': 0, 'max_positions': 1, 'allocation_per_trade': 0.95, 'reward_multiplier': 1.0}
        ]
        self.portfolio = PortfolioManager(1000.0, self.assets, 0.001, self.capital_tiers)
        self.current_prices = {'BTCUSDT': 10000, 'ETHUSDT': 500}

    def test_buy_action(self):
        action = np.array([0.6, 0.1])
        orders = self.translator.translate_action(action, self.portfolio, self.current_prices)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]['asset'], 'BTCUSDT')
        self.assertGreater(orders[0]['units'], 0)

    def test_sell_action(self):
        self.portfolio.execute_trade('BTCUSDT', 0.1, 10000)
        action = np.array([-0.7, 0.2])
        orders = self.translator.translate_action(action, self.portfolio, self.current_prices)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]['asset'], 'BTCUSDT')
        self.assertLess(orders[0]['units'], 0)

    def test_hold_action(self):
        action = np.array([0.1, -0.2])
        orders = self.translator.translate_action(action, self.portfolio, self.current_prices)
        self.assertEqual(len(orders), 0)

if __name__ == '__main__':
    unittest.main()
