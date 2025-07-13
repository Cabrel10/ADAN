#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import pandas as pd
from src.adan_trading_bot.environment.portfolio_manager import PortfolioManager

class TestPortfolioManager(unittest.TestCase):
    def setUp(self):
        self.assets = ['BTCUSDT', 'ETHUSDT']
        self.capital_tiers = [
            {'threshold': 0, 'max_positions': 1, 'allocation_per_trade': 0.95, 'reward_multiplier': 1.0},
            {'threshold': 1500, 'max_positions': 2, 'allocation_per_trade': 0.45, 'reward_multiplier': 1.1}
        ]
        self.portfolio = PortfolioManager(1000.0, self.assets, 0.001, self.capital_tiers)

    def test_initial_state(self):
        self.assertEqual(self.portfolio.capital, 1000.0)
        self.assertEqual(self.portfolio.portfolio_value, 1000.0)
        self.assertEqual(self.portfolio.positions['BTCUSDT']['units'], 0)

    def test_buy_trade(self):
        self.portfolio.execute_trade('BTCUSDT', 0.1, 10000)
        self.assertLess(self.portfolio.capital, 0)
        self.assertEqual(self.portfolio.positions['BTCUSDT']['units'], 0.1)
        self.assertEqual(self.portfolio.positions['BTCUSDT']['avg_price'], 10000)

    def test_sell_trade(self):
        self.portfolio.execute_trade('BTCUSDT', 0.1, 10000)
        self.portfolio.execute_trade('BTCUSDT', -0.05, 11000)
        self.assertAlmostEqual(self.portfolio.capital, 548.45, places=2)
        self.assertAlmostEqual(self.portfolio.positions['BTCUSDT']['units'], 0.05)

    def test_get_tier(self):
        self.assertEqual(self.portfolio.get_current_tier()['threshold'], 0)
        self.portfolio.capital = 2000
        self.portfolio.update_portfolio_value(None)
        self.assertEqual(self.portfolio.get_current_tier()['threshold'], 1500)

if __name__ == '__main__':
    unittest.main()
