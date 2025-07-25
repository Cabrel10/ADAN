#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import pandas as pd
from adan_trading_bot.environment.data_handler import DataHandler

class TestDataHandler(unittest.TestCase):
    def setUp(self):
        data = {
            'BTCUSDT_close_1m': [10000, 10100, 10200],
            'ETHUSDT_close_1m': [500, 505, 510]
        }
        self.df = pd.DataFrame(data)
        self.handler = DataHandler(self.df)

    def test_get_current_observation(self):
        obs = self.handler.get_current_observation()
        self.assertEqual(obs['BTCUSDT_close_1m'], 10000)

    def test_get_current_prices(self):
        prices = self.handler.get_current_prices(['BTCUSDT', 'ETHUSDT'])
        self.assertEqual(prices['BTCUSDT'], 10000)
        self.assertEqual(prices['ETHUSDT'], 500)

    def test_step(self):
        self.handler.step()
        obs = self.handler.get_current_observation()
        self.assertEqual(obs['BTCUSDT_close_1m'], 10100)
        self.assertTrue(self.handler.step())
        self.assertFalse(self.handler.step())

if __name__ == '__main__':
    unittest.main()
