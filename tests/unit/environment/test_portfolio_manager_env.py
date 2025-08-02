#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import pandas as pd
import logging
from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

class TestPortfolioManager(unittest.TestCase):
    def setUp(self):
        self.env_config = {
            'assets': ['BTCUSDT', 'ETHUSDT'],
            'initial_capital': 1000.0,
            'trading_rules': {
                'commission_pct': 0.001,
                'futures_enabled': False,
                'leverage': 1,
                'min_trade_size': 0.0001,
                'min_notional_value': 10.0,
                'max_notional_value': 100000.0,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04,
                'liquidation_threshold': 0.05
            },
            'risk_management': {
                'capital_tiers': [
                    {'threshold': 0, 'max_positions': 1, 'allocation_per_trade': 0.95},
                    {'threshold': 1500, 'max_positions': 2, 'allocation_per_trade': 0.45}
                ],
                'position_sizing': {
                    'concentration_limits': {
                        'max_single_asset': 0.5
                    }
                }
            }
        }
        self.portfolio = PortfolioManager(env_config=self.env_config)

    def test_initial_state(self):
        self.assertEqual(self.portfolio.initial_capital, 1000.0)
        self.assertEqual(self.portfolio.total_capital, 1000.0)
        self.assertEqual(self.portfolio.cash, 1000.0)
        self.assertFalse(self.portfolio.positions['BTCUSDT'].is_open)

    def test_validate_position_valid(self):
        # Test a valid position
        self.assertTrue(self.portfolio.validate_position('BTCUSDT', 0.001, 20000))

    def test_validate_position_invalid_size(self):
        # Test invalid size (too small)
        self.assertFalse(self.portfolio.validate_position('BTCUSDT', 0.00001, 20000))

    def test_validate_position_invalid_notional_value_low(self):
        # Test invalid notional value (too low)
        self.assertFalse(self.portfolio.validate_position('BTCUSDT', 0.0001, 5000))

    def test_validate_position_invalid_notional_value_high(self):
        # Test invalid notional value (too high)
        self.assertFalse(self.portfolio.validate_position('BTCUSDT', 10.0, 20000))

    def test_validate_position_insufficient_cash(self):
        # Test insufficient cash
        self.portfolio.cash = 5.0
        self.assertFalse(self.portfolio.validate_position('BTCUSDT', 0.001, 20000))

    def test_open_and_close_position_spot(self):
        # Open a position
        self.portfolio.open_position('BTCUSDT', 20000, 0.01)
        self.assertTrue(self.portfolio.positions['BTCUSDT'].is_open)
        self.assertAlmostEqual(self.portfolio.cash, 1000 - (20000 * 0.01) - (20000 * 0.01 * 0.001), places=2)

        # Update market price and close position
        self.portfolio.update_market_price({'BTCUSDT': 20500})
        realized_pnl = self.portfolio.close_position('BTCUSDT', 20500)
        self.assertAlmostEqual(realized_pnl, 4.795, places=4)
        self.assertAlmostEqual(self.portfolio.cash, 1000 - (20000 * 0.01) - (20000 * 0.01 * 0.001) + (20500 * 0.01) - (20500 * 0.01 * 0.001), places=2)

    def test_futures_commission(self):
        self.env_config['trading_rules']['futures_enabled'] = True
        self.env_config['trading_rules']['futures_commission_pct'] = 0.0004
        self.env_config['trading_rules']['leverage'] = 10
        self.portfolio = PortfolioManager(env_config=self.env_config)

        # Open a futures position
        self.portfolio.open_position('BTCUSDT', 20000, 0.01) # Notional value 200
        # Cash used = (20000 * 0.01) / 10 + (20000 * 0.01 * 0.0004) = 20 + 0.08 = 20.08
        self.assertAlmostEqual(self.portfolio.cash, 1000 - 20.08, places=3)

        # Close a futures position
        realized_pnl = self.portfolio.close_position('BTCUSDT', 20500)
        # PnL = (20500 - 20000) * 0.01 - commission = 50 - (20500 * 0.01 * 0.0004) = 50 - 0.082 = 49.918
        self.assertAlmostEqual(realized_pnl, 4.918, places=4)
        # Cash after close = 979.92 + (20500 * 0.01) / 10 - (20500 * 0.01 * 0.0004) = 979.92 + 20.5 - 0.082 = 1000.338
        self.assertAlmostEqual(self.portfolio.cash, 1000 - 20.08 + 20.5 - 0.082, places=4)

    def test_rebalance(self):
        self.portfolio.open_position('BTCUSDT', 20000, 0.005) # 100 USDT position
        self.portfolio.update_market_price({'BTCUSDT': 20000})
        # Max single asset limit is 0.5, so 100 USDT is 10% of 1000 USDT, which is fine.
        # Let's make it exceed the limit
        self.portfolio.open_position('ETHUSDT', 2000, 0.2) # 400 USDT position
        self.portfolio.update_market_price({'BTCUSDT': 20000, 'ETHUSDT': 2000})
        # Total portfolio value is now approx 1000 - 100 - 400 = 500 cash + 100 BTC + 400 ETH = 1000
        # BTC allocation: 100/1000 = 0.1 (10%)
        # ETH allocation: 400/1000 = 0.4 (40%)
        # Let's change max_single_asset to 0.3 for ETH to be over
        self.env_config['risk_management']['position_sizing']['concentration_limits']['max_single_asset'] = 0.3
        self.portfolio = PortfolioManager(env_config=self.env_config)
        self.portfolio.cash = 500 # Reset cash for this test
        self.portfolio.positions['BTCUSDT'].open(20000, 0.005)
        self.portfolio.positions['ETHUSDT'].open(2000, 0.2)
        self.portfolio.update_market_price({'BTCUSDT': 20000, 'ETHUSDT': 2000})

        # Rebalance
        self.portfolio.rebalance({'BTCUSDT': 20000, 'ETHUSDT': 2000})
        # ETH should be reduced from 0.2 to 0.15 (0.3 * 500 / 2000)
        # 0.3 * 1000 (total portfolio value) = 300. So ETH should be 300/2000 = 0.15
        self.assertAlmostEqual(self.portfolio.positions['ETHUSDT'].size, 0.15, places=4)

    def test_liquidation(self):
        self.env_config['trading_rules']['futures_enabled'] = True
        self.env_config['trading_rules']['leverage'] = 10
        self.env_config['trading_rules']['liquidation_threshold'] = 0.1 # 10% margin level
        self.portfolio = PortfolioManager(env_config=self.env_config)

        # Open a position that uses a significant portion of capital
        # Initial capital 1000. Notional value 9000. Margin used = 9000 / 10 = 900
        # Equity = 1000 - 900 = 100
        # Margin level = Used Margin / Equity = 900 / 100 = 9
        # This is not how liquidation threshold works. Liquidation threshold is usually a percentage of notional value.
        # Or, if equity drops below a certain percentage of initial margin.
        # Let's assume liquidation_threshold is a percentage of total_capital.
        # If total_capital drops below 10% of initial capital (1000 * 0.1 = 100)

        # Open a position that uses a significant portion of capital
        self.portfolio.open_position('BTCUSDT', 20000, 0.45) # Notional value 9000
        # Simulate a price drop that causes total_capital to fall below 100
        # Current total_capital is 1000. Position 0.45 BTC @ 20000. Notional 9000.
        # If price drops to 17997.78, PnL = (17997.78 - 20000) * 0.45 = -901
        # Total capital = 1000 - 901 = 99

        self.portfolio.update_market_price({'BTCUSDT': 17997.78})
        self.portfolio.check_liquidation({'BTCUSDT': 17997.78})
        self.assertFalse(self.portfolio.positions['BTCUSDT'].is_open) # Position should be closed
        self.assertAlmostEqual(self.portfolio.total_capital, self.portfolio.cash, places=2) # Total capital should be cash after liquidation

if __name__ == '__main__':
    unittest.main()
