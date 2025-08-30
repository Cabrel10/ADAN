import unittest
import logging
from unittest.mock import Mock
from adan_trading_bot.trading.order_manager import OrderManager, Order, OrderType, OrderSide, OrderStatus
from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

class TestOrderManager(unittest.TestCase):
    def setUp(self):
        self.env_config = {
            'environment': {
                'assets': ['BTCUSDT', 'ETHUSDT'],
                'initial_balance': 1000.0
            },
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
                    {'name': 'Micro Capital', 'min_capital': 0.0, 'max_position_size_pct': 90, 'risk_per_trade_pct': 1.0, 'max_drawdown_pct': 5.0, 'leverage': 1.0},
                    {'name': 'Small Capital', 'min_capital': 1500.0, 'max_position_size_pct': 70, 'risk_per_trade_pct': 1.5, 'max_drawdown_pct': 4.0, 'leverage': 1.0}
                ],
                'position_sizing': {
                    'concentration_limits': {
                        'max_single_asset': 0.5
                    }
                }
            }
        }
        self.mock_portfolio_manager = PortfolioManager(env_config=self.env_config)
        self.order_manager = OrderManager(portfolio_manager=self.mock_portfolio_manager)

    def test_place_order(self):
        order = Order(id='test_order_1', symbol='BTCUSDT', type=OrderType.MARKET, side=OrderSide.BUY, price=20000, quantity=0.01)
        self.assertTrue(self.order_manager.place_order(order))
        self.assertIn('test_order_1', self.order_manager.open_orders)
        self.assertEqual(self.order_manager.open_orders['test_order_1'].status, OrderStatus.NEW)

    def test_cancel_order(self):
        order = Order(id='test_order_2', symbol='BTCUSDT', type=OrderType.MARKET, side=OrderSide.BUY, price=20000, quantity=0.01)
        self.order_manager.place_order(order)
        self.assertTrue(self.order_manager.cancel_order('test_order_2'))
        self.assertNotIn('test_order_2', self.order_manager.open_orders)

    def test_process_order_full_fill(self):
        order = Order(id='test_order_3', symbol='BTCUSDT', type=OrderType.MARKET, side=OrderSide.BUY, price=20000, quantity=0.01)
        self.order_manager.place_order(order)
        self.order_manager.process_order('test_order_3', 0.01, 20000)
        self.assertNotIn('test_order_3', self.order_manager.open_orders)

    def test_process_order_partial_fill(self):
        order = Order(id='test_order_4', symbol='BTCUSDT', type=OrderType.MARKET, side=OrderSide.BUY, price=20000, quantity=0.01)
        self.order_manager.place_order(order)
        self.order_manager.process_order('test_order_4', 0.005, 20000)
        self.assertIn('test_order_4', self.order_manager.open_orders)
        self.assertEqual(self.order_manager.open_orders['test_order_4'].status, OrderStatus.PARTIALLY_FILLED)
        self.assertAlmostEqual(self.order_manager.open_orders['test_order_4'].quantity, 0.005)

    def test_execute_action_buy(self):
        initial_cash = self.mock_portfolio_manager.cash
        self.order_manager.execute_action(1, 20000, 'BTCUSDT') # Action 1 is BUY
        self.assertTrue(self.mock_portfolio_manager.positions['BTCUSDT'].is_open)
        self.assertLess(self.mock_portfolio_manager.cash, initial_cash)

    def test_execute_action_sell(self):
        # First, open a position to be able to sell
        self.order_manager.execute_action(1, 20000, 'BTCUSDT') # BUY
        self.assertTrue(self.mock_portfolio_manager.positions['BTCUSDT'].is_open)
        
        initial_cash = self.mock_portfolio_manager.cash
        realized_pnl = self.order_manager.execute_action(2, 20500, 'BTCUSDT') # Action 2 is SELL
        self.assertFalse(self.mock_portfolio_manager.positions['BTCUSDT'].is_open)
        self.assertGreater(realized_pnl, 0) # Expect a positive PnL for this price increase

    def test_execute_action_hold(self):
        initial_cash = self.mock_portfolio_manager.cash
        self.order_manager.execute_action(0, 20000, 'BTCUSDT') # Action 0 is HOLD
        self.assertEqual(self.mock_portfolio_manager.cash, initial_cash)
        self.assertFalse(self.mock_portfolio_manager.positions['BTCUSDT'].is_open) # No position opened

if __name__ == '__main__':
    unittest.main()
