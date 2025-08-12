
import unittest
from unittest.mock import Mock, patch
from src.adan_trading_bot.trading.manual_trading_interface import ManualTradingInterface, OrderSide, OrderStatus
from src.adan_trading_bot.trading.secure_api_manager import ExchangeType

class TestTradingInterface(unittest.TestCase):

    def setUp(self):
        self.mock_api_manager = Mock()
        self.interface = ManualTradingInterface(self.mock_api_manager)

        # Mock exchange info
        self.exchange_info = {
            'symbols': [{
                'symbol': 'BTCUSDT',
                'filters': [
                    {'filterType': 'PRICE_FILTER', 'minPrice': '0.01', 'maxPrice': '1000000', 'tickSize': '0.01'},
                    {'filterType': 'LOT_SIZE', 'minQty': '0.00001', 'maxQty': '100', 'stepSize': '0.00001'},
                    {'filterType': 'MIN_NOTIONAL', 'minNotional': '10.0'}
                ]
            }]
        }
        self.mock_api_manager.get_exchange_info.return_value = self.exchange_info

    def test_create_limit_order_validation_pass(self):
        order_id = self.interface.create_limit_order("BTCUSDT", OrderSide.BUY, 0.1, 50000)
        self.assertIn(order_id, self.interface.orders)
        self.assertEqual(self.interface.orders[order_id].status, OrderStatus.PENDING)

    def test_create_limit_order_validation_fail(self):
        # Test invalid price
        order_id_price = self.interface.create_limit_order("BTCUSDT", OrderSide.BUY, 0.1, 0.001)
        self.assertEqual(self.interface.orders[order_id_price].status, OrderStatus.REJECTED)

        # Test invalid quantity
        order_id_qty = self.interface.create_limit_order("BTCUSDT", OrderSide.BUY, 0.000001, 50000)
        self.assertEqual(self.interface.orders[order_id_qty].status, OrderStatus.REJECTED)

        # Test notional
        order_id_notional = self.interface.create_limit_order("BTCUSDT", OrderSide.BUY, 0.0001, 50000)
        self.assertEqual(self.interface.orders[order_id_notional].status, OrderStatus.REJECTED)

    def test_submit_order(self):
        self.mock_api_manager.send_order.return_value = {"orderId": 123, "status": "NEW"}
        order_id = self.interface.create_limit_order("BTCUSDT", OrderSide.BUY, 0.1, 50000)
        self.interface.confirm_order(order_id)
        self.mock_api_manager.send_order.assert_called_once()
        self.assertEqual(self.interface.orders[order_id].status, OrderStatus.SUBMITTED)

    def test_cancel_order(self):
        self.mock_api_manager.send_order.return_value = {"orderId": 123, "status": "NEW"}
        self.mock_api_manager.cancel_order.return_value = {"orderId": 123, "status": "CANCELED"}
        order_id = self.interface.create_limit_order("BTCUSDT", OrderSide.BUY, 0.1, 50000)
        self.interface.confirm_order(order_id)
        self.interface.cancel_order(order_id)
        self.mock_api_manager.cancel_order.assert_called_once()
        self.assertEqual(self.interface.orders[order_id].status, OrderStatus.CANCELLED)

    def test_check_order_status(self):
        self.mock_api_manager.send_order.return_value = {"orderId": 123, "status": "NEW", "executedQty": 0}
        self.mock_api_manager.get_order.return_value = {"orderId": 123, "status": "FILLED", "executedQty": 0.1, "cummulativeQuoteQty": 5000}
        order_id = self.interface.create_limit_order("BTCUSDT", OrderSide.BUY, 0.1, 50000)
        order = self.interface.orders[order_id]
        order.status = OrderStatus.SUBMITTED # Manually set to submitted
        order.exchange_order_id = "123"

        self.interface._check_order_status(order)
        self.mock_api_manager.get_order.assert_called_once()
        self.assertEqual(self.interface.orders[order_id].status, OrderStatus.FILLED)

if __name__ == '__main__':
    unittest.main()
