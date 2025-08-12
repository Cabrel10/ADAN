
import unittest
import time
from unittest.mock import patch, Mock
from src.adan_trading_bot.trading.secure_api_manager import SecureAPIManager, ExchangeType, APICredentials

class TestSecureAPIManager(unittest.TestCase):

    def setUp(self):
        self.manager = SecureAPIManager()
        self.manager.set_master_password("test_password")
        self.credentials = APICredentials(
            exchange=ExchangeType.BINANCE,
            api_key="test_api_key",
            api_secret="test_api_secret",
            sandbox=True
        )
        self.manager.add_credentials(self.credentials)

    @patch('requests.get')
    def test_get_exchange_info(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"symbols": [{"symbol": "BTCUSDT"}]}
        mock_get.return_value = mock_response

        # First call - should fetch from API
        exchange_info = self.manager.get_exchange_info(ExchangeType.BINANCE)
        self.assertIsNotNone(exchange_info)
        self.assertEqual(exchange_info["symbols"][0]["symbol"], "BTCUSDT")
        mock_get.assert_called_once()

        # Second call - should use cache
        exchange_info_cached = self.manager.get_exchange_info(ExchangeType.BINANCE)
        self.assertIsNotNone(exchange_info_cached)
        mock_get.assert_called_once() # Should not be called again

    @patch('requests.post')
    def test_send_order(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"orderId": 123, "status": "NEW"}
        mock_post.return_value = mock_response

        order_params = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": 0.1,
            "price": 50000,
            "timeInForce": "GTC"
        }
        response = self.manager.send_order(ExchangeType.BINANCE, order_params)
        self.assertIsNotNone(response)
        self.assertEqual(response["orderId"], 123)
        mock_post.assert_called_once()

    @patch('requests.get')
    def test_get_order(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"orderId": 123, "status": "FILLED"}
        mock_get.return_value = mock_response

        response = self.manager.get_order(ExchangeType.BINANCE, "BTCUSDT", "123")
        self.assertIsNotNone(response)
        self.assertEqual(response["status"], "FILLED")
        mock_get.assert_called_once()

    @patch('requests.delete')
    def test_cancel_order(self, mock_delete):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"orderId": 123, "status": "CANCELED"}
        mock_delete.return_value = mock_response

        response = self.manager.cancel_order(ExchangeType.BINANCE, "BTCUSDT", "123")
        self.assertIsNotNone(response)
        self.assertEqual(response["status"], "CANCELED")
        mock_delete.assert_called_once()

if __name__ == '__main__':
    unittest.main()
