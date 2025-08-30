
import unittest
import os
import tempfile
import time
from unittest.mock import patch, Mock
from pathlib import Path
from src.adan_trading_bot.trading.secure_api_manager import SecureAPIManager, ExchangeType, APICredentials, SecurityError

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

    def test_environment_variable_loading(self):
        """Test loading credentials from environment variables"""
        # Set test environment variables
        os.environ['BINANCE_API_KEY'] = 'env_test_key'
        os.environ['BINANCE_API_SECRET'] = 'env_test_secret'
        os.environ['BINANCE_SANDBOX'] = 'true'
        
        try:
            # Create new manager to trigger environment loading
            manager = SecureAPIManager()
            
            # Get credentials - should return environment ones
            credentials = manager.get_credentials(ExchangeType.BINANCE)
            
            self.assertIsNotNone(credentials)
            self.assertEqual(credentials.name, "Environment")
            self.assertEqual(credentials.api_key, 'env_test_key')
            self.assertEqual(credentials.api_secret, 'env_test_secret')
            self.assertTrue(credentials.sandbox)
            
        finally:
            # Clean up environment variables
            for key in ['BINANCE_API_KEY', 'BINANCE_API_SECRET', 'BINANCE_SANDBOX']:
                os.environ.pop(key, None)

    def test_environment_priority_over_encrypted(self):
        """Test that environment variables have priority over encrypted files"""
        # Set environment variables
        os.environ['BINANCE_API_KEY'] = 'env_priority_key'
        os.environ['BINANCE_API_SECRET'] = 'env_priority_secret'
        os.environ['BINANCE_SANDBOX'] = 'false'
        
        try:
            # Create manager with environment variables
            manager = SecureAPIManager()
            manager.set_master_password("test_password")
            
            # Add encrypted credentials with different values
            encrypted_creds = APICredentials(
                exchange=ExchangeType.BINANCE,
                api_key='encrypted_key',
                api_secret='encrypted_secret',
                sandbox=True,
                name='Default'
            )
            manager.add_credentials(encrypted_creds)
            
            # Get credentials - should return environment ones
            credentials = manager.get_credentials(ExchangeType.BINANCE)
            
            self.assertIsNotNone(credentials)
            self.assertEqual(credentials.api_key, 'env_priority_key')
            self.assertEqual(credentials.name, "Environment")
            self.assertFalse(credentials.sandbox)  # From environment
            
        finally:
            # Clean up
            for key in ['BINANCE_API_KEY', 'BINANCE_API_SECRET', 'BINANCE_SANDBOX']:
                os.environ.pop(key, None)

    def test_security_validation_with_hardcoded_keys(self):
        """Test security validation detects hardcoded API keys"""
        # Create a temporary file with hardcoded keys
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("AIzaSyC-PnZoTqGlr_VvCc9XHs7h5oXMdKKds0I\n")
            f.write("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\n")
            temp_file = f.name
        
        try:
            # Rename to a suspicious filename
            suspicious_file = Path("gemini_api_keys.txt")
            Path(temp_file).rename(suspicious_file)
            
            # Should raise SecurityError
            with self.assertRaises(SecurityError):
                SecureAPIManager()
                
        finally:
            # Clean up
            if suspicious_file.exists():
                suspicious_file.unlink()

    def test_create_env_setup_guide(self):
        """Test creation of environment setup guide"""
        with tempfile.TemporaryDirectory() as temp_dir:
            guide_path = Path(temp_dir) / "test_env_guide.md"
            
            manager = SecureAPIManager()
            manager.create_env_setup_guide(str(guide_path))
            
            self.assertTrue(guide_path.exists())
            
            content = guide_path.read_text()
            self.assertIn("Environment Variables Setup Guide", content)
            self.assertIn("BINANCE_API_KEY", content)
            self.assertIn("Security Best Practices", content)
            self.assertIn("Never commit API keys to version control", content)

if __name__ == '__main__':
    unittest.main()
