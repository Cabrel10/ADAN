
import unittest
from unittest.mock import MagicMock

from adan_trading_bot.risk_management.position_manager import PositionManager


class TestPositionManager(unittest.TestCase):
    def setUp(self):
        self.config = {
            'position_management': {
                'trailing_stop_loss_pct': 0.05
            }
        }
        self.position_manager = PositionManager(self.config)
        self.portfolio_manager = MagicMock()

    def test_open_position(self):
        self.position_manager.open_position("BTC", 1, 50000, self.portfolio_manager)
        self.portfolio_manager.open_position.assert_called_once_with("BTC", 1, 50000)

    def test_close_position(self):
        self.position_manager.close_position("BTC", 52000, self.portfolio_manager)
        self.portfolio_manager.close_position.assert_called_once_with("BTC", 52000)

    def test_adjust_position(self):
        self.position_manager.adjust_position("BTC", 2, 51000, self.portfolio_manager)
        self.portfolio_manager.adjust_position.assert_called_once_with("BTC", 2, 51000)

    def test_update_trailing_stop_loss(self):
        position = MagicMock()
        position.is_open = True
        position.stop_loss = 48000
        self.portfolio_manager.get_position.return_value = position

        self.position_manager.update_trailing_stop_loss("BTC", 52000, self.portfolio_manager)
        self.portfolio_manager.update_position_stop_loss.assert_called_once_with("BTC", 49400.0)


if __name__ == '__main__':
    unittest.main()
