
import unittest
import pandas as pd
from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

class TestDynamicSizingIntegration(unittest.TestCase):
    def setUp(self):
        self.base_config = {
            'initial_equity': 100.0,
            'assets': ['BTC/USDT'],
            'trading_rules': {
                'min_trade_size': 0.001,
                'min_notional_value': 10.0,
                'max_notional_value': 100000.0,
                'stop_loss_pct': 5.0,
                'take_profit_pct': 10.0,
                'futures_enabled': False,
                'commission_pct': 0.1
            },
            'risk_management': {
                'capital_tiers': [
                    {'name': 'Micro Capital', 'min_capital': 0, 'max_capital': 30, 'max_position_size_pct': 90.0, 'leverage': 1.0, 'risk_per_trade_pct': 2.0, 'max_drawdown_pct': 15.0},
                    {'name': 'Small Capital', 'min_capital': 30, 'max_capital': 100, 'max_position_size_pct': 70.0, 'leverage': 1.0, 'risk_per_trade_pct': 1.5, 'max_drawdown_pct': 20.0},
                    {'name': 'Medium Capital', 'min_capital': 100, 'max_capital': 300, 'max_position_size_pct': 60.0, 'leverage': 1.0, 'risk_per_trade_pct': 1.0, 'max_drawdown_pct': 25.0},
                    {'name': 'High Capital', 'min_capital': 300, 'max_capital': 1000, 'max_position_size_pct': 35.0, 'leverage': 1.0, 'risk_per_trade_pct': 0.75, 'max_drawdown_pct': 30.0},
                    {'name': 'Enterprise', 'min_capital': 1000, 'max_capital': None, 'max_position_size_pct': 20.0, 'leverage': 1.0, 'risk_per_trade_pct': 0.5, 'max_drawdown_pct': 35.0}
                ]
            }
        }

    def test_tier_2_sizing(self):
        """Test position sizing for a Tier 2 portfolio."""
        config = self.base_config.copy()
        config['initial_equity'] = 50.0  # Tier 2
        portfolio_manager = PortfolioManager(config)

        position_size = portfolio_manager.calculate_position_size(
            action_type='buy',
            asset='BTC/USDT',
            current_price=50000,
            confidence=0.8,
            stop_loss_pct=2.0
        )
        
        # 50 * 0.015 / (50000 * 0.02) * 0.8 = 0.0006, clamped to min_trade_size (0.001), then clamped by max_position_value (50 * 0.7 = 35), so 35/50000 = 0.0007
        self.assertAlmostEqual(position_size, 0.0007, places=5)

    def test_tier_4_sizing(self):
        """Test position sizing for a Tier 4 portfolio."""
        config = self.base_config.copy()
        config['initial_equity'] = 5000.0  # Tier 4
        portfolio_manager = PortfolioManager(config)

        position_size = portfolio_manager.calculate_position_size(
            action_type='buy',
            asset='BTC/USDT',
            current_price=50000,
            confidence=0.8,
            stop_loss_pct=2.0
        )
        
        # 5000 * 0.005 / (50000 * 0.02) * 0.8 = 0.02
        self.assertAlmostEqual(position_size, 0.02, places=5)

if __name__ == '__main__':
    unittest.main()
