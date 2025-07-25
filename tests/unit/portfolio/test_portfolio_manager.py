import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager, Position

class TestPortfolioManager(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPortfolioManager, self).__init__(*args, **kwargs)
        
    def setUp(self):
        self.env_config = {
            'initial_equity': 1000.0,
            'assets': ['BTC/USDT'],
            'trading_rules': {
                'futures_enabled': False,
                'commission_pct': 0.1,
                'min_trade_size': 0.001,
                'min_notional_value': 10.0,
                'max_notional_value': 100000.0,
                'stop_loss': 0.05,
                'take_profit': 0.1
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
        self.portfolio_manager = PortfolioManager(self.env_config)

    def test_get_current_tier(self):
        # Test edge case: 30 USDT (should be Small Capital)
        self.portfolio_manager.portfolio_value = 30.0
        tier = self.portfolio_manager.get_current_tier()
        self.assertEqual(tier['name'], 'Small Capital')

        # Test below min_capital
        self.portfolio_manager.portfolio_value = 5.0
        tier = self.portfolio_manager.get_current_tier()
        self.assertEqual(tier['name'], 'Micro Capital')

        # Test above max_capital of a tier
        self.portfolio_manager.portfolio_value = 150.0
        tier = self.portfolio_manager.get_current_tier()
        self.assertEqual(tier['name'], 'Medium Capital')

        # Test above all tiers
        self.portfolio_manager.portfolio_value = 5000.0
        tier = self.portfolio_manager.get_current_tier()
        self.assertEqual(tier['name'], 'Enterprise')

    def test_calculate_position_size(self):
        # Scenario where stop-loss limits position size
        self.portfolio_manager.portfolio_value = 100.0  # Medium Capital
        position_size = self.portfolio_manager.calculate_position_size(
            action_type='buy',
            asset='BTC/USDT',
            current_price=50000,
            confidence=0.8,
            stop_loss_pct=2.0
        )
        # Expected: risk_amount = 100 * 0.01 = 1. risk_per_share = 50000 * 0.02 = 1000. size = 1 / 1000 = 0.001
        # With confidence: 0.001 * 0.8 = 0.0008. But min_trade_size is 0.001
        self.assertGreaterEqual(position_size, 0.001)


        # "Normal" scenario: no stop-loss, uses max_position_size_pct
        position_size_no_sl = self.portfolio_manager.calculate_position_size(
            action_type='buy',
            asset='BTC/USDT',
            current_price=50000,
            confidence=0.8
        )
        # Expected: max_pos_value = 100 * 0.6 = 60. size = 60 / 50000 = 0.0012
        # With confidence: 0.0012 * 0.8 = 0.00096. But min_trade_size is 0.001
        self.assertGreaterEqual(position_size_no_sl, 0.001)

    def test_check_liquidation(self):
        # Simulate drawdown just below the threshold
        self.portfolio_manager.initial_equity = 100.0
        self.portfolio_manager.portfolio_value = 81.0  # 19% drawdown < 25% (Medium Capital)
        liquidated = self.portfolio_manager.check_liquidation(current_prices={'BTC/USDT': 50000})
        self.assertFalse(liquidated)

        # Simulate drawdown just above the threshold with no open positions
        self.portfolio_manager.portfolio_value = 74.0  # 26% drawdown > 25% (Medium Capital)
        liquidated = self.portfolio_manager.check_liquidation(current_prices={'BTC/USDT': 50000})
        self.assertTrue(liquidated)  # Should return True even with no positions to close
        
        # Test with an open position
        position = Position()
        position.open(entry_price=49000, size=0.1)
        self.portfolio_manager.positions['BTC/USDT'] = position
        
        with patch.object(self.portfolio_manager, 'close_position') as mock_close_position:
            liquidated = self.portfolio_manager.check_liquidation(current_prices={'BTC/USDT': 50000})
            self.assertTrue(liquidated)
            mock_close_position.assert_called_once_with('BTC/USDT', 50000)
    
    def test_initialization(self):
        # Test avec une configuration valide
        pm = PortfolioManager(self.env_config)
        self.assertEqual(pm.initial_equity, 1000.0)
        self.assertEqual(pm.current_equity, 1000.0)
        self.assertIsInstance(pm.positions, dict)
        self.assertEqual(len(pm.positions), 0)
    
    def test_update_market_price(self):
        # Test sans position ouverte
        self.portfolio_manager.update_market_price({'BTC/USDT': 50000})
        self.assertEqual(self.portfolio_manager.current_equity, 1000.0)
        
        # Test avec position ouverte
        position = Position()
        position.open(entry_price=49000, size=0.1)
        self.portfolio_manager.positions['BTC/USDT'] = position
        self.portfolio_manager.update_market_price({'BTC/USDT': 51000})
        # La valeur du portefeuille devrait être mise à jour dans update_market_price
        # 1000 + (51000 - 49000) * 0.1 = 1200
        self.assertAlmostEqual(self.portfolio_manager.current_equity, 1200.0, places=2)
    
    def test_close_position(self):
        # Test avec position existante
        position = Position()
        position.open(entry_price=49000, size=0.1)
        self.portfolio_manager.positions['BTC/USDT'] = position
        
        # Fermer la position à un prix plus élevé
        pnl = self.portfolio_manager.close_position('BTC/USDT', 50000)
        # PnL = (50000 - 49000) * 0.1 = 100
        self.assertAlmostEqual(pnl, 100.0, places=2)
        self.assertNotIn('BTC/USDT', self.portfolio_manager.positions)
        
        # Test avec position inexistante (ne devrait pas lever d'erreur)
        pnl = self.portfolio_manager.close_position('ETH/USDT', 2000)
        self.assertEqual(pnl, 0.0)
    
    def test_get_portfolio_value(self):
        # Test sans position ouverte
        self.assertEqual(self.portfolio_manager.get_portfolio_value(), 1000.0)
        
        # Test avec position ouverte
        position = Position()
        position.open(entry_price=49000, size=0.1)
        self.portfolio_manager.positions['BTC/USDT'] = position
        self.portfolio_manager.update_market_price({'BTC/USDT': 51000})
        # 1000 + (51000 - 49000) * 0.1 = 1200
        self.assertAlmostEqual(self.portfolio_manager.get_portfolio_value(), 1200.0, places=2)

if __name__ == '__main__':
    unittest.main()
