"""
Tests unitaires pour la gestion des frais et des limites de capital dans le PortfolioManager.
"""

import unittest
import os
import sys
import yaml

# Assurer que le chemin du projet est dans sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

class TestFeesAndCaps(unittest.TestCase):
    """Valide la logique de frais et de cash gating du PortfolioManager."""

    def setUp(self):
        """Initialise un PortfolioManager avec une configuration de test."""
        self.config = {
            'environment': {
                'initial_balance': 10000.0,
                'commission': 0.001  # 0.1% de frais
            },
            'assets': ['BTCUSDT']
        }
        self.portfolio_manager = PortfolioManager(config=self.config)

    def test_01_entry_fee_deduction_and_cash_gate(self):
        """Vérifie que les frais d'entrée sont déduits et que le cash gate fonctionne."""
        pm = self.portfolio_manager
        initial_cash = pm.get_cash()
        self.assertEqual(initial_cash, 10000.0)

        # Tenter d'ouvrir une position qui est trop grande
        size_too_big = 1.0  # 1 BTC
        price = 50000.0
        cost = size_too_big * price # 50000
        fee = cost * 0.001 # 50
        required = cost + fee # 50050
        
        self.assertGreater(required, initial_cash)
        receipt_fail = pm.open_position('BTCUSDT', price, size_too_big, 0.02, 0.04, timestamp=pm._normalize_timestamp('2023-01-01T00:00:00'))
        
        # Vérifier que la position a été rejetée
        self.assertIsNone(receipt_fail, "La position aurait dû être rejetée pour cash insuffisant")
        self.assertEqual(pm.get_cash(), initial_cash, "Le cash ne doit pas changer après un trade rejeté")

        # Ouvrir une position valide
        size_valid = 0.1 # 0.1 BTC
        price = 50000.0
        cost = size_valid * price # 5000
        fee = cost * 0.001 # 5
        required = cost + fee # 5005

        self.assertLess(required, initial_cash)
        receipt_ok = pm.open_position('BTCUSDT', price, size_valid, 0.02, 0.04, timestamp=pm._normalize_timestamp('2023-01-01T01:00:00'))

        # Vérifier que la position est ouverte et que le cash a été débité
        self.assertIsNotNone(receipt_ok, "La position valide n'a pas pu être ouverte")
        expected_cash_after_open = initial_cash - required
        self.assertAlmostEqual(pm.get_cash(), expected_cash_after_open, places=4, msg="Le cash après ouverture est incorrect")
        
        position = pm.positions['BTCUSDT']
        self.assertTrue(position.is_open)
        self.assertEqual(position.size, size_valid)

    def test_02_net_pnl_includes_all_fees_on_close(self):
        """Vérifie que le PnL net et le cash final après clôture comptabilisent les deux frais."""
        pm = self.portfolio_manager
        initial_cash = pm.get_cash()

        # Ouvrir une position
        open_price = 50000.0
        size = 0.1
        open_cost = size * open_price # 5000
        entry_fee = open_cost * 0.001 # 5
        
        pm.open_position('BTCUSDT', open_price, size, 0.02, 0.04, timestamp=pm._normalize_timestamp('2023-01-01T01:00:00'))
        cash_after_open = initial_cash - (open_cost + entry_fee)
        self.assertAlmostEqual(pm.get_cash(), cash_after_open, places=4)

        # Clôturer la position avec un profit
        close_price = 52000.0
        exit_value = size * close_price # 5200
        exit_fee = exit_value * 0.001 # 5.2

        pnl_gross = (close_price - open_price) * size # (2000) * 0.1 = 200
        total_fees = entry_fee + exit_fee # 5 + 5.2 = 10.2
        pnl_net = pnl_gross - total_fees # 200 - 10.2 = 189.8

        close_receipt = pm.close_position('BTCUSDT', close_price, timestamp=pm._normalize_timestamp('2023-01-01T02:00:00'))

        # Vérifier le PnL dans le reçu
        self.assertIsNotNone(close_receipt)
        self.assertAlmostEqual(close_receipt['pnl'], pnl_net, places=4, msg="Le PnL net dans le reçu est incorrect")
        self.assertAlmostEqual(close_receipt['fees'], total_fees, places=4, msg="Le total des frais dans le reçu est incorrect")

        # Vérifier le cash final
        # cash_after_close = cash_after_open + (exit_value - exit_fee)
        # cash_after_close = (initial_cash - (open_cost + entry_fee)) + (exit_value - exit_fee)
        # cash_after_close = initial_cash + (exit_value - open_cost) - (entry_fee + exit_fee)
        # cash_after_close = initial_cash + pnl_gross - total_fees = initial_cash + pnl_net
        expected_final_cash = initial_cash + pnl_net
        self.assertAlmostEqual(pm.get_cash(), expected_final_cash, places=4, msg="Le solde de cash final est incorrect")

    def test_03_reject_non_finite_inputs_on_open(self):
        """Rejette les ouvertures avec des entrées non finies (NaN/Inf)."""
        pm = self.portfolio_manager
        initial_cash = pm.get_cash()

        # NaN size
        rcpt_nan = pm.open_position('BTCUSDT', price=100.0, size=float('nan'), stop_loss_pct=0.02, take_profit_pct=0.04, timestamp=pm._normalize_timestamp('2023-01-01T03:00:00'))
        self.assertIsNone(rcpt_nan)
        self.assertEqual(pm.get_cash(), initial_cash)

        # Inf price
        rcpt_inf = pm.open_position('BTCUSDT', price=float('inf'), size=0.1, stop_loss_pct=0.02, take_profit_pct=0.04, timestamp=pm._normalize_timestamp('2023-01-01T03:05:00'))
        self.assertIsNone(rcpt_inf)
        self.assertEqual(pm.get_cash(), initial_cash)

    def test_04_reject_notional_above_cap(self):
        """Rejette si le notional demandé dépasse le cap (pos_size_pct par défaut 10%)."""
        pm = self.portfolio_manager
        initial_cash = pm.get_cash()

        # Equity ~ 10000, cap 10% => 1000 USDT max notional
        # Demander 2000 USDT
        price = 20000.0
        size = 0.1  # notional=2000
        rcpt = pm.open_position('BTCUSDT', price=price, size=size, stop_loss_pct=0.02, take_profit_pct=0.04, timestamp=pm._normalize_timestamp('2023-01-01T04:00:00'))
        self.assertIsNone(rcpt, "La position aurait dû être rejetée par le cap notional")
        self.assertEqual(pm.get_cash(), initial_cash)

if __name__ == '__main__':
    unittest.main()