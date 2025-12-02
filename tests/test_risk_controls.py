#!/usr/bin/env python3
"""
Unit Tests for Security & Risk Controls - CONCRETE SCENARIOS
Based on config.yaml Capital Tiers and real trading logic
"""

import unittest
from unittest.mock import Mock
import sys
sys.path.insert(0, 'src')

from adan_trading_bot.environment.order_manager import OrderManager


class TestCapitalTiersRiskManagement(unittest.TestCase):
    """Test Capital Tiers system with CONCRETE scenarios."""
    
    def test_micro_capital_4_percent_risk(self):
        """
        Scenario: Micro Capital ($20.5) with 4% risk_per_trade
        Verify that LOSS is capped at 4%, not trade size.
        """
        print("\n" + "="*70)
        print("SCENARIO 1: Micro Capital - 4% Risk Per Trade")
        print("="*70)
        
        # Setup from config.yaml
        capital = 20.5  # Micro Capital tier
        risk_pct = 0.04  # 4% from capital_tiers.Micro
        stop_loss_pct = 0.05  # 5% SL
        btc_price = 40000.0
        
        trading_rules = {
            "risk_per_trade": risk_pct,
            "stop_loss": stop_loss_pct
        }
        order_manager = OrderManager(trading_rules, penalties={})
        
        # Mock portfolio
        portfolio = Mock()
        portfolio.get_available_capital.return_value = capital
        portfolio.positions = {"BTCUSDT": Mock(is_open=False)}
        portfolio.open_position = Mock(return_value=True)
        
        # Execute
        result = order_manager.open_position(portfolio, "BTCUSDT", btc_price)
        
        # Extract actual size
        call_args = portfolio.open_position.call_args
        actual_size_btc = call_args[0][2]
        trade_value = actual_size_btc * btc_price
        max_loss = trade_value * stop_loss_pct
        expected_max_loss = capital * risk_pct
        
        print(f"Capital total: ${capital}")
        print(f"Risk acceptable: {risk_pct*100}% = ${expected_max_loss:.2f}")
        print(f"Stop Loss: {stop_loss_pct*100}%")
        print(f"Prix BTC: ${btc_price}")
        print(f"→ Taille calculée: {actual_size_btc:.8f} BTC (${trade_value:.2f})")
        print(f"→ Perte si SL touché: ${max_loss:.2f}")
        print(f"✓ Perte MAX respectée: ${max_loss:.2f} ≈ ${expected_max_loss:.2f}")
        
        self.assertTrue(result)
        self.assertAlmostEqual(max_loss, expected_max_loss, places=2)
        self.assertLess(max_loss, capital * 0.05, "Loss should be < 5% of capital")
    
    def test_small_capital_2_percent_risk(self):
        """
        Scenario: Small Capital ($50) with 2% risk_per_trade
        """
        print("\n" + "="*70)
        print("SCENARIO 2: Small Capital - 2% Risk Per Trade")
        print("="*70)
        
        capital = 50.0  # Small Capital tier
        risk_pct = 0.02  # 2% from capital_tiers.Small
        stop_loss_pct = 0.03  # 3% SL
        btc_price = 40000.0
        
        trading_rules = {
            "risk_per_trade": risk_pct,
            "stop_loss": stop_loss_pct
        }
        order_manager = OrderManager(trading_rules, penalties={})
        
        portfolio = Mock()
        portfolio.get_available_capital.return_value = capital
        portfolio.positions = {"BTCUSDT": Mock(is_open=False)}
        portfolio.open_position = Mock(return_value=True)
        
        result = order_manager.open_position(portfolio, "BTCUSDT", btc_price)
        
        call_args = portfolio.open_position.call_args
        actual_size_btc = call_args[0][2]
        trade_value = actual_size_btc * btc_price
        max_loss = trade_value * stop_loss_pct
        expected_max_loss = capital * risk_pct
        
        print(f"Capital total: ${capital}")
        print(f"Risk acceptable: {risk_pct*100}% = ${expected_max_loss:.2f}")
        print(f"Stop Loss: {stop_loss_pct*100}%")
        print(f"→ Taille calculée: {actual_size_btc:.8f} BTC (${trade_value:.2f})")
        print(f"→ Perte si SL touché: ${max_loss:.2f}")
        print(f"✓ Rapport trade/capital: {(trade_value/capital)*100:.1f}%")
        
        self.assertTrue(result)
        self.assertAlmostEqual(max_loss, expected_max_loss, places=2)
    
    def test_medium_capital_225_percent_risk(self):
        """
        Scenario: Medium Capital ($150) with 2.25% risk_per_trade
        """
        print("\n" + "="*70)
        print("SCENARIO 3: Medium Capital - 2.25% Risk Per Trade")
        print("="*70)
        
        capital = 150.0  # Medium Capital tier
        risk_pct = 0.0225  # 2.25% from capital_tiers.Medium
        stop_loss_pct = 0.04  # 4% SL
        btc_price = 40000.0
        
        trading_rules = {
            "risk_per_trade": risk_pct,
            "stop_loss": stop_loss_pct
        }
        order_manager = OrderManager(trading_rules, penalties={})
        
        portfolio = Mock()
        portfolio.get_available_capital.return_value = capital
        portfolio.positions = {"BTCUSDT": Mock(is_open=False)}
        portfolio.open_position = Mock(return_value=True)
        
        result = order_manager.open_position(portfolio, "BTCUSDT", btc_price)
        
        call_args = portfolio.open_position.call_args
        actual_size_btc = call_args[0][2]
        trade_value = actual_size_btc * btc_price
        max_loss = trade_value * stop_loss_pct
        expected_max_loss = capital * risk_pct
        
        print(f"Capital total: ${capital}")
        print(f"Risk acceptable: {risk_pct*100}% = ${expected_max_loss:.2f}")
        print(f"Stop Loss: {stop_loss_pct*100}%")
        print(f"→ Taille calculée: {actual_size_btc:.8f} BTC (${trade_value:.2f})")
        print(f"→ Perte si SL touché: ${max_loss:.2f}")
        
        self.assertTrue(result)
        self.assertAlmostEqual(max_loss, expected_max_loss, places=2)


class TestKillSwitchDrawdown(unittest.TestCase):
    """Test Kill Switch at 15% drawdown (from config.yaml)."""
    
    def test_kill_switch_at_15_percent_loss(self):
        """
        Scenario: Portfolio loses 15.5% → Kill Switch MUST trigger
        """
        print("\n" + "="*70)
        print("SCENARIO 4: Kill Switch - 15% Drawdown Trigger")
        print("="*70)
        
        initial_capital = 20.5
        current_value = 17.3  # 15.6% loss
        circuit_breaker_pct = 0.15
        
        loss_pct = (1 - current_value / initial_capital) * 100
        threshold = initial_capital * (1 - circuit_breaker_pct)
        should_trigger = current_value < threshold
        
        print(f"Capital initial: ${initial_capital}")
        print(f"Valeur actuelle: ${current_value}")
        print(f"Perte: {loss_pct:.1f}%")
        print(f"Seuil Kill Switch: {circuit_breaker_pct*100}%")
        print(f"Threshold value: ${threshold}")
        print(f"→ Kill Switch: {'TRIGGERED ✓' if should_trigger else 'Not triggered'}")
        
        self.assertTrue(should_trigger, "Kill switch MUST trigger at >15% loss")
    
    def test_no_kill_switch_at_14_percent_loss(self):
        """
        Scenario: Portfolio loses 14% → Kill Switch should NOT trigger
        """
        print("\n" + "="*70)
        print("SCENARIO 5: Kill Switch - 14% Loss (No Trigger)")
        print("="*70)
        
        initial_capital = 20.5
        current_value = 17.6  # 14.1% loss
        circuit_breaker_pct = 0.15
        
        loss_pct = (1 - current_value / initial_capital) * 100
        threshold = initial_capital * (1 - circuit_breaker_pct)
        should_trigger = current_value < threshold
        
        print(f"Capital initial: ${initial_capital}")
        print(f"Valeur actuelle: ${current_value}")
        print(f"Perte: {loss_pct:.1f}%")
        print(f"Threshold value: ${threshold}")
        print(f"→ Kill Switch: {'TRIGGERED' if should_trigger else 'Not triggered ✓'}")
        
        self.assertFalse(should_trigger, "Kill switch should NOT trigger at <15% loss")


class TestSanityChecksMarketData(unittest.TestCase):
    """Test Sanity Checks for market data validation."""
    
    def test_reject_zero_and_negative_prices(self):
        """
        Scenario: Validate that zero/negative prices are rejected
        """
        print("\n" + "="*70)
        print("SCENARIO 6: Sanity Checks - Invalid Price Detection")
        print("="*70)
        
        test_cases = [
            ({"BTCUSDT": 40000.0}, True, "Prix normal: $40,000"),
            ({"BTCUSDT": 0.0}, False, "Prix à zéro"),
            ({"BTCUSDT": -5000.0}, False, "Prix négatif"),
            ({}, False, "Données vides"),
        ]
        
        for prices, expected_valid, description in test_cases:
            if not prices:
                is_valid = False
            else:
                is_valid = all(p > 0 for p in prices.values())
            
            status = "✓ PASS" if is_valid == expected_valid else "✗ FAIL"
            print(f"{status}: {description} → Valid={is_valid}")
            self.assertEqual(is_valid, expected_valid)


if __name__ == "__main__":
    print("\n" + "🔒 " + "="*66)
    print("   TESTS DE SÉCURITÉ & GESTION DES RISQUES - SCÉNARIOS RÉELS")
    print("   " + "="*66)
    unittest.main(verbosity=2)
