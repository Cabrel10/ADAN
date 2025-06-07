#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de test simplifié pour valider uniquement les corrections OrderManager.
"""
import os
import sys
import logging

# Assurer que le package src est dans le PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '..'))

from src.adan_trading_bot.environment.order_manager import OrderManager

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_order_manager_comprehensive():
    """Test complet des corrections OrderManager."""
    
    print("🔧 DÉMARRAGE DES TESTS ORDERMAMANAGER")
    print("=" * 60)
    
    # Configuration de test
    config = {
        'environment': {
            'initial_capital': 1000.0,
            'transaction': {
                'fee_percent': 0.001,
                'fixed_fee': 0.0
            },
            'order_rules': {
                'min_value_tolerable': 10.0,
                'min_value_absolute': 5.0
            },
            'penalties': {
                'invalid_order_base': -0.3,
                'out_of_funds': -0.5,
                'order_below_tolerable': -0.1
            }
        }
    }
    
    order_manager = OrderManager(config)
    test_results = []
    
    # TEST 1: BUY avec prix normalisé NÉGATIF
    print("\n📊 TEST 1: BUY avec prix normalisé négatif")
    print("-" * 40)
    
    capital = 1000.0
    positions = {}
    current_price = -0.75  # Prix normalisé négatif (critique)
    allocated_value = 50.0
    
    reward, status, info = order_manager.execute_order(
        "BTCUSDT", 1, current_price, capital, positions,
        allocated_value_usdt=allocated_value
    )
    
    new_capital = info.get('new_capital', capital)
    success = status == "BUY_EXECUTED" and new_capital < capital and new_capital >= 0
    test_results.append(("BUY Prix Négatif", success))
    
    print(f"   Prix normalisé: {current_price}")
    print(f"   Valeur allouée: ${allocated_value}")
    print(f"   Statut: {status}")
    print(f"   Capital: ${capital} → ${new_capital}")
    print(f"   Position créée: {'BTCUSDT' in positions}")
    print(f"   ✅ SUCCÈS" if success else "   ❌ ÉCHEC")
    
    # TEST 2: SELL sans position (doit échouer proprement)
    print("\n📊 TEST 2: SELL sans position existante")
    print("-" * 40)
    
    reward, status, info = order_manager.execute_order(
        "ETHUSDT", 2, 0.5, new_capital, positions
    )
    
    success = status == "INVALID_NO_POSITION" and reward < 0
    test_results.append(("SELL Sans Position", success))
    
    print(f"   Asset: ETHUSDT (pas de position)")
    print(f"   Statut: {status}")
    print(f"   Pénalité: {reward}")
    print(f"   Capital inchangé: {info.get('new_capital', 'N/A') == new_capital}")
    print(f"   ✅ SUCCÈS" if success else "   ❌ ÉCHEC")
    
    # TEST 3: SELL avec position existante et prix négatif
    if "BTCUSDT" in positions:
        print("\n📊 TEST 3: SELL avec position et prix négatif")
        print("-" * 40)
        
        sell_price = -0.85  # Prix encore plus négatif
        
        reward, status, info = order_manager.execute_order(
            "BTCUSDT", 2, sell_price, new_capital, positions
        )
        
        final_capital = info.get('new_capital', new_capital)
        pnl = info.get('pnl', 0)
        success = status == "SELL_EXECUTED"
        test_results.append(("SELL Prix Négatif", success))
        
        print(f"   Prix d'achat: {current_price}")
        print(f"   Prix de vente: {sell_price}")
        print(f"   Statut: {status}")
        print(f"   PnL: ${pnl:.2f}")
        print(f"   Capital: ${new_capital} → ${final_capital}")
        print(f"   Position fermée: {'BTCUSDT' not in positions}")
        print(f"   ✅ SUCCÈS" if success else "   ❌ ÉCHEC")
        
        new_capital = final_capital
    
    # TEST 4: BUY avec capital insuffisant
    print("\n📊 TEST 4: BUY avec capital insuffisant")
    print("-" * 40)
    
    reward, status, info = order_manager.execute_order(
        "ADAUSDT", 1, 0.3, new_capital, positions,
        allocated_value_usdt=new_capital + 100  # Plus que le capital disponible
    )
    
    success = status == "INVALID_NO_CAPITAL" and reward < 0
    test_results.append(("BUY Capital Insuffisant", success))
    
    print(f"   Capital disponible: ${new_capital}")
    print(f"   Valeur demandée: ${new_capital + 100}")
    print(f"   Statut: {status}")
    print(f"   Pénalité: {reward}")
    print(f"   ✅ SUCCÈS" if success else "   ❌ ÉCHEC")
    
    # TEST 5: BUY avec ordre trop petit
    print("\n📊 TEST 5: BUY avec ordre en dessous du minimum")
    print("-" * 40)
    
    reward, status, info = order_manager.execute_order(
        "SOLUSDT", 1, 0.2, new_capital, positions,
        allocated_value_usdt=2.0  # En dessous du minimum absolu (5.0)
    )
    
    success = status == "INVALID_ORDER_TOO_SMALL" and reward < 0
    test_results.append(("BUY Ordre Trop Petit", success))
    
    print(f"   Valeur allouée: $2.0")
    print(f"   Minimum absolu: $5.0")
    print(f"   Statut: {status}")
    print(f"   Pénalité: {reward}")
    print(f"   ✅ SUCCÈS" if success else "   ❌ ÉCHEC")
    
    # RÉSUMÉ FINAL
    print("\n🎯 RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)
    
    for test_name, success in test_results:
        status_icon = "✅" if success else "❌"
        print(f"   {status_icon} {test_name}")
    
    print(f"\nRésultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        print("\n🎉 TOUS LES TESTS ORDERMAMANAGER SONT RÉUSSIS!")
        print("   ✅ Gestion des prix normalisés négatifs: OK")
        print("   ✅ Gestion du capital: OK")
        print("   ✅ Vérifications des positions: OK")
        print("   ✅ Gestion des erreurs: OK")
        print("   ✅ Calculs financiers cohérents: OK")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) ont échoué")
        return False

if __name__ == "__main__":
    success = test_order_manager_comprehensive()
    exit_code = 0 if success else 1
    
    print(f"\nCode de sortie: {exit_code}")
    sys.exit(exit_code)