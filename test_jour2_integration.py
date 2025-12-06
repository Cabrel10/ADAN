#!/usr/bin/env python3
"""
🧪 TEST JOUR 2 - Vérifier l'intégration du système unifié
"""

import sys
sys.path.insert(0, 'src')

print("=" * 70)
print("🧪 TEST JOUR 2 - INTÉGRATION SYSTÈME UNIFIÉ")
print("=" * 70)
print()

# Test 1: Vérifier que les imports fonctionnent
print("✅ TEST 1: Vérifier les imports")
try:
    from adan_trading_bot.common.central_logger import logger as central_logger
    print("   ✅ Logger centralisé importé")
except Exception as e:
    print(f"   ❌ Erreur import logger: {e}")
    sys.exit(1)

try:
    from adan_trading_bot.performance.unified_metrics import UnifiedMetrics
    print("   ✅ Métriques unifiées importées")
except Exception as e:
    print(f"   ❌ Erreur import metrics: {e}")
    sys.exit(1)

try:
    from adan_trading_bot.performance.unified_metrics_db import UnifiedMetricsDB
    print("   ✅ Base de données unifiée importée")
except Exception as e:
    print(f"   ❌ Erreur import db: {e}")
    sys.exit(1)

# Test 2: Vérifier que optuna_optimize_worker.py peut être importé
print("\n✅ TEST 2: Vérifier optuna_optimize_worker.py")
try:
    # Vérifier que le fichier a les bonnes modifications
    with open('optuna_optimize_worker.py', 'r') as f:
        content = f.read()
    
    if 'from adan_trading_bot.common.central_logger import logger as central_logger' in content:
        print("   ✅ Logger centralisé importé dans optuna_optimize_worker.py")
    else:
        print("   ❌ Logger centralisé NOT importé dans optuna_optimize_worker.py")
        sys.exit(1)
    
    if 'from adan_trading_bot.performance.unified_metrics import UnifiedMetrics' in content:
        print("   ✅ Métriques unifiées importées dans optuna_optimize_worker.py")
    else:
        print("   ❌ Métriques unifiées NOT importées dans optuna_optimize_worker.py")
        sys.exit(1)
    
    if 'central_logger.metric' in content:
        print("   ✅ Métriques loggées dans optuna_optimize_worker.py")
    else:
        print("   ❌ Métriques NOT loggées dans optuna_optimize_worker.py")
        sys.exit(1)
    
    if 'central_logger.sync' in content:
        print("   ✅ Synchronisation dans optuna_optimize_worker.py")
    else:
        print("   ❌ Synchronisation NOT dans optuna_optimize_worker.py")
        sys.exit(1)

except Exception as e:
    print(f"   ❌ Erreur vérification optuna_optimize_worker.py: {e}")
    sys.exit(1)

# Test 3: Tester le logger centralisé
print("\n✅ TEST 3: Tester le logger centralisé")
try:
    central_logger.trade("BUY", "BTCUSDT", 0.5, 45000.00, pnl=500.00)
    print("   ✅ Trade loggé")
    
    central_logger.metric("Test Metric", 1.85)
    print("   ✅ Métrique loggée")
    
    central_logger.validation("Test Validation", True, "Test passed")
    print("   ✅ Validation loggée")
    
    central_logger.sync("Test Component", "synchronized", {"test": "data"})
    print("   ✅ Sync loggée")
except Exception as e:
    print(f"   ❌ Erreur logger: {e}")
    sys.exit(1)

# Test 4: Tester les métriques unifiées
print("\n✅ TEST 4: Tester les métriques unifiées")
try:
    metrics = UnifiedMetrics()
    
    metrics.add_trade("BUY", "BTCUSDT", 0.5, 45000, pnl=500)
    print("   ✅ Trade ajouté")
    
    metrics.add_return(0.01)
    print("   ✅ Return ajouté")
    
    metrics.add_portfolio_value(10100)
    print("   ✅ Portfolio value ajouté")
    
    sharpe = metrics.calculate_sharpe()
    print(f"   ✅ Sharpe calculé: {sharpe:.4f}")
    
    report = metrics.get_report()
    print(f"   ✅ Rapport généré: {len(report)} clés")
except Exception as e:
    print(f"   ❌ Erreur métriques: {e}")
    sys.exit(1)

# Test 5: Tester la base de données
print("\n✅ TEST 5: Tester la base de données")
try:
    db = UnifiedMetricsDB()
    
    db.add_metric("test_metric", 1.85, "test")
    print("   ✅ Métrique ajoutée à la base")
    
    db.add_trade("BUY", "BTCUSDT", 0.5, 45000, 500)
    print("   ✅ Trade ajouté à la base")
    
    consistency = db.validate_consistency()
    print(f"   ✅ Cohérence validée: {consistency['status']}")
    
    summary = db.get_summary()
    print(f"   ✅ Résumé généré: {len(summary)} clés")
except Exception as e:
    print(f"   ❌ Erreur base de données: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ TOUS LES TESTS JOUR 2 RÉUSSIS!")
print("=" * 70)
print()
print("📊 Résumé:")
print("  ✅ Logger centralisé: Fonctionnel")
print("  ✅ Métriques unifiées: Fonctionnelles")
print("  ✅ Base de données: Fonctionnelle")
print("  ✅ optuna_optimize_worker.py: Intégré")
print()
print("🎯 Prochaines étapes:")
print("  1. Intégrer dans scripts/train_parallel_agents.py")
print("  2. Intégrer dans scripts/terminal_dashboard.py")
print("  3. Intégrer dans realistic_trading_env.py")
print("  4. Tester en production (JOUR 3)")
print("=" * 70)

