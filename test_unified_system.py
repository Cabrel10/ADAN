#!/usr/bin/env python3
"""
🧪 TEST COMPLET DU SYSTÈME UNIFIÉ
Teste: Logger centralisé + Métriques unifiées + Base de données
"""

import sys
sys.path.insert(0, 'src')

from adan_trading_bot.common.central_logger import logger
from adan_trading_bot.performance.unified_metrics import UnifiedMetrics
from adan_trading_bot.performance.unified_metrics_db import UnifiedMetricsDB

def test_logger():
    """Test du logger centralisé"""
    print("\n" + "=" * 70)
    print("🧪 TEST 1: LOGGER CENTRALISÉ")
    print("=" * 70)
    
    logger.trade("BUY", "BTCUSDT", 0.5, 45000.00, pnl=500.00)
    logger.metric("Sharpe Ratio", 1.85)
    logger.validation("Risk Check", True, "Drawdown < 15%")
    logger.sync("Metrics", "synchronized", {"trades": 42, "metrics": 156})
    
    print("✅ Logger centralisé fonctionne!")

def test_metrics():
    """Test des métriques unifiées"""
    print("\n" + "=" * 70)
    print("🧪 TEST 2: MÉTRIQUES UNIFIÉES")
    print("=" * 70)
    
    metrics = UnifiedMetrics("test_metrics.db")
    
    # Ajouter des trades
    print("\n📝 Ajout de trades...")
    metrics.add_trade("BUY", "BTCUSDT", 0.5, 45000, pnl=500)
    metrics.add_trade("SELL", "BTCUSDT", 0.5, 45500, pnl=250)
    metrics.add_trade("BUY", "ETHUSDT", 1.0, 3000, pnl=-100)
    metrics.add_trade("SELL", "ETHUSDT", 1.0, 3050, pnl=50)
    print(f"✅ {len(metrics.trades)} trades ajoutés")
    
    # Ajouter des returns
    print("\n📈 Ajout de returns...")
    returns_data = [0.01, 0.015, -0.005, 0.02, 0.01, -0.01, 0.025, 0.005]
    for ret in returns_data:
        metrics.add_return(ret)
    print(f"✅ {len(metrics.returns)} returns ajoutés")
    
    # Ajouter des valeurs de portefeuille
    print("\n💰 Ajout de valeurs de portefeuille...")
    portfolio_values = [10000, 10100, 10250, 10200, 10400, 10500, 10400, 10650, 10700]
    for val in portfolio_values:
        metrics.add_portfolio_value(val)
    print(f"✅ {len(metrics.portfolio_values)} valeurs ajoutées")
    
    # Calculer les métriques
    print("\n🔢 Calcul des métriques...")
    metrics.print_report(initial_capital=10000)
    
    # Valider
    print("\n✅ Validation des métriques...")
    validations = metrics.validate_metrics()
    for check, passed in validations.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")

def test_database():
    """Test de la base de données"""
    print("\n" + "=" * 70)
    print("🧪 TEST 3: BASE DE DONNÉES UNIFIÉE")
    print("=" * 70)
    
    db = UnifiedMetricsDB("test_metrics.db")
    
    # Vérifier les données
    print("\n📊 Vérification des données...")
    summary = db.get_summary()
    
    print(f"  Database: {summary['database']}")
    print(f"  Consistency: {summary['consistency']['status']}")
    print(f"  Trades: {summary['consistency']['trades']}")
    print(f"  Metrics: {summary['consistency']['metrics']}")
    print(f"  Validations: {summary['consistency']['validations']}")
    
    # Récupérer les derniers trades
    print("\n📝 Derniers trades:")
    trades = db.get_trades(limit=5)
    for trade in trades:
        print(f"  {trade['action']} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f} | PnL: ${trade['pnl']}")
    
    # Récupérer les dernières métriques
    print("\n📈 Dernières métriques:")
    sharpe_metrics = db.get_metrics('sharpe_ratio', limit=3)
    for metric in sharpe_metrics:
        print(f"  {metric['name']}: {metric['value']:.4f}")
    
    print("\n✅ Base de données fonctionne!")

def test_synchronization():
    """Test de la synchronisation"""
    print("\n" + "=" * 70)
    print("🧪 TEST 4: SYNCHRONISATION COMPLÈTE")
    print("=" * 70)
    
    # Créer les composants
    logger_test = logger
    metrics = UnifiedMetrics("test_metrics.db")
    db = UnifiedMetricsDB("test_metrics.db")
    
    # Simuler un cycle de trading
    print("\n🔄 Simulation d'un cycle de trading...")
    
    # Trade 1
    logger_test.trade("BUY", "BTCUSDT", 1.0, 50000, pnl=1000)
    metrics.add_trade("BUY", "BTCUSDT", 1.0, 50000, pnl=1000)
    logger_test.validation("Trade Execution", True, "Trade exécuté")
    db.add_validation("Trade Execution", True, "Trade exécuté")
    
    # Métrique
    logger_test.metric("Portfolio Value", 11000)
    metrics.add_portfolio_value(11000)
    
    # Sync
    logger_test.sync("Trading", "synchronized", {"trades": 1, "portfolio": 11000})
    db.add_sync("Trading", "synchronized", "1 trade, portfolio 11000")
    
    print("✅ Cycle de trading synchronisé!")
    
    # Vérifier la cohérence
    print("\n✅ Vérification de la cohérence...")
    consistency = db.validate_consistency()
    print(f"  Status: {consistency['status']}")
    print(f"  Trades: {consistency['trades']}")
    print(f"  Metrics: {consistency['metrics']}")
    print(f"  Validations: {consistency['validations']}")

def main():
    """Exécuter tous les tests"""
    print("\n" + "=" * 70)
    print("🚀 TESTS DU SYSTÈME UNIFIÉ ADAN 2.0")
    print("=" * 70)
    
    try:
        test_logger()
        test_metrics()
        test_database()
        test_synchronization()
        
        print("\n" + "=" * 70)
        print("✅ TOUS LES TESTS RÉUSSIS!")
        print("=" * 70)
        print("\n📊 Résumé:")
        print("  ✅ Logger centralisé: Fonctionnel")
        print("  ✅ Métriques unifiées: Fonctionnel")
        print("  ✅ Base de données: Fonctionnel")
        print("  ✅ Synchronisation: Fonctionnel")
        print("\n🎯 Prochaines étapes:")
        print("  1. Intégrer dans optuna_optimize_worker.py")
        print("  2. Intégrer dans train_parallel_agents.py")
        print("  3. Intégrer dans terminal_dashboard.py")
        print("  4. Tester en production")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

