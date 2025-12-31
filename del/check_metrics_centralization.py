#!/usr/bin/env python3
"""
Vérifie l'état de la centralisation des métriques
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from adan_trading_bot.performance.unified_metrics_db import UnifiedMetricsDB
from adan_trading_bot.common.central_logger import logger as central_logger

def check_metrics():
    """Vérifie l'état des métriques centralisées"""
    
    print("🔍 VÉRIFICATION DE LA CENTRALISATION DES MÉTRIQUES")
    print("=" * 80)
    
    # Vérifier la base de données
    try:
        db = UnifiedMetricsDB()
        print("✅ Base de données UnifiedMetricsDB accessible")
    except Exception as e:
        print(f"❌ Erreur accès UnifiedMetricsDB: {e}")
        return
    
    # Vérifier la cohérence
    try:
        consistency = db.validate_consistency()
        print(f"\n📊 Cohérence des données:")
        print(f"  Trades: {consistency.get('trades', 0)}")
        print(f"  Metrics: {consistency.get('metrics', 0)}")
        print(f"  Validations: {consistency.get('validations', 0)}")
        print(f"  Status: {consistency.get('status', 'Unknown')}")
    except Exception as e:
        print(f"❌ Erreur validation cohérence: {e}")
    
    # Vérifier les dernières métriques
    print(f"\n📈 Dernières métriques:")
    
    metric_names = ['sharpe_ratio', 'max_drawdown', 'win_rate', 'daily_profit', 'total_pnl']
    
    for metric_name in metric_names:
        try:
            latest = db.get_latest_metric(metric_name)
            if latest is not None:
                print(f"  {metric_name}: {latest:.4f}")
            else:
                print(f"  {metric_name}: ❌ Pas de données")
        except Exception as e:
            print(f"  {metric_name}: ❌ Erreur - {e}")
    
    # Vérifier les trades
    print(f"\n💰 Trades:")
    try:
        trades = db.get_trades(limit=5)
        print(f"  Total trades: {db.get_trade_count()}")
        if trades:
            print(f"  Derniers trades:")
            for trade in trades[:3]:
                print(f"    - {trade['action']} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f} | PnL: ${trade['pnl']}")
        else:
            print(f"  ❌ Aucun trade enregistré")
    except Exception as e:
        print(f"  ❌ Erreur lecture trades: {e}")
    
    # Vérifier les validations
    print(f"\n✅ Validations:")
    try:
        validations = db.get_validations(limit=5)
        if validations:
            print(f"  Dernières validations:")
            for val in validations[:3]:
                status = "✅ PASS" if val['passed'] else "❌ FAIL"
                print(f"    - {val['check_name']}: {status}")
        else:
            print(f"  ❌ Aucune validation enregistrée")
    except Exception as e:
        print(f"  ❌ Erreur lecture validations: {e}")
    
    # Vérifier le CentralLogger
    print(f"\n📝 CentralLogger:")
    try:
        print(f"  Instance: {central_logger}")
        print(f"  Type: {type(central_logger).__name__}")
        print(f"  ✅ CentralLogger accessible")
    except Exception as e:
        print(f"  ❌ Erreur CentralLogger: {e}")
    
    # Résumé
    print("\n" + "=" * 80)
    print("📋 RÉSUMÉ")
    print("=" * 80)
    
    try:
        summary = db.get_summary()
        print(f"Database: {summary.get('database', 'Unknown')}")
        print(f"Consistency: {summary.get('consistency', {}).get('status', 'Unknown')}")
        print(f"Recent Trades: {summary.get('recent_trades', 0)}")
        print(f"Recent Validations: {summary.get('recent_validations', 0)}")
    except Exception as e:
        print(f"❌ Erreur résumé: {e}")

if __name__ == "__main__":
    check_metrics()
