#!/usr/bin/env python3
"""
Synchronise les métriques depuis les logs vers la base de données
"""
import sys
import re
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / "src"))

from adan_trading_bot.performance.unified_metrics_db import UnifiedMetricsDB

def sync_metrics_from_logs():
    """Extrait les métriques des logs et les ajoute à la base de données"""
    
    print("🔄 SYNCHRONISATION DES MÉTRIQUES DEPUIS LES LOGS")
    print("=" * 80)
    
    # Trouver le dernier fichier log
    log_dir = Path("/mnt/new_data/adan_logs")
    log_files = sorted(log_dir.glob("training_*.log"), reverse=True)
    
    if not log_files:
        print("❌ Aucun fichier log trouvé")
        return
    
    log_file = log_files[0]
    print(f"📝 Fichier log: {log_file.name}")
    
    # Initialiser la base de données
    db = UnifiedMetricsDB()
    
    # Patterns pour extraire les métriques
    patterns = {
        'reward': r'\[REWARD Worker \d+\] Base: ([\d.-]+), Freq: ([\d.-]+), PosLimit: ([\d.-]+), Outcome: ([\d.-]+), Duration: ([\d.-]+), InvalidTrade: ([\d.-]+), MultiHunt: ([\d.-]+), Total: ([\d.-]+)',
        'portfolio': r'\[STEP \d+\] Portfolio value: ([\d.]+)',
        'pnl': r'PnL: \$([+-]?[\d.]+)',
        'sharpe': r'Sharpe: ([\d.-]+)',
        'drawdown': r'Drawdown: ([\d.-]+)',
        'win_rate': r'Win Rate: ([\d.-]+)',
    }
    
    metrics_added = defaultdict(int)
    
    # Lire les dernières 50000 lignes du log
    try:
        with open(log_file, 'r', errors='ignore') as f:
            lines = f.readlines()[-50000:]
    except Exception as e:
        print(f"❌ Erreur lecture log: {e}")
        return
    
    print(f"📊 Traitement de {len(lines)} lignes...")
    
    for i, line in enumerate(lines):
        if i % 5000 == 0:
            print(f"  Progression: {i}/{len(lines)}")
        
        # Rewards
        match = re.search(patterns['reward'], line)
        if match:
            try:
                base_reward = float(match.group(1))
                freq_reward = float(match.group(2))
                outcome_reward = float(match.group(4))
                total_reward = float(match.group(8))
                
                db.add_metric("reward_base", base_reward, source="logs")
                db.add_metric("reward_freq", freq_reward, source="logs")
                db.add_metric("reward_outcome", outcome_reward, source="logs")
                db.add_metric("reward_total", total_reward, source="logs")
                
                metrics_added['rewards'] += 1
            except:
                pass
        
        # Portfolio value
        match = re.search(patterns['portfolio'], line)
        if match:
            try:
                portfolio = float(match.group(1))
                db.add_metric("portfolio_value", portfolio, source="logs")
                metrics_added['portfolio'] += 1
            except:
                pass
        
        # PnL
        match = re.search(patterns['pnl'], line)
        if match:
            try:
                pnl = float(match.group(1))
                db.add_metric("pnl", pnl, source="logs")
                metrics_added['pnl'] += 1
            except:
                pass
    
    print(f"\n✅ SYNCHRONISATION COMPLÈTE")
    print("=" * 80)
    print(f"Métriques ajoutées:")
    for metric_type, count in metrics_added.items():
        print(f"  {metric_type}: {count}")
    
    # Vérifier la cohérence
    consistency = db.validate_consistency()
    print(f"\n📊 Cohérence des données:")
    print(f"  Trades: {consistency.get('trades', 0)}")
    print(f"  Metrics: {consistency.get('metrics', 0)}")
    print(f"  Validations: {consistency.get('validations', 0)}")
    print(f"  Status: {consistency.get('status', 'Unknown')}")
    
    # Afficher les dernières métriques
    print(f"\n📈 Dernières métriques:")
    for metric_name in ['reward_total', 'portfolio_value', 'pnl']:
        latest = db.get_latest_metric(metric_name)
        if latest is not None:
            print(f"  {metric_name}: {latest:.4f}")

if __name__ == "__main__":
    sync_metrics_from_logs()
