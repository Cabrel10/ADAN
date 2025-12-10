#!/usr/bin/env python3
"""
Analyse les performances des workers en temps réel depuis les logs
"""
import re
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def analyze_training_logs():
    """Analyse les logs d'entraînement pour extraire les performances"""
    
    # Trouver le dernier fichier log
    log_dir = Path("/mnt/new_data/adan_logs")
    log_files = sorted(log_dir.glob("training_*.log"), reverse=True)
    
    if not log_files:
        print("❌ Aucun fichier log trouvé")
        return
    
    log_file = log_files[0]
    print(f"📊 Analyse du fichier: {log_file.name}")
    print("=" * 80)
    
    # Patterns pour extraire les données
    patterns = {
        'portfolio_value': r'\[STEP \d+\] Portfolio value: ([\d.]+)',
        'reward': r'\[REWARD Worker \d+\] Base: ([\d.-]+), Freq: ([\d.-]+), PosLimit: ([\d.-]+), Outcome: ([\d.-]+), Duration: ([\d.-]+), InvalidTrade: ([\d.-]+), MultiHunt: ([\d.-]+), Total: ([\d.-]+)',
        'trades': r'Counts: {\'5m\': (\d+), \'1h\': (\d+), \'4h\': (\d+), \'daily_total\': (\d+)}',
        'pnl': r'PnL: \$([+-]?[\d.]+)',
        'position': r'\[POSITION (OUVERTE|FERMÉE)\]',
        'step': r'\[STEP (\d+)',
    }
    
    # Statistiques par worker
    worker_stats = defaultdict(lambda: {
        'portfolio_values': [],
        'rewards': [],
        'trades_count': 0,
        'pnl_total': 0.0,
        'positions_opened': 0,
        'positions_closed': 0,
        'steps': 0,
    })
    
    # Lire les dernières 10000 lignes du log
    try:
        with open(log_file, 'r', errors='ignore') as f:
            lines = f.readlines()[-10000:]
    except Exception as e:
        print(f"❌ Erreur lecture log: {e}")
        return
    
    current_worker = 'w0'
    
    for line in lines:
        # Identifier le worker
        if 'worker_w' in line.lower():
            match = re.search(r'worker_(w\d)', line.lower())
            if match:
                current_worker = match.group(1)
        
        # Portfolio value
        match = re.search(patterns['portfolio_value'], line)
        if match:
            try:
                value = float(match.group(1))
                worker_stats[current_worker]['portfolio_values'].append(value)
            except:
                pass
        
        # Rewards
        match = re.search(patterns['reward'], line)
        if match:
            try:
                total_reward = float(match.group(8))
                worker_stats[current_worker]['rewards'].append(total_reward)
            except:
                pass
        
        # Trades
        match = re.search(patterns['trades'], line)
        if match:
            try:
                trades_5m = int(match.group(1))
                trades_1h = int(match.group(2))
                trades_4h = int(match.group(3))
                daily_total = int(match.group(4))
                worker_stats[current_worker]['trades_count'] = daily_total
            except:
                pass
        
        # PnL
        match = re.search(patterns['pnl'], line)
        if match:
            try:
                pnl = float(match.group(1))
                worker_stats[current_worker]['pnl_total'] += pnl
            except:
                pass
        
        # Positions
        if '[POSITION OUVERTE]' in line:
            worker_stats[current_worker]['positions_opened'] += 1
        elif '[POSITION FERMÉE]' in line:
            worker_stats[current_worker]['positions_closed'] += 1
        
        # Steps
        match = re.search(patterns['step'], line)
        if match:
            try:
                step = int(match.group(1))
                worker_stats[current_worker]['steps'] = max(worker_stats[current_worker]['steps'], step)
            except:
                pass
    
    # Afficher les résultats
    print("\n📈 PERFORMANCES DES WORKERS")
    print("=" * 80)
    
    for worker_id in sorted(worker_stats.keys()):
        stats = worker_stats[worker_id]
        
        if not stats['portfolio_values']:
            continue
        
        portfolio_current = stats['portfolio_values'][-1] if stats['portfolio_values'] else 0
        portfolio_initial = stats['portfolio_values'][0] if stats['portfolio_values'] else 20.50
        portfolio_change = ((portfolio_current - portfolio_initial) / portfolio_initial * 100) if portfolio_initial > 0 else 0
        
        avg_reward = sum(stats['rewards']) / len(stats['rewards']) if stats['rewards'] else 0
        max_reward = max(stats['rewards']) if stats['rewards'] else 0
        min_reward = min(stats['rewards']) if stats['rewards'] else 0
        
        print(f"\n🤖 {worker_id.upper()}")
        print("-" * 80)
        print(f"  Portfolio:")
        print(f"    Initial: ${portfolio_initial:.2f}")
        print(f"    Current: ${portfolio_current:.2f}")
        print(f"    Change: {portfolio_change:+.2f}%")
        print(f"  Rewards:")
        print(f"    Average: {avg_reward:.4f}")
        print(f"    Max: {max_reward:.4f}")
        print(f"    Min: {min_reward:.4f}")
        print(f"  Trading:")
        print(f"    Total Trades: {stats['trades_count']}")
        print(f"    Positions Opened: {stats['positions_opened']}")
        print(f"    Positions Closed: {stats['positions_closed']}")
        print(f"    Total PnL: ${stats['pnl_total']:+.2f}")
        print(f"  Progress:")
        print(f"    Steps: {stats['steps']}")
        print(f"    Estimated Progress: {stats['steps'] / 500000 * 100:.2f}%")
    
    # Résumé global
    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ GLOBAL")
    print("=" * 80)
    
    total_portfolio = sum(s['portfolio_values'][-1] if s['portfolio_values'] else 0 for s in worker_stats.values())
    total_pnl = sum(s['pnl_total'] for s in worker_stats.values())
    total_trades = sum(s['trades_count'] for s in worker_stats.values())
    avg_steps = sum(s['steps'] for s in worker_stats.values()) / len(worker_stats) if worker_stats else 0
    
    print(f"Total Portfolio Value: ${total_portfolio:.2f}")
    print(f"Total PnL: ${total_pnl:+.2f}")
    print(f"Total Trades: {total_trades}")
    print(f"Average Steps per Worker: {avg_steps:.0f}")
    print(f"Overall Progress: {avg_steps / 500000 * 100:.2f}%")
    
    # Estimation du temps restant
    if avg_steps > 0:
        steps_per_second = avg_steps / (len(lines) / 100)  # Approximation
        remaining_steps = 500000 - avg_steps
        remaining_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else 0
        remaining_hours = remaining_seconds / 3600
        print(f"Estimated Time Remaining: {remaining_hours:.1f} hours")

if __name__ == "__main__":
    analyze_training_logs()
