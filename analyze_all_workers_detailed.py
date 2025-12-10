#!/usr/bin/env python3
"""
Analyse détaillée des 4 workers depuis les logs
"""
import re
from pathlib import Path
from collections import defaultdict

def analyze_all_workers():
    """Analyse chaque worker séparément"""
    
    # Trouver le dernier fichier log
    log_dir = Path("/mnt/new_data/adan_logs")
    log_files = sorted(log_dir.glob("training_*.log"), reverse=True)
    
    if not log_files:
        print("❌ Aucun fichier log trouvé")
        return
    
    log_file = log_files[0]
    print(f"📊 ANALYSE DÉTAILLÉE DES 4 WORKERS")
    print(f"📝 Fichier: {log_file.name}")
    print("=" * 100)
    
    # Patterns
    patterns = {
        'step': r'\[STEP (\d+)\].*Portfolio value: ([\d.]+)',
        'reward': r'\[REWARD Worker (\d+)\].*Total: ([\d.-]+)',
        'position_open': r'\[POSITION OUVERTE\].*BTCUSDT: ([\d.]+).*SL: ([\d.]+)%.*TP: ([\d.]+)%',
        'position_close': r'\[POSITION FERMÉE\].*PnL: \$([+-]?[\d.]+)',
        'trade_count': r"Counts: {'5m': (\d+), '1h': (\d+), '4h': (\d+), 'daily_total': (\d+)}",
    }
    
    # Stats par worker
    workers = {
        'w0': {'steps': [], 'portfolio': [], 'rewards': [], 'positions': 0, 'pnl': []},
        'w1': {'steps': [], 'portfolio': [], 'rewards': [], 'positions': 0, 'pnl': []},
        'w2': {'steps': [], 'portfolio': [], 'rewards': [], 'positions': 0, 'pnl': []},
        'w3': {'steps': [], 'portfolio': [], 'rewards': [], 'positions': 0, 'pnl': []},
    }
    
    # Lire les dernières 20000 lignes
    try:
        with open(log_file, 'r', errors='ignore') as f:
            lines = f.readlines()[-20000:]
    except Exception as e:
        print(f"❌ Erreur lecture: {e}")
        return
    
    current_worker = 'w0'
    
    for line in lines:
        # Identifier le worker
        if 'worker_w' in line.lower():
            match = re.search(r'worker_(w\d)', line.lower())
            if match:
                current_worker = match.group(1)
        
        # Step et portfolio
        match = re.search(patterns['step'], line)
        if match:
            try:
                step = int(match.group(1))
                portfolio = float(match.group(2))
                workers[current_worker]['steps'].append(step)
                workers[current_worker]['portfolio'].append(portfolio)
            except:
                pass
        
        # Rewards
        match = re.search(patterns['reward'], line)
        if match:
            try:
                worker_id = f"w{match.group(1)}"
                reward = float(match.group(2))
                workers[worker_id]['rewards'].append(reward)
            except:
                pass
        
        # Positions
        if '[POSITION OUVERTE]' in line:
            workers[current_worker]['positions'] += 1
        
        # PnL
        match = re.search(patterns['position_close'], line)
        if match:
            try:
                pnl = float(match.group(1))
                workers[current_worker]['pnl'].append(pnl)
            except:
                pass
    
    # Afficher les résultats
    for worker_id in ['w0', 'w1', 'w2', 'w3']:
        stats = workers[worker_id]
        
        print(f"\n🤖 WORKER {worker_id.upper()}")
        print("-" * 100)
        
        if not stats['portfolio']:
            print(f"  ⚠️  Pas de données pour ce worker")
            continue
        
        # Portfolio
        portfolio_initial = stats['portfolio'][0]
        portfolio_current = stats['portfolio'][-1]
        portfolio_change = ((portfolio_current - portfolio_initial) / portfolio_initial * 100) if portfolio_initial > 0 else 0
        portfolio_min = min(stats['portfolio'])
        portfolio_max = max(stats['portfolio'])
        
        print(f"  💰 Portfolio:")
        print(f"    Initial: ${portfolio_initial:.2f}")
        print(f"    Current: ${portfolio_current:.2f}")
        print(f"    Change: {portfolio_change:+.2f}%")
        print(f"    Min: ${portfolio_min:.2f}")
        print(f"    Max: ${portfolio_max:.2f}")
        
        # Rewards
        if stats['rewards']:
            avg_reward = sum(stats['rewards']) / len(stats['rewards'])
            max_reward = max(stats['rewards'])
            min_reward = min(stats['rewards'])
            
            print(f"  🎯 Rewards:")
            print(f"    Average: {avg_reward:.4f}")
            print(f"    Max: {max_reward:.4f}")
            print(f"    Min: {min_reward:.4f}")
            print(f"    Count: {len(stats['rewards'])}")
        
        # Trading
        total_pnl = sum(stats['pnl']) if stats['pnl'] else 0
        avg_pnl = total_pnl / len(stats['pnl']) if stats['pnl'] else 0
        
        print(f"  📈 Trading:")
        print(f"    Positions Opened: {stats['positions']}")
        print(f"    Closed Positions: {len(stats['pnl'])}")
        print(f"    Total PnL: ${total_pnl:+.2f}")
        print(f"    Avg PnL per Trade: ${avg_pnl:+.2f}")
        
        # Progress
        if stats['steps']:
            max_step = max(stats['steps'])
            print(f"  ⏱️  Progress:")
            print(f"    Max Step: {max_step}")
            print(f"    Estimated Progress: {max_step / 500000 * 100:.2f}%")
    
    # Résumé global
    print("\n" + "=" * 100)
    print("📊 RÉSUMÉ GLOBAL")
    print("=" * 100)
    
    total_portfolio = sum(w['portfolio'][-1] if w['portfolio'] else 0 for w in workers.values())
    total_pnl = sum(sum(w['pnl']) for w in workers.values())
    total_positions = sum(w['positions'] for w in workers.values())
    avg_step = sum(max(w['steps']) if w['steps'] else 0 for w in workers.values()) / 4
    
    print(f"Total Portfolio Value: ${total_portfolio:.2f}")
    print(f"Total PnL: ${total_pnl:+.2f}")
    print(f"Total Positions Opened: {total_positions}")
    print(f"Average Steps per Worker: {avg_step:.0f}")
    print(f"Overall Progress: {avg_step / 500000 * 100:.2f}%")
    
    # Ranking
    print(f"\n🏆 RANKING DES WORKERS")
    print("-" * 100)
    
    rankings = []
    for worker_id in ['w0', 'w1', 'w2', 'w3']:
        stats = workers[worker_id]
        if stats['portfolio']:
            portfolio_change = ((stats['portfolio'][-1] - stats['portfolio'][0]) / stats['portfolio'][0] * 100)
            total_pnl = sum(stats['pnl']) if stats['pnl'] else 0
            rankings.append((worker_id, portfolio_change, total_pnl))
    
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    for i, (worker_id, change, pnl) in enumerate(rankings, 1):
        print(f"{i}. {worker_id.upper()}: {change:+.2f}% | PnL: ${pnl:+.2f}")

if __name__ == "__main__":
    analyze_all_workers()
