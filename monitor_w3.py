#!/usr/bin/env python3
"""
Script de monitoring en temps réel pour Optuna W3
"""

import re
import time
import os
from datetime import datetime
from collections import defaultdict

def parse_log():
    """Parse le log et extrait les métriques"""
    if not os.path.exists('optuna_w3_final.log'):
        return None
    
    with open('optuna_w3_final.log', 'r') as f:
        content = f.read()
    
    metrics = {
        'trials_done': len(re.findall(r'Trial \d+ finished', content)),
        'best_trial': None,
        'best_score': None,
        'last_trial': None,
        'last_score': None,
        'trades': [],
        'win_rates': [],
        'pnls': [],
        'growths': [],
        'steps': 0,
        'it_s': 0,
    }
    
    # Meilleur trial
    best_match = re.search(r'Best is trial (\d+) with value: ([-\d.]+)', content)
    if best_match:
        metrics['best_trial'] = int(best_match.group(1))
        metrics['best_score'] = float(best_match.group(2))
    
    # Dernier trial
    last_matches = list(re.finditer(r'Trial (\d+) finished with value: ([-\d.]+)', content))
    if last_matches:
        last_match = last_matches[-1]
        metrics['last_trial'] = int(last_match.group(1))
        metrics['last_score'] = float(last_match.group(2))
    
    # Trades
    trades_matches = re.findall(r'Trades complétés: (\d+)', content)
    metrics['trades'] = [int(t) for t in trades_matches[-5:]]  # Derniers 5
    
    # Win rates
    wr_matches = re.findall(r'Win Rate: ([\d.]+)%', content)
    metrics['win_rates'] = [float(w) for w in wr_matches[-5:]]
    
    # PnLs
    pnl_matches = re.findall(r'PnL: \$([-\d.]+)', content)
    metrics['pnls'] = [float(p) for p in pnl_matches[-5:]]
    
    # Growth
    growth_matches = re.findall(r'Portfolio Growth: ([-\d.]+)%', content)
    metrics['growths'] = [float(g) for g in growth_matches[-5:]]
    
    # Vitesse (it/s)
    step_matches = list(re.finditer(r'\[STEP\] Starting step (\d+)', content))
    if len(step_matches) >= 2:
        first_step_match = step_matches[0]
        last_step_match = step_matches[-1]
        
        # Extraire les timestamps
        first_line_start = max(0, first_step_match.start() - 200)
        first_line = content[first_line_start:first_step_match.end()].split('\n')[0]
        first_time_match = re.search(r'(\d{2}):(\d{2}):(\d{2}),(\d+)', first_line)
        
        last_line_start = max(0, last_step_match.start() - 200)
        last_line = content[last_line_start:last_step_match.end()].split('\n')[0]
        last_time_match = re.search(r'(\d{2}):(\d{2}):(\d{2}),(\d+)', last_line)
        
        if first_time_match and last_time_match:
            first_sec = int(first_time_match.group(1)) * 3600 + int(first_time_match.group(2)) * 60 + int(first_time_match.group(3))
            last_sec = int(last_time_match.group(1)) * 3600 + int(last_time_match.group(2)) * 60 + int(last_time_match.group(3))
            
            elapsed = last_sec - first_sec
            if elapsed > 0:
                last_step_num = int(last_step_match.group(1))
                metrics['steps'] = last_step_num
                metrics['it_s'] = last_step_num / elapsed
    
    return metrics

def display_metrics(metrics):
    """Affiche les métriques"""
    os.system('clear')
    
    print("=" * 60)
    print(f"🔍 MONITORING OPTUNA W3 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    if not metrics:
        print("⏳ En attente de données...")
        return
    
    # Trials
    print(f"\n📊 TRIALS")
    print(f"   Terminés: {metrics['trials_done']}/20")
    if metrics['best_trial'] is not None:
        print(f"   ⭐ Meilleur: Trial {metrics['best_trial']} (Score: {metrics['best_score']:.4f})")
    if metrics['last_trial'] is not None:
        print(f"   🔄 Dernier: Trial {metrics['last_trial']} (Score: {metrics['last_score']:.4f})")
    
    # Métriques du dernier trial
    print(f"\n📈 MÉTRIQUES (derniers 5 trials)")
    if metrics['trades']:
        avg_trades = sum(metrics['trades']) / len(metrics['trades'])
        print(f"   Trades: {metrics['trades'][-1]} (avg: {avg_trades:.1f})")
    if metrics['win_rates']:
        avg_wr = sum(metrics['win_rates']) / len(metrics['win_rates'])
        print(f"   Win Rate: {metrics['win_rates'][-1]:.1f}% (avg: {avg_wr:.1f}%)")
    if metrics['pnls']:
        avg_pnl = sum(metrics['pnls']) / len(metrics['pnls'])
        print(f"   PnL: ${metrics['pnls'][-1]:.2f} (avg: ${avg_pnl:.2f})")
    if metrics['growths']:
        avg_growth = sum(metrics['growths']) / len(metrics['growths'])
        print(f"   Growth: {metrics['growths'][-1]:.2f}% (avg: {avg_growth:.2f}%)")
    
    # Vitesse
    print(f"\n⚡ VITESSE")
    print(f"   it/s: {metrics['it_s']:.2f}")
    print(f"   Steps: {metrics['steps']}")
    if metrics['it_s'] > 0:
        time_for_1000 = 1000 / metrics['it_s']
        print(f"   Temps pour 1000 steps: {time_for_1000:.0f}s ({time_for_1000/60:.1f}min)")
    
    print("\n" + "=" * 60)
    print("🔄 Mise à jour toutes les 10 secondes (Ctrl+C pour arrêter)")
    print("=" * 60)

if __name__ == '__main__':
    try:
        while True:
            metrics = parse_log()
            display_metrics(metrics)
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring arrêté")
