#!/usr/bin/env python3
"""Analyse les métriques de performance de chaque worker"""

import re
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

LOG_FILE = "/mnt/new_data/adan_logs/training_final_1765088250.log"

def parse_metrics_from_logs():
    """Extrait les métriques de chaque worker des logs"""
    
    workers_data = defaultdict(lambda: {
        'steps': [],
        'portfolio_values': [],
        'trades': [],
        'pnl_values': [],
        'sharpe_values': [],
        'win_rates': [],
        'drawdowns': [],
        'timestamps': []
    })
    
    print("🔍 ANALYSE DES MÉTRIQUES PAR WORKER")
    print("=" * 80)
    
    if not Path(LOG_FILE).exists():
        print(f"❌ Fichier log non trouvé: {LOG_FILE}")
        return
    
    try:
        with open(LOG_FILE, 'r', errors='ignore') as f:
            for line in f:
                # Extraire le worker ID
                worker_match = re.search(r'\[Worker (\d+)\]', line)
                if not worker_match:
                    worker_match = re.search(r'Worker=w(\d+)', line)
                    if worker_match:
                        worker_id = f"w{worker_match.group(1)}"
                    else:
                        continue
                else:
                    worker_id = f"w{worker_match.group(1)}"
                
                # Extraire le timestamp
                ts_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if ts_match:
                    workers_data[worker_id]['timestamps'].append(ts_match.group(1))
                
                # Extraire les steps
                step_match = re.search(r'\[STEP (\d+)\]', line)
                if step_match:
                    workers_data[worker_id]['steps'].append(int(step_match.group(1)))
                
                # Extraire les valeurs de portfolio
                pv_match = re.search(r'Portfolio value: ([\d.]+)', line)
                if pv_match:
                    workers_data[worker_id]['portfolio_values'].append(float(pv_match.group(1)))
                
                # Extraire les PnL
                pnl_match = re.search(r'pnl[:\s]+([0-9.-]+)', line, re.IGNORECASE)
                if pnl_match:
                    try:
                        workers_data[worker_id]['pnl_values'].append(float(pnl_match.group(1)))
                    except:
                        pass
                
                # Extraire les trades
                if '[TRADE]' in line:
                    workers_data[worker_id]['trades'].append(line.strip()[:100])
                
                # Extraire Sharpe
                sharpe_match = re.search(r'sharpe[:\s]+([0-9.-]+)', line, re.IGNORECASE)
                if sharpe_match:
                    try:
                        workers_data[worker_id]['sharpe_values'].append(float(sharpe_match.group(1)))
                    except:
                        pass
                
                # Extraire Win Rate
                wr_match = re.search(r'win_rate[:\s]+([0-9.-]+)', line, re.IGNORECASE)
                if wr_match:
                    try:
                        workers_data[worker_id]['win_rates'].append(float(wr_match.group(1)))
                    except:
                        pass
                
                # Extraire Drawdown
                dd_match = re.search(r'drawdown[:\s]+([0-9.-]+)', line, re.IGNORECASE)
                if dd_match:
                    try:
                        workers_data[worker_id]['drawdowns'].append(float(dd_match.group(1)))
                    except:
                        pass
    
    except Exception as e:
        print(f"❌ Erreur lors de la lecture: {e}")
        return
    
    # Affichage des résultats
    print(f"\n📊 RÉSUMÉ PAR WORKER\n")
    
    for worker_id in sorted(workers_data.keys()):
        data = workers_data[worker_id]
        print(f"\n{'='*80}")
        print(f"👷 {worker_id.upper()}")
        print(f"{'='*80}")
        
        if data['steps']:
            print(f"  📈 Steps:")
            print(f"     Min: {min(data['steps'])}")
            print(f"     Max: {max(data['steps'])}")
            print(f"     Moyenne: {sum(data['steps'])/len(data['steps']):.0f}")
            print(f"     Total: {len(data['steps'])} mesures")
        
        if data['portfolio_values']:
            print(f"\n  💰 Portfolio Value:")
            print(f"     Min: ${min(data['portfolio_values']):.2f}")
            print(f"     Max: ${max(data['portfolio_values']):.2f}")
            print(f"     Moyenne: ${sum(data['portfolio_values'])/len(data['portfolio_values']):.2f}")
            print(f"     Dernier: ${data['portfolio_values'][-1]:.2f}")
        
        if data['pnl_values']:
            print(f"\n  💵 PnL:")
            print(f"     Min: {min(data['pnl_values']):.4f}")
            print(f"     Max: {max(data['pnl_values']):.4f}")
            print(f"     Moyenne: {sum(data['pnl_values'])/len(data['pnl_values']):.4f}")
            print(f"     Positifs: {len([x for x in data['pnl_values'] if x > 0])}/{len(data['pnl_values'])}")
        
        if data['trades']:
            print(f"\n  📊 Trades:")
            print(f"     Total: {len(data['trades'])}")
            print(f"     Derniers:")
            for trade in data['trades'][-3:]:
                print(f"       - {trade}")
        
        if data['sharpe_values']:
            print(f"\n  📈 Sharpe Ratio:")
            print(f"     Min: {min(data['sharpe_values']):.4f}")
            print(f"     Max: {max(data['sharpe_values']):.4f}")
            print(f"     Moyenne: {sum(data['sharpe_values'])/len(data['sharpe_values']):.4f}")
        
        if data['win_rates']:
            print(f"\n  🎯 Win Rate:")
            print(f"     Min: {min(data['win_rates']):.2%}")
            print(f"     Max: {max(data['win_rates']):.2%}")
            print(f"     Moyenne: {sum(data['win_rates'])/len(data['win_rates']):.2%}")
        
        if data['drawdowns']:
            print(f"\n  📉 Drawdown:")
            print(f"     Min: {min(data['drawdowns']):.4f}")
            print(f"     Max: {max(data['drawdowns']):.4f}")
            print(f"     Moyenne: {sum(data['drawdowns'])/len(data['drawdowns']):.4f}")
        
        if data['timestamps']:
            print(f"\n  ⏱️  Temps:")
            print(f"     Premier: {data['timestamps'][0]}")
            print(f"     Dernier: {data['timestamps'][-1]}")
    
    # Comparaison globale
    print(f"\n\n{'='*80}")
    print(f"📊 COMPARAISON GLOBALE")
    print(f"{'='*80}\n")
    
    all_pv = []
    all_pnl = []
    all_trades = 0
    
    for worker_id, data in workers_data.items():
        if data['portfolio_values']:
            all_pv.extend(data['portfolio_values'])
        if data['pnl_values']:
            all_pnl.extend(data['pnl_values'])
        all_trades += len(data['trades'])
    
    if all_pv:
        print(f"💰 Portfolio Value Global:")
        print(f"   Min: ${min(all_pv):.2f}")
        print(f"   Max: ${max(all_pv):.2f}")
        print(f"   Moyenne: ${sum(all_pv)/len(all_pv):.2f}")
    
    if all_pnl:
        print(f"\n💵 PnL Global:")
        print(f"   Min: {min(all_pnl):.4f}")
        print(f"   Max: {max(all_pnl):.4f}")
        print(f"   Moyenne: {sum(all_pnl)/len(all_pnl):.4f}")
        print(f"   Positifs: {len([x for x in all_pnl if x > 0])}/{len(all_pnl)}")
    
    print(f"\n📊 Trades Global: {all_trades}")
    print(f"Workers actifs: {len(workers_data)}")

if __name__ == '__main__':
    parse_metrics_from_logs()
