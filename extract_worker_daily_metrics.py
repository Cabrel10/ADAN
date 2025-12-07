#!/usr/bin/env python3
"""Extrait les métriques journalières de chaque worker"""

import re
from collections import defaultdict
from pathlib import Path

LOG_FILE = "/mnt/new_data/adan_logs/training_final_1765088250.log"

def extract_metrics():
    """Extrait les métriques de chaque worker"""
    
    workers = defaultdict(lambda: {
        'portfolio_values': [],
        'trades_count': 0,
        'steps': [],
        'first_time': None,
        'last_time': None,
        'risk_updates': []
    })
    
    print("📊 EXTRACTION DES MÉTRIQUES PAR WORKER")
    print("=" * 80)
    
    if not Path(LOG_FILE).exists():
        print(f"❌ Fichier log non trouvé: {LOG_FILE}")
        return
    
    try:
        with open(LOG_FILE, 'r', errors='ignore') as f:
            for line in f:
                # Extraire timestamp
                ts_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                timestamp = ts_match.group(1) if ts_match else None
                
                # Extraire Worker ID
                worker_match = re.search(r'\[Worker (\d+)\]', line)
                if not worker_match:
                    worker_match = re.search(r'Worker=w(\d+)', line)
                    if worker_match:
                        worker_id = f"w{worker_match.group(1)}"
                    else:
                        continue
                else:
                    worker_id = f"w{worker_match.group(1)}"
                
                # Initialiser first_time
                if workers[worker_id]['first_time'] is None and timestamp:
                    workers[worker_id]['first_time'] = timestamp
                if timestamp:
                    workers[worker_id]['last_time'] = timestamp
                
                # Portfolio value
                if 'Portfolio value:' in line:
                    pv_match = re.search(r'Portfolio value: ([\d.]+)', line)
                    if pv_match:
                        workers[worker_id]['portfolio_values'].append(float(pv_match.group(1)))
                
                # Trades
                if '[TRADE]' in line:
                    workers[worker_id]['trades_count'] += 1
                
                # Steps
                if '[STEP' in line:
                    step_match = re.search(r'\[STEP (\d+)\]', line)
                    if step_match:
                        workers[worker_id]['steps'].append(int(step_match.group(1)))
                
                # Risk updates
                if '[RISK_UPDATE]' in line:
                    workers[worker_id]['risk_updates'].append(line.strip()[:100])
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return
    
    # Affichage
    print(f"\n📈 RÉSUMÉ PAR WORKER\n")
    
    for worker_id in sorted(workers.keys()):
        data = workers[worker_id]
        print(f"\n{'='*80}")
        print(f"👷 {worker_id.upper()}")
        print(f"{'='*80}")
        
        print(f"  ⏱️  Temps:")
        print(f"     Début: {data['first_time']}")
        print(f"     Fin: {data['last_time']}")
        
        if data['steps']:
            print(f"\n  📊 Steps:")
            print(f"     Min: {min(data['steps'])}")
            print(f"     Max: {max(data['steps'])}")
            print(f"     Moyenne: {sum(data['steps'])/len(data['steps']):.0f}")
            print(f"     Total mesures: {len(data['steps'])}")
        
        if data['portfolio_values']:
            pv_min = min(data['portfolio_values'])
            pv_max = max(data['portfolio_values'])
            pv_avg = sum(data['portfolio_values']) / len(data['portfolio_values'])
            pv_last = data['portfolio_values'][-1]
            pv_first = data['portfolio_values'][0]
            pv_change = ((pv_last - pv_first) / pv_first * 100) if pv_first > 0 else 0
            
            print(f"\n  💰 Portfolio Value:")
            print(f"     Min: ${pv_min:.2f}")
            print(f"     Max: ${pv_max:.2f}")
            print(f"     Moyenne: ${pv_avg:.2f}")
            print(f"     Dernier: ${pv_last:.2f}")
            print(f"     Premier: ${pv_first:.2f}")
            print(f"     Changement: {pv_change:+.2f}%")
        
        print(f"\n  📊 Trades:")
        print(f"     Total: {data['trades_count']}")
        
        if data['risk_updates']:
            print(f"\n  ⚙️  Risk Updates:")
            print(f"     Total: {len(data['risk_updates'])}")
            print(f"     Dernier:")
            print(f"       {data['risk_updates'][-1]}")
    
    # Comparaison globale
    print(f"\n\n{'='*80}")
    print(f"📊 COMPARAISON GLOBALE")
    print(f"{'='*80}\n")
    
    total_trades = sum(w['trades_count'] for w in workers.values())
    total_steps = sum(len(w['steps']) for w in workers.values())
    all_pv = []
    
    for w in workers.values():
        all_pv.extend(w['portfolio_values'])
    
    print(f"👷 Workers actifs: {len(workers)}")
    print(f"📊 Total trades: {total_trades}")
    print(f"📈 Total steps mesurés: {total_steps}")
    
    if all_pv:
        print(f"\n💰 Portfolio Value Global:")
        print(f"   Min: ${min(all_pv):.2f}")
        print(f"   Max: ${max(all_pv):.2f}")
        print(f"   Moyenne: ${sum(all_pv)/len(all_pv):.2f}")
    
    # Comparaison par worker
    print(f"\n\n{'='*80}")
    print(f"📊 COMPARAISON DÉTAILLÉE")
    print(f"{'='*80}\n")
    
    print(f"{'Worker':<10} {'Trades':<10} {'Steps':<10} {'PV Min':<12} {'PV Max':<12} {'PV Change':<12}")
    print(f"{'-'*80}")
    
    for worker_id in sorted(workers.keys()):
        data = workers[worker_id]
        trades = data['trades_count']
        steps = len(data['steps'])
        
        if data['portfolio_values']:
            pv_min = min(data['portfolio_values'])
            pv_max = max(data['portfolio_values'])
            pv_first = data['portfolio_values'][0]
            pv_last = data['portfolio_values'][-1]
            pv_change = ((pv_last - pv_first) / pv_first * 100) if pv_first > 0 else 0
            
            print(f"{worker_id:<10} {trades:<10} {steps:<10} ${pv_min:<11.2f} ${pv_max:<11.2f} {pv_change:+.2f}%")
        else:
            print(f"{worker_id:<10} {trades:<10} {steps:<10} N/A         N/A         N/A")

if __name__ == '__main__':
    extract_metrics()
