#!/usr/bin/env python3
"""Analyser les performances de l'entraînement"""

import re
from collections import defaultdict
from pathlib import Path

log_file = Path("/mnt/new_data/adan_logs/training_20251207_200003.log")

# Dictionnaires pour stocker les données
worker_portfolios = defaultdict(list)
worker_trades = defaultdict(int)
worker_steps = defaultdict(int)
worker_pnl = defaultdict(float)

print("🔍 ANALYSE DE L'ENTRAÎNEMENT EN COURS...")
print("=" * 70)

# Lire le log
with open(log_file, 'r') as f:
    for line in f:
        # Extraire Portfolio value
        if "Portfolio value:" in line:
            match = re.search(r'Portfolio value: ([\d.]+)', line)
            if match:
                value = float(match.group(1))
                # Déterminer le worker (Worker 0, 1, 2, 3)
                if "Worker 0" in line or "[Worker 0]" in line:
                    worker_portfolios[0].append(value)
                elif "Worker 1" in line or "[Worker 1]" in line:
                    worker_portfolios[1].append(value)
                elif "Worker 2" in line or "[Worker 2]" in line:
                    worker_portfolios[2].append(value)
                elif "Worker 3" in line or "[Worker 3]" in line:
                    worker_portfolios[3].append(value)
        
        # Compter les trades
        if "POSITION OUVERTE" in line:
            if "Worker 0" in line:
                worker_trades[0] += 1
            elif "Worker 1" in line:
                worker_trades[1] += 1
            elif "Worker 2" in line:
                worker_trades[2] += 1
            elif "Worker 3" in line:
                worker_trades[3] += 1
        
        # Extraire PnL
        if "PnL:" in line and "POSITION FERMÉE" in line:
            match = re.search(r'PnL: \$([+-][\d.]+)', line)
            if match:
                pnl = float(match.group(1))
                if "Worker 0" in line:
                    worker_pnl[0] += pnl
                elif "Worker 1" in line:
                    worker_pnl[1] += pnl
                elif "Worker 2" in line:
                    worker_pnl[2] += pnl
                elif "Worker 3" in line:
                    worker_pnl[3] += pnl

# Afficher les résultats
print("\n📊 PERFORMANCES PAR WORKER:\n")

for worker_id in range(4):
    print(f"{'='*70}")
    print(f"🤖 WORKER {worker_id} (w{worker_id}):")
    print(f"{'='*70}")
    
    if worker_id in worker_portfolios and worker_portfolios[worker_id]:
        portfolios = worker_portfolios[worker_id]
        initial = 20.50
        final = portfolios[-1]
        max_val = max(portfolios)
        min_val = min(portfolios)
        
        profit = final - initial
        profit_pct = (profit / initial) * 100
        
        print(f"  Portfolio Initial:     ${initial:.2f}")
        print(f"  Portfolio Final:       ${final:.2f}")
        print(f"  Portfolio Max:         ${max_val:.2f}")
        print(f"  Portfolio Min:         ${min_val:.2f}")
        print(f"  Profit/Loss:           ${profit:+.2f} ({profit_pct:+.1f}%)")
        print(f"  Trades Ouvertes:       {worker_trades.get(worker_id, 0)}")
        print(f"  PnL Total:             ${worker_pnl.get(worker_id, 0):+.2f}")
        print(f"  Observations:          {len(portfolios)}")
    else:
        print(f"  ❌ Pas de données pour ce worker")
    print()

print("=" * 70)
print("\n🏆 CLASSEMENT (par Portfolio Final):\n")

rankings = []
for worker_id in range(4):
    if worker_id in worker_portfolios and worker_portfolios[worker_id]:
        final = worker_portfolios[worker_id][-1]
        rankings.append((worker_id, final))

rankings.sort(key=lambda x: x[1], reverse=True)

for rank, (worker_id, final) in enumerate(rankings, 1):
    initial = 20.50
    profit_pct = ((final - initial) / initial) * 100
    print(f"  {rank}. Worker {worker_id}: ${final:.2f} ({profit_pct:+.1f}%)")

print("\n" + "=" * 70)
