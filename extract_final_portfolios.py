#!/usr/bin/env python3
import re
from pathlib import Path
from collections import defaultdict

log_file = Path("/mnt/new_data/adan_logs/training_20251208_005653.log")

# Lire tout le fichier
with open(log_file, 'r') as f:
    lines = f.readlines()

# Dictionnaire pour stocker les derniers portfolios par worker
worker_portfolios = {}

# Parcourir les lignes en reverse pour trouver les derniers portfolios
for line in reversed(lines):
    if "Portfolio value:" in line:
        # Extraire le worker ID et la valeur
        match = re.search(r'Portfolio value: ([\d.]+)', line)
        if match:
            value = float(match.group(1))
            
            # Déterminer le worker
            if "[Worker 0]" in line or "Worker 0" in line:
                if 0 not in worker_portfolios:
                    worker_portfolios[0] = value
            elif "[Worker 1]" in line or "Worker 1" in line:
                if 1 not in worker_portfolios:
                    worker_portfolios[1] = value
            elif "[Worker 2]" in line or "Worker 2" in line:
                if 2 not in worker_portfolios:
                    worker_portfolios[2] = value
            elif "[Worker 3]" in line or "Worker 3" in line:
                if 3 not in worker_portfolios:
                    worker_portfolios[3] = value

print("=" * 80)
print("🏆 PORTFOLIOS FINAUX PAR WORKER")
print("=" * 80)
print()

initial = 20.50
rankings = []

for worker_id in range(4):
    if worker_id in worker_portfolios:
        final = worker_portfolios[worker_id]
        profit = final - initial
        profit_pct = (profit / initial) * 100
        rankings.append((worker_id, final, profit_pct))
        
        print(f"Worker {worker_id}:")
        print(f"  Initial:  ${initial:.2f}")
        print(f"  Final:    ${final:.2f}")
        print(f"  Profit:   ${profit:+.2f} ({profit_pct:+.1f}%)")
        print()

print("=" * 80)
print("🥇 CLASSEMENT (du plus riche au plus pauvre):")
print("=" * 80)
print()

rankings.sort(key=lambda x: x[1], reverse=True)
medals = ["🥇", "🥈", "🥉", "4️⃣"]

for rank, (worker_id, final, profit_pct) in enumerate(rankings):
    status = "✅ RICHE" if profit_pct > 0 else "❌ PAUVRE"
    print(f"{medals[rank]} Worker {worker_id}: ${final:.2f} ({profit_pct:+.1f}%) {status}")

print()
print("=" * 80)
print("📊 ANALYSE:")
print("=" * 80)

# Compter les riches et pauvres
rich = sum(1 for _, _, pct in rankings if pct > 0)
poor = sum(1 for _, _, pct in rankings if pct < 0)

print(f"Riches (profit > 0): {rich}/4")
print(f"Pauvres (profit < 0): {poor}/4")

if rich > 0:
    best_worker = rankings[0][0]
    best_profit = rankings[0][2]
    print(f"\n🏆 LE PLUS RICHE: Worker {best_worker} avec {best_profit:+.1f}%")

print()
