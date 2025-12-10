#!/usr/bin/env python3
import re
from pathlib import Path

log_file = Path("/mnt/new_data/adan_logs/training_20251207_200003.log")

# Lire les dernières 5000 lignes
with open(log_file, 'r') as f:
    lines = f.readlines()[-5000:]

# Chercher les derniers portfolios de chaque worker
worker_data = {}

for line in reversed(lines):
    # Chercher les patterns "Worker X" avec Portfolio value
    if "Portfolio value:" in line:
        # Extraire le worker ID
        if "[Worker 0]" in line or "Worker 0" in line:
            if 0 not in worker_data:
                match = re.search(r'Portfolio value: ([\d.]+)', line)
                if match:
                    worker_data[0] = float(match.group(1))
        elif "[Worker 1]" in line or "Worker 1" in line:
            if 1 not in worker_data:
                match = re.search(r'Portfolio value: ([\d.]+)', line)
                if match:
                    worker_data[1] = float(match.group(1))
        elif "[Worker 2]" in line or "Worker 2" in line:
            if 2 not in worker_data:
                match = re.search(r'Portfolio value: ([\d.]+)', line)
                if match:
                    worker_data[2] = float(match.group(1))
        elif "[Worker 3]" in line or "Worker 3" in line:
            if 3 not in worker_data:
                match = re.search(r'Portfolio value: ([\d.]+)', line)
                if match:
                    worker_data[3] = float(match.group(1))

print("🏆 PORTFOLIOS FINAUX PAR WORKER:\n")
print("=" * 60)

initial = 20.50
rankings = []

for worker_id in range(4):
    if worker_id in worker_data:
        final = worker_data[worker_id]
        profit = final - initial
        profit_pct = (profit / initial) * 100
        rankings.append((worker_id, final, profit_pct))
        
        print(f"Worker {worker_id}:")
        print(f"  Initial:  ${initial:.2f}")
        print(f"  Final:    ${final:.2f}")
        print(f"  Profit:   ${profit:+.2f} ({profit_pct:+.1f}%)")
        print()

print("=" * 60)
print("\n🥇 CLASSEMENT:\n")

rankings.sort(key=lambda x: x[1], reverse=True)
medals = ["🥇", "🥈", "🥉", "4️⃣"]

for rank, (worker_id, final, profit_pct) in enumerate(rankings):
    print(f"{medals[rank]} Worker {worker_id}: ${final:.2f} ({profit_pct:+.1f}%)")

print("\n" + "=" * 60)
