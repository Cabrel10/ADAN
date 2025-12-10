#!/usr/bin/env python3
"""
Analyse détaillée des logs d'entraînement par worker.
Usage: python analyze_logs_detailed.py [log_file]
"""

import sys
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def parse_log_file(log_path):
    """Parse le fichier log et extrait les données par worker."""
    
    if not Path(log_path).exists():
        print(f"❌ Log file not found: {log_path}")
        return None
    
    workers_data = defaultdict(lambda: {
        'steps': [],
        'rewards': [],
        'pnl': [],
        'portfolio': [],
        'trades': [],
        'errors': [],
        'last_update': None
    })
    
    print(f"📖 Analysing log file: {log_path}")
    print(f"   Size: {Path(log_path).stat().st_size / 1024 / 1024:.1f} MB")
    print("")
    
    with open(log_path) as f:
        for line_num, line in enumerate(f, 1):
            # Extraire Worker ID
            worker_match = re.search(r'\[Worker (\d)\]', line)
            if not worker_match:
                continue
            
            worker_id = f"W{worker_match.group(1)}"
            
            # Extraire Step
            step_match = re.search(r'Step[:\s]+(\d+)', line)
            if step_match:
                step = int(step_match.group(1))
                workers_data[worker_id]['steps'].append(step)
            
            # Extraire Reward
            reward_match = re.search(r'[Rr]eward[:\s]+([-\d.]+)', line)
            if reward_match:
                reward = float(reward_match.group(1))
                workers_data[worker_id]['rewards'].append(reward)
            
            # Extraire PnL
            pnl_match = re.search(r'PnL[:\s]*\$?([-\d.]+)', line)
            if pnl_match:
                pnl = float(pnl_match.group(1))
                workers_data[worker_id]['pnl'].append(pnl)
            
            # Extraire Portfolio Value
            portfolio_match = re.search(r'[Pp]ortfolio[:\s]*\$?([\d.]+)', line)
            if portfolio_match:
                portfolio = float(portfolio_match.group(1))
                workers_data[worker_id]['portfolio'].append(portfolio)
            
            # Extraire Trades
            trades_match = re.search(r'[Tt]rades[:\s]*(\d+)', line)
            if trades_match:
                trades = int(trades_match.group(1))
                workers_data[worker_id]['trades'].append(trades)
            
            # Chercher les erreurs
            if 'error' in line.lower() or 'exception' in line.lower():
                workers_data[worker_id]['errors'].append(line.strip())
            
            # Chercher les NaN/Inf
            if 'nan' in line.lower() or 'inf' in line.lower():
                workers_data[worker_id]['errors'].append(f"NaN/Inf detected: {line.strip()}")
            
            workers_data[worker_id]['last_update'] = datetime.now()
    
    return workers_data


def print_worker_summary(worker_id, data):
    """Affiche un résumé pour un worker."""
    
    print(f"\n{'='*70}")
    print(f"📊 {worker_id} SUMMARY")
    print(f"{'='*70}")
    
    if not data['steps']:
        print(f"❌ No data found for {worker_id}")
        return
    
    # Steps
    steps = data['steps']
    print(f"\n✅ Steps: {len(steps)} entries")
    print(f"   Min: {min(steps)}, Max: {max(steps)}")
    if len(steps) > 1:
        print(f"   Latest: {steps[-1]}")
    
    # Rewards
    if data['rewards']:
        rewards = data['rewards']
        avg_reward = sum(rewards) / len(rewards)
        print(f"\n💰 Rewards: {len(rewards)} entries")
        print(f"   Min: {min(rewards):.4f}, Max: {max(rewards):.4f}")
        print(f"   Avg: {avg_reward:.4f}")
        if len(rewards) > 1:
            print(f"   Latest: {rewards[-1]:.4f}")
            if avg_reward < -0.5:
                print(f"   ⚠️  VERY NEGATIVE - Check reward function!")
    
    # PnL
    if data['pnl']:
        pnl = data['pnl']
        total_pnl = sum(pnl)
        print(f"\n💵 PnL: {len(pnl)} entries")
        print(f"   Min: ${min(pnl):.2f}, Max: ${max(pnl):.2f}")
        print(f"   Total: ${total_pnl:.2f}")
        if len(pnl) > 1:
            print(f"   Latest: ${pnl[-1]:.2f}")
            if total_pnl < 0:
                print(f"   ❌ NEGATIVE PnL - Model losing money!")
    
    # Portfolio
    if data['portfolio']:
        portfolio = data['portfolio']
        print(f"\n📈 Portfolio: {len(portfolio)} entries")
        print(f"   Min: ${min(portfolio):.2f}, Max: ${max(portfolio):.2f}")
        if len(portfolio) > 1:
            print(f"   Latest: ${portfolio[-1]:.2f}")
            print(f"   Change: ${portfolio[-1] - portfolio[0]:.2f}")
    
    # Trades
    if data['trades']:
        trades = data['trades']
        total_trades = sum(trades)
        print(f"\n🔄 Trades: {len(trades)} entries")
        print(f"   Total: {total_trades}")
        print(f"   Avg per entry: {total_trades / len(trades):.2f}")
    
    # Errors
    if data['errors']:
        print(f"\n❌ Errors: {len(data['errors'])} found")
        for error in data['errors'][:5]:
            print(f"   - {error[:100]}")
        if len(data['errors']) > 5:
            print(f"   ... and {len(data['errors']) - 5} more")


def main():
    """Main function."""
    
    # Get log file path
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        # Find latest log
        log_dir = Path("/mnt/new_data/adan_logs")
        if log_dir.exists():
            logs = sorted(log_dir.glob("training_*.log"), reverse=True)
            if logs:
                log_path = str(logs[0])
                print(f"📁 Using latest log: {logs[0].name}\n")
            else:
                print("❌ No training logs found")
                return 1
        else:
            print(f"❌ Log directory not found: {log_dir}")
            return 1
    
    # Parse log
    workers_data = parse_log_file(log_path)
    if not workers_data:
        return 1
    
    # Print summary for each worker
    for worker_id in sorted(workers_data.keys()):
        print_worker_summary(worker_id, workers_data[worker_id])
    
    # Global summary
    print(f"\n{'='*70}")
    print(f"🎯 GLOBAL SUMMARY")
    print(f"{'='*70}")
    
    active_workers = [w for w in workers_data if workers_data[w]['steps']]
    print(f"\nActive workers: {len(active_workers)}/4")
    for worker in active_workers:
        print(f"  ✅ {worker}: {len(workers_data[worker]['steps'])} steps")
    
    inactive_workers = [w for w in workers_data if not workers_data[w]['steps']]
    if inactive_workers:
        print(f"\nInactive workers: {len(inactive_workers)}/4")
        for worker in inactive_workers:
            print(f"  ❌ {worker}: No data")
    
    # Recommendations
    print(f"\n{'='*70}")
    print(f"💡 RECOMMENDATIONS")
    print(f"{'='*70}")
    
    if len(active_workers) < 4:
        print(f"\n⚠️  Only {len(active_workers)}/4 workers are logging data!")
        print(f"   Action: Check if W{inactive_workers} are running or crashing")
    
    for worker in active_workers:
        if workers_data[worker]['rewards']:
            avg_reward = sum(workers_data[worker]['rewards']) / len(workers_data[worker]['rewards'])
            if avg_reward < -0.5:
                print(f"\n⚠️  {worker} has very negative rewards ({avg_reward:.4f})")
                print(f"   Action: Check reward function or model learning")
    
    for worker in active_workers:
        if workers_data[worker]['pnl']:
            total_pnl = sum(workers_data[worker]['pnl'])
            if total_pnl < 0:
                print(f"\n⚠️  {worker} has negative PnL (${total_pnl:.2f})")
                print(f"   Action: Check stop loss / take profit settings")
    
    print(f"\n{'='*70}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
