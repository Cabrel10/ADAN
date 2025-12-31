#!/usr/bin/env python3
"""
Extract and analyze portfolio metrics from training logs
"""
import re
from collections import defaultdict
from datetime import datetime

# Read the log file
log_file = "nohup_training_350k_20251212_024226.log"

portfolio_data = defaultdict(list)
pnl_data = defaultdict(list)

try:
    with open(log_file, 'r') as f:
        for line in f:
            # Extract portfolio values
            if "Portfolio value:" in line:
                match = re.search(r'Portfolio value: ([\d.]+)', line)
                if match:
                    value = float(match.group(1))
                    # Determine worker from context
                    if "STEP" in line:
                        step_match = re.search(r'\[STEP (\d+)\]', line)
                        if step_match:
                            portfolio_data['all'].append(value)
            
            # Extract PnL values
            if "Realized PnL for step:" in line:
                match = re.search(r'Realized PnL for step: \$([+-]?[\d.]+)', line)
                if match:
                    pnl = float(match.group(1))
                    pnl_data['all'].append(pnl)
            
            # Extract position closes with PnL
            if "[POSITION FERMÉE]" in line:
                match = re.search(r'PnL: \$([+-]?[\d.]+)', line)
                if match:
                    pnl = float(match.group(1))
                    pnl_data['trades'].append(pnl)

except FileNotFoundError:
    print(f"❌ Log file not found: {log_file}")
    exit(1)

# Analysis
print("╔════════════════════════════════════════════════════════════════╗")
print("║           📊 PORTFOLIO METRICS ANALYSIS                        ║")
print("╚════════════════════════════════════════════════════════════════╝")
print()

if portfolio_data['all']:
    values = portfolio_data['all']
    print("💰 PORTFOLIO VALUES")
    print("═" * 64)
    print(f"Latest Value:        ${values[-1]:.2f}")
    print(f"Highest Value:       ${max(values):.2f}")
    print(f"Lowest Value:        ${min(values):.2f}")
    print(f"Average Value:       ${sum(values)/len(values):.2f}")
    print(f"Total Observations:  {len(values)}")
    print()

if pnl_data['all']:
    pnl_all = pnl_data['all']
    print("📈 REALIZED PnL (All Steps)")
    print("═" * 64)
    total_pnl = sum(pnl_all)
    positive = sum(1 for p in pnl_all if p > 0)
    negative = sum(1 for p in pnl_all if p < 0)
    zero = sum(1 for p in pnl_all if p == 0)
    
    print(f"Total PnL:           ${total_pnl:.2f}")
    print(f"Positive Steps:      {positive} ({100*positive/len(pnl_all):.1f}%)")
    print(f"Negative Steps:      {negative} ({100*negative/len(pnl_all):.1f}%)")
    print(f"Zero Steps:          {zero} ({100*zero/len(pnl_all):.1f}%)")
    print(f"Average PnL/Step:    ${sum(pnl_all)/len(pnl_all):.4f}")
    print(f"Max Gain:            ${max(pnl_all):.2f}")
    print(f"Max Loss:            ${min(pnl_all):.2f}")
    print()

if pnl_data['trades']:
    trades = pnl_data['trades']
    print("🎯 TRADE PnL (Closed Positions)")
    print("═" * 64)
    total_trade_pnl = sum(trades)
    winning_trades = sum(1 for t in trades if t > 0)
    losing_trades = sum(1 for t in trades if t < 0)
    
    print(f"Total Trades:        {len(trades)}")
    print(f"Total Trade PnL:     ${total_trade_pnl:.2f}")
    print(f"Winning Trades:      {winning_trades} ({100*winning_trades/len(trades):.1f}%)")
    print(f"Losing Trades:       {losing_trades} ({100*losing_trades/len(trades):.1f}%)")
    print(f"Average Trade PnL:   ${sum(trades)/len(trades):.4f}")
    print(f"Best Trade:          ${max(trades):.2f}")
    print(f"Worst Trade:         ${min(trades):.2f}")
    print()

print("✅ ANALYSIS COMPLETE")
print("═" * 64)
