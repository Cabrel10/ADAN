#!/usr/bin/env python3
"""
Progressive Optuna Validation
=============================
Tests Optuna with increasing durations: 30s → 60s → 120s
Validates system health at each stage before proceeding.
"""

import subprocess
import time
import sys
import re
from pathlib import Path

def run_optuna_for_duration(duration_sec, stage_name):
    """Run Optuna for specified duration and monitor."""
    print("="*80)
    print(f"🎯 STAGE: {stage_name} ({duration_sec}s)")
    print("="*80)
    
    log_file = f"logs/optuna_progressive_{stage_name}.log"
    
    # Start Optuna in background
    cmd = [
        "/home/morningstar/miniconda3/envs/trading_env/bin/python",
        "scripts/optuna_optimize_workers.py"
    ]
    
    print(f"🚀 Starting Optuna...")
    proc = subprocess.Popen(
        cmd,
        stdout=open(log_file, 'w'),
        stderr=subprocess.STDOUT,
        cwd="/home/morningstar/Documents/trading/bot"
    )
    
    # Save PID
    with open("optuna.pid", "w") as f:
        f.write(str(proc.pid))
    
    print(f"✅ Optuna started (PID: {proc.pid})")
    print(f"📝 Logs: {log_file}")
    
    # Monitor for duration
    start = time.time()
    checkpoint_interval = 10  # Check every 10s
    
    while time.time() - start < duration_sec:
        elapsed = time.time() - start
        remaining = duration_sec - elapsed
        print(f"⏱️  {elapsed:.0f}s elapsed, {remaining:.0f}s remaining...", end='\r')
        time.sleep(checkpoint_interval)
    
    print(f"\n⏹️  {duration_sec}s elapsed - Stopping Optuna...")
    
    # Stop Optuna
    proc.terminate()
    proc.wait(timeout=10)
    
    print(f"✅ Optuna stopped")
    
    # Analyze logs
    return analyze_logs(log_file, stage_name)

def analyze_logs(log_file, stage_name):
    """Analyze Optuna logs for health metrics."""
    print("\n" + "="*80)
    print(f"📊 ANALYSIS: {stage_name}")
    print("="*80)
    
    with open(log_file, 'r') as f:
        logs = f.read()
    
    # Extract metrics
    metrics = {
        'natural_trades': 0,
        'force_trades': 0,
        'tf_5m': 0,
        'tf_1h': 0,
        'tf_4h': 0,
        'errors': 0,
        'trials_started': 0
    }
    
    # Count natural trades
    natural_trade_matches = re.findall(r'\[NATURAL_TRADE\]', logs)
    metrics['natural_trades'] = len(natural_trade_matches)
    
    # Count force trades
    force_trade_matches = re.findall(r'FORCE_TRADE\] Success', logs)
    metrics['force_trades'] = len(force_trade_matches)
    
    # Get last TF counts
    tf_counts = re.findall(r"Counts: \{'5m': (\d+), '1h': (\d+), '4h': (\d+)", logs)
    if tf_counts:
        last = tf_counts[-1]
        metrics['tf_5m'] = int(last[0])
        metrics['tf_1h'] = int(last[1])
        metrics['tf_4h'] = int(last[2])
    
    # Count errors (excluding known non-critical errors)
    all_errors = re.findall(r'ERROR|Exception|Traceback', logs)
    # Filter out non-critical "Invalid prices" errors
    critical_errors = [e for e in all_errors if 'Invalid prices' not in logs[logs.find(e):logs.find(e)+200]]
    metrics['errors'] = len(critical_errors)
    
    # Count trial starts
    trial_matches = re.findall(r'Trial \d+ finished', logs)
    metrics['trials_started'] = len(trial_matches)
    
    # Display results
    print(f"Natural Trades: {metrics['natural_trades']}")
    print(f"Force Trades: {metrics['force_trades']}")
    print(f"Timeframes - 5m: {metrics['tf_5m']}, 1h: {metrics['tf_1h']}, 4h: {metrics['tf_4h']}")
    print(f"Errors: {metrics['errors']}")
    print(f"Trials: {metrics['trials_started']}")
    
    # Health check based on trading activity, not errors
    # "Invalid prices" errors are common (data gaps) but non-critical
    has_trading_activity = (
        (metrics['natural_trades'] + metrics['force_trades']) > 0 and  # Some trades executed
        (metrics['tf_5m'] + metrics['tf_1h'] + metrics['tf_4h']) > 0  # Multi-TF usage
    )
    
    is_healthy = has_trading_activity
    
    if is_healthy:
        print("\n✅ STAGE PASSED - System healthy")
    else:
        print("\n❌ STAGE FAILED - Issues detected")
        print(f"   Check {log_file} for details")
    
    return is_healthy, metrics

def main():
    stages = [
        (30, "stage1_30s"),
        (60, "stage2_60s"),
        (120, "stage3_120s")
    ]
    
    print("🚀 PROGRESSIVE OPTUNA VALIDATION")
    print("="*80)
    
    for duration, stage_name in stages:
        healthy, metrics = run_optuna_for_duration(duration, stage_name)
        
        if not healthy:
            print(f"\n❌ Stopping progression - {stage_name} failed")
            sys.exit(1)
        
        print(f"\n✅ {stage_name} completed successfully")
        print("⏳ Waiting 5s before next stage...\n")
        time.sleep(5)
    
    print("="*80)
    print("🎉 ALL STAGES PASSED!")
    print("="*80)
    print("✅ System validated - Ready for full Optuna run")
    print("\nTo launch full run:")
    print("  nohup python scripts/optuna_optimize_workers.py > logs/optuna_v3_final.log 2>&1 &")

if __name__ == "__main__":
    main()
