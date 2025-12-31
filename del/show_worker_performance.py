#!/usr/bin/env python3
"""
Show current worker performance metrics
Extracts: Portfolio Value, Win Rate, Sharpe Ratio, Trades
"""
import subprocess
import re
from pathlib import Path
from collections import defaultdict

LOG_DIR = "/mnt/new_data/t10_training/logs"


def get_latest_log():
    """Get the most recent log file"""
    try:
        logs = sorted(
            Path(LOG_DIR).glob("training_final_*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return logs[0] if logs else None
    except (OSError, FileNotFoundError):
        return None


def extract_metrics():
    """Extract metrics from log using grep"""
    log_file = get_latest_log()

    if not log_file:
        print("❌ No log file found")
        return

    metrics = defaultdict(lambda: {
        'portfolio_value': None,
        'win_rate': None,
        'sharpe': None,
        'sortino': None,
        'trades': 0,
        'step': 0
    })

    try:
        # Get METRICS_SYNC lines (most recent metrics)
        result = subprocess.run(
            ["grep", "-E", "METRICS_SYNC.*Worker", str(log_file)],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Parse each line
        for line in result.stdout.strip().split('\n')[-20:]:
            if not line:
                continue

            # Extract Worker ID
            worker_match = re.search(r'Worker (\d+)', line)
            if not worker_match:
                continue

            worker_id = int(worker_match.group(1))

            # Extract Step
            step_match = re.search(r'Step (\d+)', line)
            if step_match:
                metrics[worker_id]['step'] = int(step_match.group(1))

            # Extract Sharpe
            sharpe_match = re.search(r'Sharpe=([\d.]+)', line)
            if sharpe_match:
                metrics[worker_id]['sharpe'] = float(sharpe_match.group(1))

            # Extract Sortino
            sortino_match = re.search(r'Sortino=([\d.]+)', line)
            if sortino_match:
                metrics[worker_id]['sortino'] = float(sortino_match.group(1))

            # Extract Win Rate
            wr_match = re.search(r'WinRate=([\d.]+)%', line)
            if wr_match:
                metrics[worker_id]['win_rate'] = float(wr_match.group(1))

            # Extract Trades
            trades_match = re.search(r'Trades=(\d+)', line)
            if trades_match:
                metrics[worker_id]['trades'] = int(trades_match.group(1))

        # Get Portfolio Values
        result = subprocess.run(
            ["grep", "-E", "Portfolio Value:", str(log_file)],
            capture_output=True,
            text=True,
            timeout=10
        )

        for line in result.stdout.strip().split('\n')[-50:]:
            if not line:
                continue

            worker_match = re.search(r'\[Worker (\d+)\]', line)
            if not worker_match:
                continue

            worker_id = int(worker_match.group(1))

            pv_match = re.search(r'Portfolio Value:\s*([\d.]+)', line)
            if pv_match:
                metrics[worker_id]['portfolio_value'] = float(
                    pv_match.group(1)
                )

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Display results
    print("\n" + "=" * 100)
    print("🚀 WORKER PERFORMANCE METRICS")
    print("=" * 100)
    print()

    if not metrics:
        print("No metrics found yet. Training may still be initializing...")
        print("=" * 100 + "\n")
        return

    # Header
    print(f"{'Worker':<10} {'Step':<8} {'Portfolio':<15} {'Win Rate':<12} "
          f"{'Sharpe':<10} {'Sortino':<10} {'Trades':<8}")
    print("-" * 100)

    # Data rows
    for worker_id in sorted(metrics.keys()):
        data = metrics[worker_id]

        pv = (f"${data['portfolio_value']:.2f}"
              if data['portfolio_value'] is not None else "N/A")
        wr = (f"{data['win_rate']:.1f}%"
              if data['win_rate'] is not None else "N/A")
        sharpe = (f"{data['sharpe']:.2f}"
                  if data['sharpe'] is not None else "N/A")
        sortino = (f"{data['sortino']:.2f}"
                   if data['sortino'] is not None else "N/A")
        trades = data['trades'] if data['trades'] > 0 else "N/A"
        step = data['step'] if data['step'] > 0 else "N/A"

        print(f"Worker {worker_id:<2} {step:<8} {pv:<15} {wr:<12} "
              f"{sharpe:<10} {sortino:<10} {trades:<8}")

    print("=" * 100)
    print()

    # Show raw examples
    print("📝 Raw log examples:")
    print("-" * 100)

    print("\n📌 Latest METRICS_SYNC entries:")
    subprocess.run(
        ["grep", "-E", "METRICS_SYNC.*Worker", str(log_file)],
        timeout=10
    )

    print("\n📌 Latest Portfolio Values:")
    subprocess.run(
        ["grep", "-E", "Portfolio Value:", str(log_file)],
        timeout=10
    )


if __name__ == '__main__':
    try:
        extract_metrics()
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")

