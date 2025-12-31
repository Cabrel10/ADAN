#!/usr/bin/env python3
"""
Extract worker metrics from training logs
Shows: Portfolio Value, Drawdown, Win Rate for each worker
"""
import re
import subprocess
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


def extract_metrics_from_log(log_file):
    """Extract metrics from log file using grep and regex"""
    metrics = defaultdict(lambda: {
        'portfolio_value': None,
        'drawdown': None,
        'win_rate': None,
        'trades': 0,
        'wins': 0,
        'losses': 0,
        'last_update': None
    })

    try:
        # Read last 5000 lines for performance
        with open(log_file, 'r') as f:
            lines = f.readlines()[-5000:]

        for line in lines:
            # Extract Worker ID
            worker_match = re.search(r'\[Worker (\d+)\]', line)
            if not worker_match:
                continue

            worker_id = f"Worker {worker_match.group(1)}"

            # Portfolio Value pattern
            if 'Portfolio Value:' in line:
                pv_match = re.search(
                    r'Portfolio Value:\s*([\d.]+)',
                    line
                )
                if pv_match:
                    metrics[worker_id]['portfolio_value'] = float(
                        pv_match.group(1)
                    )
                    metrics[worker_id]['last_update'] = line.split(' - ')[0]

            # Drawdown pattern
            if 'Drawdown' in line or 'DD' in line:
                dd_match = re.search(
                    r'(?:Drawdown|DD):\s*([-\d.]+)%?',
                    line
                )
                if dd_match:
                    metrics[worker_id]['drawdown'] = float(
                        dd_match.group(1)
                    )

            # Win Rate pattern
            if 'Win Rate' in line or 'Winrate' in line:
                wr_match = re.search(
                    r'(?:Win Rate|Winrate):\s*([\d.]+)%?',
                    line
                )
                if wr_match:
                    metrics[worker_id]['win_rate'] = float(
                        wr_match.group(1)
                    )

            # Trade outcomes
            if 'POSITION CLOSED' in line or 'Trade' in line:
                metrics[worker_id]['trades'] += 1

                if 'WIN' in line or 'profit' in line.lower():
                    metrics[worker_id]['wins'] += 1
                elif 'LOSS' in line or 'loss' in line.lower():
                    metrics[worker_id]['losses'] += 1

    except (IOError, OSError) as e:
        print(f"Error reading log: {e}")
        return metrics

    return metrics


def extract_with_grep(log_file):
    """Extract metrics using grep for more reliable parsing"""
    metrics = defaultdict(lambda: {
        'portfolio_value': None,
        'drawdown': None,
        'win_rate': None,
        'trades': 0,
        'wins': 0,
        'losses': 0,
        'last_update': None
    })

    try:
        # Get Portfolio Value lines
        result = subprocess.run(
            ["grep", "-i", "portfolio value", str(log_file)],
            capture_output=True,
            text=True,
            timeout=10
        )

        for line in result.stdout.strip().split('\n')[-10:]:
            if not line:
                continue

            worker_match = re.search(r'\[Worker (\d+)\]', line)
            if worker_match:
                worker_id = f"Worker {worker_match.group(1)}"
                pv_match = re.search(
                    r'Portfolio Value:\s*([\d.]+)',
                    line
                )
                if pv_match:
                    metrics[worker_id]['portfolio_value'] = float(
                        pv_match.group(1)
                    )
                    metrics[worker_id]['last_update'] = line.split(' - ')[0]

        # Get Drawdown lines
        result = subprocess.run(
            ["grep", "-iE", "drawdown|\\bDD\\b", str(log_file)],
            capture_output=True,
            text=True,
            timeout=10
        )

        for line in result.stdout.strip().split('\n')[-10:]:
            if not line:
                continue

            worker_match = re.search(r'\[Worker (\d+)\]', line)
            if worker_match:
                worker_id = f"Worker {worker_match.group(1)}"
                dd_match = re.search(
                    r'(?:Drawdown|DD):\s*([-\d.]+)',
                    line
                )
                if dd_match:
                    metrics[worker_id]['drawdown'] = float(
                        dd_match.group(1)
                    )

        # Get Win Rate lines
        result = subprocess.run(
            ["grep", "-iE", "win.?rate|winrate", str(log_file)],
            capture_output=True,
            text=True,
            timeout=10
        )

        for line in result.stdout.strip().split('\n')[-10:]:
            if not line:
                continue

            worker_match = re.search(r'\[Worker (\d+)\]', line)
            if worker_match:
                worker_id = f"Worker {worker_match.group(1)}"
                wr_match = re.search(
                    r'(?:Win Rate|Winrate):\s*([\d.]+)',
                    line
                )
                if wr_match:
                    metrics[worker_id]['win_rate'] = float(
                        wr_match.group(1)
                    )

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return metrics


def display_metrics(metrics):
    """Display metrics in a formatted table"""
    print("\n" + "=" * 90)
    print("📊 WORKER METRICS (Current Training Session)")
    print("=" * 90)

    if not metrics:
        print("No metrics found in logs yet. Training may still be initializing...")
        print("=" * 90 + "\n")
        return

    # Header
    print(f"{'Worker':<12} {'Portfolio Value':<18} {'Drawdown':<15} "
          f"{'Win Rate':<12} {'Last Update':<20}")
    print("-" * 90)

    # Data rows
    for worker_id in sorted(metrics.keys()):
        data = metrics[worker_id]

        pv = (f"${data['portfolio_value']:.2f}"
              if data['portfolio_value'] is not None else "N/A")
        dd = (f"{data['drawdown']:.2f}%"
              if data['drawdown'] is not None else "N/A")
        wr = (f"{data['win_rate']:.1f}%"
              if data['win_rate'] is not None else "N/A")
        last_update = data['last_update'] if data['last_update'] else "N/A"

        print(f"{worker_id:<12} {pv:<18} {dd:<15} {wr:<12} {last_update:<20}")

    print("=" * 90 + "\n")


def main():
    """Main function"""
    log_file = get_latest_log()

    if not log_file:
        print("❌ No log file found")
        return

    print(f"📝 Reading log: {log_file.name}")
    print(f"   Size: {log_file.stat().st_size / (1024**2):.1f} MB")

    # Try grep-based extraction first (more reliable)
    metrics = extract_with_grep(log_file)

    # Fallback to regex if grep didn't find much
    if not metrics or all(
        m['portfolio_value'] is None for m in metrics.values()
    ):
        print("   Using regex parsing...")
        metrics = extract_metrics_from_log(log_file)

    display_metrics(metrics)

    # Show raw grep examples
    print("💡 Raw log examples:")
    print("-" * 90)

    print("\n📌 Portfolio Value entries:")
    subprocess.run(
        ["grep", "-i", "portfolio value", str(log_file)],
        timeout=10
    )

    print("\n📌 Drawdown entries:")
    subprocess.run(
        ["grep", "-iE", "drawdown|\\bDD\\b", str(log_file)],
        timeout=10
    )

    print("\n📌 Win Rate entries:")
    subprocess.run(
        ["grep", "-iE", "win.?rate|winrate", str(log_file)],
        timeout=10
    )


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")

