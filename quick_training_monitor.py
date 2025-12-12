#!/usr/bin/env python3
"""
Quick Training Monitor - Lightweight status check
"""
import os
import subprocess
import psutil
from datetime import datetime
from pathlib import Path

LOG_DIR = "/mnt/new_data/t10_training/logs"
CHECKPOINT_DIR = "/mnt/new_data/t10_training/checkpoints"


def get_latest_log():
    """Get the most recent log file"""
    try:
        logs = sorted(
            Path(LOG_DIR).glob("training_final_*.log"),
            key=os.path.getmtime,
            reverse=True
        )
        return logs[0] if logs else None
    except (OSError, FileNotFoundError):
        return None


def get_process_info():
    """Get training process information"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "train_parallel_agents.py"],
            capture_output=True,
            text=True,
            timeout=5
        )
        pids = result.stdout.strip().split('\n')
        pids = [p for p in pids if p]
        return len(pids), pids
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 0, []


def get_system_info():
    """Get system resource usage"""
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/mnt/new_data')

    return {
        'ram_percent': mem.percent,
        'ram_used': mem.used / (1024 ** 3),
        'ram_total': mem.total / (1024 ** 3),
        'disk_percent': disk.percent,
        'disk_free': disk.free / (1024 ** 3),
    }


def get_checkpoint_info():
    """Get checkpoint directory info"""
    try:
        result = subprocess.run(
            ["du", "-sh", CHECKPOINT_DIR],
            capture_output=True,
            text=True,
            timeout=5
        )
        size = result.stdout.split()[0] if result.stdout else "0"

        count = len([
            d for d in os.listdir(CHECKPOINT_DIR)
            if os.path.isdir(os.path.join(CHECKPOINT_DIR, d))
        ])

        return count, size
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return 0, "0"


def get_log_stats(log_file):
    """Get log file statistics"""
    try:
        size_mb = os.path.getsize(log_file) / (1024 ** 2)

        with open(log_file, 'r') as f:
            lines = f.readlines()

        # Get last meaningful line
        last_line = ""
        for line in reversed(lines[-50:]):
            if line.strip() and not line.startswith('\n'):
                last_line = line.strip()[:80]
                break

        # Check for errors
        has_errors = any(
            "ERROR" in line or "CRITICAL" in line
            for line in lines[-100:]
        )

        return size_mb, last_line, has_errors
    except (IOError, OSError):
        return 0, "", False


def main():
    """Display training status"""
    print("\n" + "=" * 80)
    print("🚀 T10 TRAINING STATUS")
    print("=" * 80 + "\n")

    # Process info
    count, pids = get_process_info()
    if count > 0:
        print(f"✅ RUNNING - {count} processes (Parent + {count-1} workers)")
        print(f"   PIDs: {', '.join(pids[:3])}{'...' if len(pids) > 3 else ''}")
    else:
        print("❌ STOPPED")

    print()

    # System info
    sys_info = get_system_info()
    print(f"💾 SYSTEM")
    print(f"   RAM: {sys_info['ram_percent']:.1f}% "
          f"({sys_info['ram_used']:.1f}/{sys_info['ram_total']:.1f} GB)")
    print(f"   Disk: {sys_info['disk_percent']:.1f}% "
          f"({sys_info['disk_free']:.1f} GB free)")

    print()

    # Checkpoint info
    cp_count, cp_size = get_checkpoint_info()
    print(f"📈 CHECKPOINTS")
    print(f"   Count: {cp_count}")
    print(f"   Size: {cp_size}")

    print()

    # Log info
    log_file = get_latest_log()
    if log_file:
        size_mb, last_line, has_errors = get_log_stats(log_file)
        print(f"📝 LOG FILE")
        print(f"   Name: {log_file.name}")
        print(f"   Size: {size_mb:.1f} MB")
        if has_errors:
            print(f"   ⚠️  Errors detected")
        if last_line:
            print(f"   Last: {last_line}")
    else:
        print("📝 LOG FILE: Not found")

    print()
    print("=" * 80)
    print("Commands:")
    print("  ./check_training_status.sh tail     - Last 20 lines")
    print("  ./check_training_status.sh monitor  - Live tail")
    print("  tail -f /mnt/new_data/t10_training/logs/training_final_*.log")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()

