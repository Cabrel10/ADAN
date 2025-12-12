#!/usr/bin/env python3
"""
T10 Training Monitoring - Real-time progress tracking
Monitors system resources, training logs, and worker status
"""
import time
import psutil
import subprocess
import os
from datetime import datetime
from pathlib import Path

WORK_DIR = "/mnt/new_data/t10_training"
LOG_DIR = f"{WORK_DIR}/logs"
CHECKPOINT_DIR = f"{WORK_DIR}/checkpoints"


def get_system_stats():
    """Get current system statistics"""
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/mnt/new_data')

    return {
        'timestamp': datetime.now().isoformat(),
        'ram_percent': mem.percent,
        'ram_used_gb': mem.used / 1024 / 1024 / 1024,
        'ram_total_gb': mem.total / 1024 / 1024 / 1024,
        'disk_percent': disk.percent,
        'disk_free_gb': disk.free / 1024 / 1024 / 1024,
        'disk_total_gb': disk.total / 1024 / 1024 / 1024,
    }


def check_process_status():
    """Check if training process is running"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "train_parallel_agents.py"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return len(result.stdout.strip()) > 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_latest_log_file():
    """Get the most recent training log file"""
    if not os.path.exists(LOG_DIR):
        return None

    log_files = [
        f for f in os.listdir(LOG_DIR)
        if f.startswith('training_final_') and f.endswith('.log')
    ]

    if not log_files:
        return None

    return os.path.join(LOG_DIR, sorted(log_files)[-1])


def get_log_tail(filepath, lines=10):
    """Get last N lines from log file"""
    try:
        with open(filepath, 'r') as f:
            all_lines = f.readlines()
        return all_lines[-lines:] if all_lines else []
    except (IOError, OSError):
        return []


def check_for_errors(filepath):
    """Check if log contains error messages"""
    try:
        with open(filepath, 'r') as f:
            recent_lines = f.readlines()[-100:]
        return any(
            "ERROR" in line or "CRITICAL" in line or "Exception" in line
            for line in recent_lines
        )
    except (IOError, OSError):
        return False


def get_checkpoint_status():
    """Get checkpoint directory status"""
    if not os.path.exists(CHECKPOINT_DIR):
        return {'count': 0, 'size_gb': 0}

    try:
        checkpoints = [
            d for d in os.listdir(CHECKPOINT_DIR)
            if os.path.isdir(os.path.join(CHECKPOINT_DIR, d))
        ]

        total_size = 0
        for checkpoint in checkpoints:
            path = os.path.join(CHECKPOINT_DIR, checkpoint)
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (IOError, OSError):
                        pass

        return {
            'count': len(checkpoints),
            'size_gb': total_size / (1024 ** 3)
        }
    except (IOError, OSError):
        return {'count': 0, 'size_gb': 0}


def check_critical_issues(stats):
    """Check for critical system issues"""
    issues = []

    if stats['ram_percent'] > 85:
        issues.append(
            f"🚨 RAM CRITICAL: {stats['ram_percent']:.1f}%"
        )

    if stats['disk_free_gb'] < 5:
        issues.append(
            f"🚨 DISK CRITICAL: {stats['disk_free_gb']:.1f} GB remaining"
        )

    return issues


def main():
    """Main monitoring loop"""
    print("=" * 80)
    print("T10 : TRAINING MONITORING")
    print("=" * 80)
    print(f"📁 Directories:")
    print(f"   Logs: {LOG_DIR}")
    print(f"   Checkpoints: {CHECKPOINT_DIR}")
    print("")

    monitoring_log = os.path.join(LOG_DIR, "monitoring.log")
    start_time = time.time()

    # Initialize monitoring log
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(monitoring_log, 'w') as f:
        f.write(f"T10 Monitoring Started: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")

    iteration = 0
    while True:
        iteration += 1
        stats = get_system_stats()
        process_running = check_process_status()
        latest_log = get_latest_log_file()
        issues = check_critical_issues(stats)
        checkpoint_status = get_checkpoint_status()

        elapsed_hours = (time.time() - start_time) / 3600

        # Display status
        os.system('clear')
        print(f"\n🔄 T10 MONITORING - Iteration {iteration} - "
              f"{elapsed_hours:.1f}h elapsed\n")

        print("📊 SYSTEM")
        print(f"   RAM: {stats['ram_percent']:.1f}% "
              f"({stats['ram_used_gb']:.1f}/{stats['ram_total_gb']:.1f} GB)")
        print(f"   Disk: {stats['disk_percent']:.1f}% "
              f"({stats['disk_free_gb']:.1f} GB free)")
        print(f"   Process: {'✅ RUNNING' if process_running else '❌ STOPPED'}")
        print()

        print("📈 TRAINING")
        print(f"   Checkpoints: {checkpoint_status['count']} "
              f"({checkpoint_status['size_gb']:.2f} GB)")
        if latest_log:
            log_size_mb = os.path.getsize(latest_log) / (1024 * 1024)
            print(f"   Log file: {os.path.basename(latest_log)} "
                  f"({log_size_mb:.1f} MB)")

            # Show last few log lines
            tail_lines = get_log_tail(latest_log, 3)
            if tail_lines:
                print("   Recent log:")
                for line in tail_lines:
                    line_text = line.strip()[:70]
                    if line_text:
                        print(f"      └─ {line_text}")

            # Check for errors
            if check_for_errors(latest_log):
                print("   ⚠️  Errors detected in log")
        else:
            print("   ⏳ Waiting for log file...")
        print()

        if issues:
            print("🚨 CRITICAL ALERTS")
            for issue in issues:
                print(f"   {issue}")
            print()

        # Log monitoring data
        with open(monitoring_log, 'a') as f:
            f.write(f"\n[{datetime.now().isoformat()}] Iteration {iteration}\n")
            f.write(f"RAM: {stats['ram_percent']:.1f}% | "
                    f"Disk: {stats['disk_percent']:.1f}% | "
                    f"Process: {'RUNNING' if process_running else 'STOPPED'}\n")
            f.write(f"Checkpoints: {checkpoint_status['count']} | "
                    f"Size: {checkpoint_status['size_gb']:.2f} GB\n")
            if issues:
                for issue in issues:
                    f.write(f"ALERT: {issue}\n")

        print(f"⏳ Next check in 5 minutes...")
        time.sleep(300)  # 5 minutes


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped (training continues in background)")

