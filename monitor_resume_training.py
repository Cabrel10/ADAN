#!/usr/bin/env python3
"""
Monitor Resume Training Progress
Affiche l'état de chaque worker en temps réel
"""
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

def get_latest_checkpoint_info(worker_id):
    """Get info about latest checkpoint for a worker"""
    checkpoint_dir = f"/mnt/new_data/t10_training/checkpoints/{worker_id}"
    
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        if f.endswith(".zip") and "_model_" in f:
            try:
                steps = int(f.split("_model_")[1].split("_steps")[0])
                path = os.path.join(checkpoint_dir, f)
                mtime = os.path.getmtime(path)
                checkpoints.append((steps, f, mtime))
            except (ValueError, IndexError):
                continue
    
    if not checkpoints:
        return None
    
    # Sort by steps (descending)
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    steps, filename, mtime = checkpoints[0]
    
    return {
        "steps": steps,
        "filename": filename,
        "mtime": mtime,
        "age_seconds": time.time() - mtime,
        "path": os.path.join(checkpoint_dir, filename),
    }

def get_log_tail(worker_id, lines=5):
    """Get last N lines from worker log"""
    log_dir = "/mnt/new_data/t10_training/logs"
    
    # Try to find worker log
    for f in os.listdir(log_dir):
        if worker_id in f and f.endswith(".log"):
            log_path = os.path.join(log_dir, f)
            try:
                with open(log_path, 'r') as lf:
                    all_lines = lf.readlines()
                    return all_lines[-lines:] if all_lines else []
            except:
                pass
    
    return []

def main():
    print("\n" + "=" * 80)
    print("📊 RESUME TRAINING MONITOR")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    workers = ["w1", "w2", "w3", "w4"]
    target_steps = 250000
    
    while True:
        print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Status Update")
        print("-" * 80)
        
        all_done = True
        
        for worker_id in workers:
            info = get_latest_checkpoint_info(worker_id)
            
            if info is None:
                print(f"❌ {worker_id}: No checkpoint found")
                all_done = False
                continue
            
            steps = info["steps"]
            age = info["age_seconds"]
            remaining = max(0, target_steps - steps)
            progress_pct = (steps / target_steps) * 100
            
            # Determine status
            if age < 60:
                status = "🟢 ACTIVE (updated < 1 min ago)"
            elif age < 300:
                status = "🟡 ACTIVE (updated < 5 min ago)"
            elif age < 3600:
                status = "🟠 SLOW (updated < 1 hour ago)"
            else:
                status = "🔴 STALLED (no update > 1 hour)"
                all_done = False
            
            print(f"{worker_id}: {steps:,} / {target_steps:,} steps ({progress_pct:.1f}%) {status}")
            print(f"     Remaining: {remaining:,} steps | Last update: {age:.0f}s ago")
            
            if remaining > 0:
                all_done = False
        
        print("-" * 80)
        
        if all_done:
            print("✅ ALL WORKERS COMPLETED!")
            break
        
        # Wait before next update
        print("Waiting 60 seconds before next update...")
        time.sleep(60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped by user")
        sys.exit(0)
