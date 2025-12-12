#!/usr/bin/env python3
"""
T10 : Monitoring en temps réel de l'entraînement
"""
import time
import psutil
import subprocess
import os
from datetime import datetime
from pathlib import Path

WORK_DIR = "/mnt/new_data/t10_training"
LOG_DIR = f"{WORK_DIR}/logs"

def monitor_worker(worker_id, log_file):
    """Monitor un worker en temps réel"""
    try:
        # Vérifier processus
        pid_file = f"{LOG_DIR}/{worker_id}.pid"
        if not os.path.exists(pid_file):
            return None, "PID file not found"
        
        with open(pid_file) as f:
            pid = int(f.read().strip())
        
        if not psutil.pid_exists(pid):
            return None, "Process not running"
        
        # Récupérer stats RAM/CPU
        process = psutil.Process(pid)
        mem_mb = process.memory_info().rss / 1024 / 1024
        cpu_pct = process.cpu_percent(interval=1)
        
        # Analyser log
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Dernières lignes
            tail = ''.join(lines[-50:])
            
            # Extraire métriques
            metrics = {}
            if "Sharpe" in tail:
                for line in lines[-20:]:
                    if "Sharpe" in line:
                        metrics['sharpe'] = line.strip()
            
            # Vérifier erreurs
            if "ERROR" in tail or "NaN" in tail:
                return None, "Error detected in log"
            
            return {
                'pid': pid,
                'ram_mb': mem_mb,
                'cpu_pct': cpu_pct,
                'metrics': metrics
            }, None
        
        return {
            'pid': pid,
            'ram_mb': mem_mb,
            'cpu_pct': cpu_pct,
            'metrics': {}
        }, None
    
    except Exception as e:
        return None, str(e)

def main():
    workers = {
        'W1': f'{LOG_DIR}/training_W1.log',
        'W2': f'{LOG_DIR}/training_W2.log',
        'W3': f'{LOG_DIR}/training_W3.log',
        'W4': f'{LOG_DIR}/training_W4.log'
    }
    
    print(f"🔄 Monitoring T10 - {datetime.now()}")
    print(f"📁 Logs: {LOG_DIR}")
    print("")
    
    while True:
        os.system('clear')
        print(f"\n🔄 MONITORING T10 - {datetime.now()}\n")
        
        # RAM globale
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/mnt/new_data')
        
        print(f"💾 RESSOURCES GLOBALES")
        print(f"   RAM: {mem.percent}% ({mem.used/1024/1024/1024:.1f}/{mem.total/1024/1024/1024:.1f} GB)")
        print(f"   Disque /mnt/new_data: {disk.percent}% ({disk.free/1024/1024/1024:.1f} GB libre)")
        print()
        
        # Monitorer chaque worker
        print(f"📊 WORKERS")
        for worker_id, log_file in workers.items():
            stats, error = monitor_worker(worker_id, log_file)
            
            if stats:
                print(f"   ✅ {worker_id}: PID={stats['pid']}, RAM={stats['ram_mb']:.0f}MB, CPU={stats['cpu_pct']:.1f}%")
                if stats['metrics']:
                    for key, val in stats['metrics'].items():
                        print(f"      📈 {val}")
            elif error:
                print(f"   ⏳ {worker_id}: {error}")
            else:
                print(f"   ⏳ {worker_id}: Waiting...")
        
        print()
        print("Press Ctrl+C to stop monitoring")
        time.sleep(300)  # 5 minutes

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")
