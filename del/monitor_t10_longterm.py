#!/usr/bin/env python3
"""
T10 : Surveillance Long Terme de l'Entraînement
Monitoring continu avec alertes critiques
"""
import time
import psutil
import subprocess
import os
from datetime import datetime
from pathlib import Path
import json

WORK_DIR = "/mnt/new_data/t10_training"
LOG_DIR = f"{WORK_DIR}/logs"
CHECKPOINT_DIR = f"{WORK_DIR}/checkpoints"

def get_system_stats():
    """Récupérer les stats système"""
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
    """Vérifier l'état du processus principal"""
    try:
        result = subprocess.run(
            "ps aux | grep 'train_parallel_agents.py' | grep -v grep",
            shell=True,
            capture_output=True,
            text=True
        )
        return len(result.stdout.strip()) > 0
    except:
        return False

def check_worker_logs():
    """Vérifier les logs des workers"""
    workers_status = {}
    for worker_id in ['w1', 'w2', 'w3', 'w4']:
        log_file = f"{LOG_DIR}/training_{worker_id}.log"
        if os.path.exists(log_file):
            size_mb = os.path.getsize(log_file) / (1024 * 1024)
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            last_line = lines[-1] if lines else ""
            has_error = any("ERROR" in line or "CRITICAL" in line for line in lines[-50:])
            
            workers_status[worker_id] = {
                'log_size_mb': round(size_mb, 2),
                'last_update': last_line.strip()[:100],
                'has_errors': has_error,
                'status': '❌ ERROR' if has_error else '✅ OK'
            }
        else:
            workers_status[worker_id] = {
                'log_size_mb': 0,
                'last_update': 'No log file',
                'has_errors': False,
                'status': '⏳ WAITING'
            }
    
    return workers_status

def check_critical_issues(stats):
    """Vérifier les problèmes critiques"""
    issues = []
    
    if stats['ram_percent'] > 85:
        issues.append(f"🚨 RAM CRITIQUE: {stats['ram_percent']}%")
    
    if stats['disk_free_gb'] < 5:
        issues.append(f"🚨 DISQUE CRITIQUE: {stats['disk_free_gb']:.1f} GB restants")
    
    return issues

def main():
    print("="*80)
    print("T10 : SURVEILLANCE LONG TERME")
    print("="*80)
    print(f"📁 Répertoires:")
    print(f"   Logs: {LOG_DIR}")
    print(f"   Checkpoints: {CHECKPOINT_DIR}")
    print("")
    
    monitoring_log = f"{LOG_DIR}/monitoring.log"
    start_time = time.time()
    
    with open(monitoring_log, 'w') as f:
        f.write(f"T10 Monitoring Started: {datetime.now().isoformat()}\n")
        f.write("="*80 + "\n\n")
    
    iteration = 0
    while True:
        iteration += 1
        stats = get_system_stats()
        process_running = check_process_status()
        workers_status = check_worker_logs()
        issues = check_critical_issues(stats)
        
        elapsed_hours = (time.time() - start_time) / 3600
        
        # Afficher le statut
        os.system('clear')
        print(f"\n🔄 T10 MONITORING - Iteration {iteration} - {elapsed_hours:.1f}h elapsed\n")
        
        print(f"📊 SYSTÈME")
        print(f"   RAM: {stats['ram_percent']:.1f}% ({stats['ram_used_gb']:.1f}/{stats['ram_total_gb']:.1f} GB)")
        print(f"   Disque: {stats['disk_percent']:.1f}% ({stats['disk_free_gb']:.1f} GB libre)")
        print(f"   Processus: {'✅ RUNNING' if process_running else '❌ STOPPED'}")
        print()
        
        print(f"👷 WORKERS")
        for worker_id, status in workers_status.items():
            print(f"   {worker_id}: {status['status']} ({status['log_size_mb']} MB)")
            if status['last_update'] != 'No log file':
                print(f"      └─ {status['last_update']}")
        print()
        
        if issues:
            print(f"🚨 ALERTES CRITIQUES")
            for issue in issues:
                print(f"   {issue}")
            print()
        
        # Enregistrer dans le log
        with open(monitoring_log, 'a') as f:
            f.write(f"\n[{datetime.now().isoformat()}] Iteration {iteration}\n")
            f.write(f"RAM: {stats['ram_percent']:.1f}% | Disk: {stats['disk_percent']:.1f}% | Process: {'RUNNING' if process_running else 'STOPPED'}\n")
            for worker_id, status in workers_status.items():
                f.write(f"  {worker_id}: {status['status']} ({status['log_size_mb']} MB)\n")
            if issues:
                for issue in issues:
                    f.write(f"  ALERT: {issue}\n")
        
        # Attendre avant la prochaine itération
        print(f"⏳ Prochaine vérification dans 5 minutes...")
        time.sleep(300)  # 5 minutes

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")
