#!/usr/bin/env python3
"""
Script de monitoring pour l'optimisation Optuna
Surveille la progression en temps réel et affiche les métriques
"""
import sys
import time
import os
from pathlib import Path
from datetime import datetime
import sqlite3
import json

def get_latest_db(worker: str) -> Path:
    """Trouve la dernière DB Optuna pour un worker"""
    optuna_dir = Path("optuna_results")
    dbs = list(optuna_dir.glob(f"{worker}_ppo_*.db"))
    if not dbs:
        return None
    return sorted(dbs, key=lambda x: x.stat().st_mtime)[-1]

def get_trials_from_db(db_path: Path) -> list:
    """Récupère les trials depuis la DB Optuna"""
    try:
        conn = sqlite3.connect(f"sqlite:///{db_path}")
        cursor = conn.cursor()
        
        # Récupérer les trials
        cursor.execute("""
            SELECT trial_id, value, state FROM trials 
            ORDER BY trial_id DESC LIMIT 1
        """)
        
        latest = cursor.fetchone()
        conn.close()
        
        if latest:
            return {
                'trial_id': latest[0],
                'value': latest[1],
                'state': latest[2]
            }
        return None
    except Exception as e:
        return None

def read_log_tail(log_file: Path, lines: int = 10) -> str:
    """Lit les dernières lignes du fichier log"""
    try:
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            return ''.join(all_lines[-lines:])
    except:
        return ""

def monitor_worker(worker: str, duration_minutes: int = 120):
    """Monitore un worker pendant une durée donnée"""
    
    print(f"\n{'='*80}")
    print(f"🔍 MONITORING OPTUNA - {worker}")
    print(f"{'='*80}")
    print(f"Durée estimée : {duration_minutes} minutes")
    print(f"Démarrage : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    log_file = Path(f"optuna_results/{worker}_optimization.log")
    start_time = time.time()
    last_trial = -1
    
    while True:
        elapsed = (time.time() - start_time) / 60
        
        if elapsed > duration_minutes:
            print(f"\n⏱️  Timeout atteint ({duration_minutes} min)")
            break
        
        # Vérifier le log
        if log_file.exists():
            tail = read_log_tail(log_file, 5)
            
            # Chercher les trials complétés
            for line in tail.split('\n'):
                if 'Trial' in line and 'Score=' in line:
                    # Parser la ligne
                    try:
                        parts = line.split('Trial')[1].split(':')[0].strip()
                        trial_num = int(parts)
                        
                        if trial_num > last_trial:
                            last_trial = trial_num
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] {line.strip()}")
                    except:
                        pass
        
        # Afficher la progression
        progress = min(100, int((elapsed / duration_minutes) * 100))
        bar = '█' * (progress // 5) + '░' * (20 - progress // 5)
        print(f"\r[{bar}] {progress}% ({elapsed:.1f}/{duration_minutes} min)", end='', flush=True)
        
        time.sleep(10)  # Vérifier toutes les 10 secondes
    
    print(f"\n\n✅ Monitoring terminé")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        worker = sys.argv[1]
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    else:
        worker = "W1"
        duration = 120
    
    monitor_worker(worker, duration)
