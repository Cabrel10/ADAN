#!/usr/bin/env python3
"""
Monitoring continu de l'entraînement - Toutes les 10 minutes
Vérifie chaque worker et rapporte les performances
"""

import time
import subprocess
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Configuration
LOG_FILE = "/mnt/new_data/adan_logs/adan_training_final_20251210_025524.log"
MONITORING_INTERVAL = 600  # 10 minutes
MONITORING_LOG = "/mnt/new_data/adan_logs/monitoring_report.log"
MAX_ITERATIONS = 144  # 24 heures * 6 (10 min intervals)

def log_message(msg):
    """Log un message dans le fichier de monitoring"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)
    
    try:
        with open(MONITORING_LOG, 'a') as f:
            f.write(log_msg + '\n')
    except:
        pass

def get_training_status():
    """Vérifier si l'entraînement est en cours"""
    try:
        result = subprocess.run(
            "ps aux | grep 'train_parallel_agents.py' | grep -v grep | wc -l",
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        count = int(result.stdout.strip())
        return count > 0, count
    except:
        return False, 0

def get_log_size():
    """Obtenir la taille du log"""
    try:
        if os.path.exists(LOG_FILE):
            size_mb = os.path.getsize(LOG_FILE) / (1024 * 1024)
            return f"{size_mb:.1f}MB"
    except:
        pass
    return "N/A"

def get_worker_status(worker_id):
    """Extraire le statut d'un worker"""
    try:
        # Chercher les dernières lignes du worker
        result = subprocess.run(
            f"grep -i 'worker.*{worker_id}\\|{worker_id.upper()}' {LOG_FILE} 2>/dev/null | tail -5",
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        lines = result.stdout.split('\n')
        
        status = {
            'worker': worker_id,
            'active': False,
            'sharpe': 'N/A',
            'trades': 'N/A',
            'drawdown': 'N/A',
            'balance': 'N/A',
            'step': 'N/A'
        }
        
        for line in lines:
            if line.strip():
                status['active'] = True
                
                # Chercher Sharpe
                if 'sharpe' in line.lower():
                    match = re.search(r'sharpe[=:\s]+([0-9.-]+)', line, re.IGNORECASE)
                    if match:
                        status['sharpe'] = f"{float(match.group(1)):.2f}"
                
                # Chercher Trades
                if 'trade' in line.lower():
                    match = re.search(r'trade[s]?[=:\s]+(\d+)', line, re.IGNORECASE)
                    if match:
                        status['trades'] = match.group(1)
                
                # Chercher Step
                if 'step' in line.lower():
                    match = re.search(r'step[=:\s]+(\d+)', line, re.IGNORECASE)
                    if match:
                        status['step'] = match.group(1)
        
        return status
    except Exception as e:
        return {
            'worker': worker_id,
            'active': False,
            'error': str(e)[:30]
        }

def print_report(iteration):
    """Afficher un rapport de monitoring"""
    is_running, process_count = get_training_status()
    
    report = "\n" + "="*80
    report += f"\n📊 MONITORING ITERATION {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    report += "\n" + "="*80
    
    if is_running:
        report += f"\n✅ Entraînement EN COURS ({process_count} processus)"
    else:
        report += f"\n❌ Entraînement TERMINÉ ou ARRÊTÉ"
    
    report += f"\n📁 Log size: {get_log_size()}"
    report += "\n\n📈 STATUT PAR WORKER:"
    report += "\n" + "-"*80
    
    for worker_id in ['w1', 'w2', 'w3', 'w4']:
        status = get_worker_status(worker_id)
        
        if status.get('active'):
            report += f"\n{worker_id.upper()}: ✅ ACTIVE"
            report += f"\n  Sharpe: {status.get('sharpe', 'N/A')}"
            report += f"\n  Trades: {status.get('trades', 'N/A')}"
            report += f"\n  Step: {status.get('step', 'N/A')}"
            report += f"\n  Balance: {status.get('balance', 'N/A')}"
        else:
            report += f"\n{worker_id.upper()}: ⏳ Waiting or Inactive"
    
    report += "\n" + "="*80
    report += f"\n⏰ Prochain check dans 10 minutes..."
    report += "\n" + "="*80
    
    print(report)
    log_message(report)
    
    return is_running

def main():
    """Boucle principale de monitoring"""
    log_message("✅ Monitoring lancé")
    
    iteration = 0
    while iteration < MAX_ITERATIONS:
        iteration += 1
        
        try:
            is_running = print_report(iteration)
            
            if not is_running and iteration > 1:
                log_message("✅ Entraînement terminé, arrêt du monitoring")
                break
            
            # Attendre 10 minutes
            log_message(f"⏳ Attente 10 minutes (iteration {iteration}/{MAX_ITERATIONS})...")
            time.sleep(MONITORING_INTERVAL)
            
        except KeyboardInterrupt:
            log_message("⚠️ Monitoring arrêté par l'utilisateur")
            break
        except Exception as e:
            log_message(f"❌ Erreur: {str(e)}")
            time.sleep(60)  # Attendre 1 minute avant de réessayer
    
    log_message("✅ Monitoring terminé")

if __name__ == "__main__":
    main()
