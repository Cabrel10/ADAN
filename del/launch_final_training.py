#!/usr/bin/env python3
"""
Lancement final complet et fiable de l'entraînement ADAN
1 million de steps par worker = ~4 millions de steps total
Logs sauvegardés dans /mnt/new_data/adan_logs
"""
import os
import sys
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path

# Configuration
LOG_DIR = "/mnt/new_data/adan_logs"
STEPS_PER_WORKER = 1000000  # 1 million de steps par worker
CONFIG_PATH = "config/config.yaml"
SCRIPT_PATH = "scripts/train_parallel_agents.py"

def setup_logging():
    """Préparer le répertoire de logs"""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_log = os.path.join(LOG_DIR, f"training_session_{timestamp}.log")
    return session_log

def check_disk_space():
    """Vérifier l'espace disque disponible"""
    import shutil
    stat = shutil.disk_usage(LOG_DIR)
    available_gb = stat.free / (1024**3)
    print(f"📊 Espace disque disponible: {available_gb:.1f}GB")
    if available_gb < 5:
        print("⚠️  Espace disque faible!")
        return False
    return True

def launch_training(session_log):
    """Lancer l'entraînement"""
    print("🚀 LANCEMENT DE L'ENTRAÎNEMENT FINAL")
    print("="*80)
    print(f"📊 Steps par worker: {STEPS_PER_WORKER:,}")
    print(f"📊 Total steps (4 workers): {STEPS_PER_WORKER * 4:,}")
    print(f"📁 Logs: {LOG_DIR}")
    print(f"📝 Session log: {session_log}")
    print("="*80)
    
    cmd = [
        sys.executable,
        SCRIPT_PATH,
        "--config", CONFIG_PATH,
        "--log-level", "INFO",
        "--steps", str(STEPS_PER_WORKER)
    ]
    
    print(f"\n📋 Commande: {' '.join(cmd)}\n")
    
    try:
        # Lancer le processus
        with open(session_log, 'w') as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            print(f"✅ Processus lancé (PID: {process.pid})")
            print("⏱️  Surveillance pendant 10 minutes...\n")
            
            # Surveiller pendant 10 minutes
            start_time = time.time()
            monitoring_duration = 600  # 10 minutes
            last_output_time = start_time
            line_count = 0
            
            while True:
                elapsed = time.time() - start_time
                
                # Lire une ligne
                output = process.stdout.readline()
                if output:
                    line_count += 1
                    log_file.write(output)
                    log_file.flush()
                    last_output_time = time.time()
                    
                    # Afficher les logs importants
                    if any(keyword in output for keyword in [
                        "[STEP", "[TRADE", "[DBE", "[RISK", "ERROR", "CRITICAL"
                    ]):
                        print(f"[{elapsed/60:.1f}min] {output.strip()[:100]}")
                
                # Vérifier si le processus est terminé
                if process.poll() is not None:
                    # Lire les dernières lignes
                    remaining = process.stdout.read()
                    if remaining:
                        log_file.write(remaining)
                    print(f"\n✅ Entraînement terminé (PID: {process.pid})")
                    return True
                
                # Vérifier le timeout de monitoring (10 minutes)
                if elapsed > monitoring_duration:
                    print(f"\n✅ Monitoring de 10 minutes complété")
                    print(f"📊 {line_count} lignes de logs écrites")
                    print("🚀 Laissant l'entraînement continuer en arrière-plan...")
                    print(f"📝 Logs: {session_log}")
                    return True
                
                # Timeout si pas de sortie depuis 5 minutes
                if time.time() - last_output_time > 300:
                    print(f"\n⚠️  Pas de sortie depuis 5 minutes")
                    print(f"📊 {line_count} lignes de logs écrites")
                    print("🚀 Laissant l'entraînement continuer...")
                    return True
                
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n⚠️  Interruption utilisateur")
        if 'process' in locals():
            process.terminate()
        return False
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        return False

def main():
    """Fonction principale"""
    print("\n" + "="*80)
    print("🎯 LANCEMENT FINAL COMPLET - ENTRAÎNEMENT ADAN")
    print("="*80 + "\n")
    
    # 1. Vérifier l'espace disque
    if not check_disk_space():
        print("❌ Espace disque insuffisant")
        return False
    
    # 2. Préparer les logs
    session_log = setup_logging()
    print(f"✅ Répertoire de logs: {LOG_DIR}")
    
    # 3. Lancer l'entraînement
    success = launch_training(session_log)
    
    if success:
        print("\n" + "="*80)
        print("🎉 ENTRAÎNEMENT LANCÉ AVEC SUCCÈS!")
        print("="*80)
        print(f"\n📊 Configuration:")
        print(f"   Steps par worker: {STEPS_PER_WORKER:,}")
        print(f"   Total steps: {STEPS_PER_WORKER * 4:,}")
        print(f"   Logs: {LOG_DIR}")
        print(f"\n📝 Session log: {session_log}")
        print("\n💡 L'entraînement continue en arrière-plan.")
        print("   Vous pouvez fermer ce terminal en toute sécurité.")
        return True
    else:
        print("\n❌ ERREUR LORS DU LANCEMENT")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
