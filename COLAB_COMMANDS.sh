#!/bin/bash

################################################################################
# 🚀 ADAN TRADING BOT - COLAB COMMANDS REFERENCE
#
# Ce fichier contient toutes les commandes utiles pour Colab
# Copiez-collez les commandes dans les cellules Colab
################################################################################

# ============================================================================
# CELLULE 1: INSTALLATION (5-10 min)
# ============================================================================

# Option A: Installation complète (recommandé)
!curl -sSL https://raw.githubusercontent.com/Cabrel10/ADAN0/main/setup_colab.sh | bash

# Option B: Installation manuelle
!apt-get update -qq
!apt-get install -y -qq build-essential wget curl git python3-dev python3-pip
!pip install -q numpy pandas torch gymnasium stable-baselines3 optuna pyyaml


# ============================================================================
# CELLULE 2: GOOGLE DRIVE (optionnel, 1 min)
# ============================================================================

from google.colab import drive
import os

drive.mount('/content/drive')
BACKUP_DIR = '/content/drive/MyDrive/ADAN_Training'
os.makedirs(BACKUP_DIR, exist_ok=True)
print(f"✅ Google Drive monté - Sauvegardes dans: {BACKUP_DIR}")


# ============================================================================
# CELLULE 3: LANCEMENT (1-2h pour 500k)
# ============================================================================

# Option A: 500k timesteps (défaut, 1-2h)
!cd /content/ADAN0 && bash launch_training.sh 500000

# Option B: 1M timesteps (2-4h)
!cd /content/ADAN0 && bash launch_training.sh 1000000

# Option C: 5M timesteps (8-12h)
!cd /content/ADAN0 && bash launch_training.sh 5000000

# Option D: 10M timesteps (24-48h, Colab Pro)
!cd /content/ADAN0 && bash launch_training.sh 10000000


# ============================================================================
# CELLULE 4: MONITORING EN TEMPS RÉEL
# ============================================================================

import subprocess
import time
import os
from IPython.display import clear_output
from datetime import datetime

print("📊 Monitoring des logs d'entraînement...")
print("(Appuyez sur le bouton Stop pour arrêter le monitoring)\n")

log_dir = "/content/ADAN0/logs"
log_files = sorted([f for f in os.listdir(log_dir) if f.startswith('training_')], reverse=True)

if not log_files:
    print("❌ Aucun fichier de log trouvé")
else:
    log_file = os.path.join(log_dir, log_files[0])
    
    try:
        while True:
            clear_output(wait=True)
            
            print(f"🕐 Mise à jour: {datetime.now().strftime('%H:%M:%S')}")
            print(f"📝 Fichier: {log_files[0]}\n")
            print("=" * 80)
            
            # Afficher les dernières lignes pertinentes
            result = subprocess.run(
                f"tail -50 {log_file} | grep -E 'DBE_DECISION|Episode|timestep|Portfolio|REWARD|Step' | tail -20",
                shell=True,
                capture_output=True,
                text=True
            )
            if result.stdout:
                print(result.stdout)
            else:
                print("⏳ En attente des premiers logs...")
            
            print("=" * 80)
            
            # Statistiques
            dbe_count = subprocess.run(
                f"grep -c 'DBE_DECISION' {log_file}",
                shell=True,
                capture_output=True,
                text=True
            ).stdout.strip()
            
            regime_count = subprocess.run(
                f"grep -c 'REGIME_DETECTION' {log_file}",
                shell=True,
                capture_output=True,
                text=True
            ).stdout.strip()
            
            print(f"\n📈 Statistiques:")
            print(f"   - Décisions DBE: {dbe_count}")
            print(f"   - Détections régime: {regime_count}")
            print(f"\n⏳ Rafraîchissement dans 10 secondes... (Ctrl+C pour arrêter)")
            
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n⏸️  Monitoring arrêté (l'entraînement continue)")


# ============================================================================
# CELLULE 5: SAUVEGARDE SUR GOOGLE DRIVE
# ============================================================================

import shutil
import os
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = f"/content/drive/MyDrive/ADAN_Training/run_{timestamp}"
os.makedirs(backup_path, exist_ok=True)

print("💾 Sauvegarde des résultats...\n")

# Checkpoints
if os.path.exists("/content/ADAN0/checkpoints"):
    shutil.copytree("/content/ADAN0/checkpoints", 
                    f"{backup_path}/checkpoints", 
                    dirs_exist_ok=True)
    print("✅ Checkpoints sauvegardés")

# Logs
if os.path.exists("/content/ADAN0/logs"):
    shutil.copytree("/content/ADAN0/logs", 
                    f"{backup_path}/logs", 
                    dirs_exist_ok=True)
    print("✅ Logs sauvegardés")

# Résultats
if os.path.exists("/content/ADAN0/results"):
    shutil.copytree("/content/ADAN0/results", 
                    f"{backup_path}/results", 
                    dirs_exist_ok=True)
    print("✅ Résultats sauvegardés")

print(f"\n✅ Sauvegarde complète!")
print(f"📁 Emplacement: {backup_path}")


# ============================================================================
# CELLULE 6: ANALYSE DES RÉSULTATS
# ============================================================================

import subprocess
import os

log_dir = "/content/ADAN0/logs"
log_files = sorted([f for f in os.listdir(log_dir) if f.startswith('training_')], reverse=True)

if not log_files:
    print("❌ Aucun fichier de log trouvé")
else:
    log_file = os.path.join(log_dir, log_files[0])
    
    print("📊 Analyse des résultats d'entraînement\n")
    print("=" * 80)
    
    # Décisions par worker
    workers = {
        'Trial26 Ultra-Stable': 'w0',
        'Moderate Optimized': 'w1',
        'Sharpe Optimized': 'w2',
        'Aggressive Optimized': 'w3'
    }
    
    print("\n🤖 Décisions par Worker:\n")
    for worker_name, worker_id in workers.items():
        result = subprocess.run(
            f"grep -c '{worker_name}' {log_file}",
            shell=True,
            capture_output=True,
            text=True
        )
        count = result.stdout.strip()
        print(f"   ✅ {worker_name:25s} ({worker_id}): {count:>6s} décisions")
    
    print("\n" + "=" * 80)
    print("\n📈 Statistiques Globales:\n")
    
    dbe_result = subprocess.run(
        f"grep -c 'DBE_DECISION' {log_file}",
        shell=True,
        capture_output=True,
        text=True
    )
    dbe_count = dbe_result.stdout.strip()
    
    regime_result = subprocess.run(
        f"grep -c 'REGIME_DETECTION' {log_file}",
        shell=True,
        capture_output=True,
        text=True
    )
    regime_count = regime_result.stdout.strip()
    
    lines_result = subprocess.run(
        f"wc -l < {log_file}",
        shell=True,
        capture_output=True,
        text=True
    )
    lines_count = lines_result.stdout.strip()
    
    error_result = subprocess.run(
        f"grep -i 'error\\|exception' {log_file} | wc -l",
        shell=True,
        capture_output=True,
        text=True
    )
    error_count = error_result.stdout.strip()
    
    print(f"   - Décisions DBE: {dbe_count}")
    print(f"   - Détections régime: {regime_count}")
    print(f"   - Lignes de log: {lines_count}")
    print(f"   - Erreurs: {error_count}")
    
    print("\n" + "=" * 80)
    print("\n📋 Dernières lignes du log:\n")
    
    tail_result = subprocess.run(
        f"tail -10 {log_file}",
        shell=True,
        capture_output=True,
        text=True
    )
    
    for line in tail_result.stdout.split('\n')[-10:]:
        if line:
            print(f"   {line}")
    
    print("\n" + "=" * 80)
    print("\n✅ Analyse terminée!")


# ============================================================================
# COMMANDES UTILES
# ============================================================================

# Voir les logs en temps réel
!tail -f /content/ADAN0/logs/training_*.log

# Compter les décisions DBE
!grep -c "[DBE_DECISION]" /content/ADAN0/logs/training_*.log

# Chercher les erreurs
!grep -i "error\|exception" /content/ADAN0/logs/training_*.log | head -20

# Vérifier les données
!find /content/ADAN0/data -name "*.parquet" | wc -l

# Vérifier l'espace disque
!df -h /content/

# Vérifier la mémoire
!free -h

# Vérifier les ressources CPU
!nproc

# Lister les fichiers de checkpoint
!ls -lh /content/ADAN0/checkpoints/

# Lister les fichiers de log
!ls -lh /content/ADAN0/logs/

# Afficher la configuration
!cat /content/ADAN0/config/config.yaml | head -30

# Tester les imports
!python3 -c "import torch; import gymnasium; from stable_baselines3 import PPO; print('✅ Tous les imports OK')"

# Arrêter l'entraînement
!pkill -f train_parallel_agents.py

# Compter les fichiers parquet
!find /content/ADAN0/data -name "*.parquet" -exec ls -lh {} \; | wc -l

# Calculer la taille totale des données
!du -sh /content/ADAN0/data/

# Afficher les dernières décisions DBE
!grep "[DBE_DECISION]" /content/ADAN0/logs/training_*.log | tail -20

# Afficher les dernières détections de régime
!grep "[REGIME_DETECTION]" /content/ADAN0/logs/training_*.log | tail -20

# Afficher les dernières positions fermées
!grep "[POSITION FERMÉE]" /content/ADAN0/logs/training_*.log | tail -20

# Afficher les dernières récompenses
!grep "[REWARD]" /content/ADAN0/logs/training_*.log | tail -20


# ============================================================================
# DÉPANNAGE
# ============================================================================

# Vérifier que TA-Lib est installé
!python3 -c "import talib; print('✅ TA-Lib OK')"

# Vérifier que PyTorch est installé
!python3 -c "import torch; print(f'✅ PyTorch {torch.__version__} OK')"

# Vérifier que Stable-Baselines3 est installé
!python3 -c "from stable_baselines3 import PPO; print('✅ Stable-Baselines3 OK')"

# Vérifier que Optuna est installé
!python3 -c "import optuna; print(f'✅ Optuna {optuna.__version__} OK')"

# Vérifier que les données sont présentes
!find /content/ADAN0/data/processed/indicators -name "*.parquet" | head -10

# Vérifier la configuration
!python3 -c "import yaml; print(yaml.safe_load(open('/content/ADAN0/config/config.yaml'))['training'])"

# Nettoyer les fichiers temporaires
!rm -f /content/ADAN0/dbe_state_*.pkl
!rm -rf /content/ADAN0/results/

# Réinitialiser les logs
!rm -f /content/ADAN0/logs/training_*.log

################################################################################
# FIN DES COMMANDES
################################################################################
