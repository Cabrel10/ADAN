# Guide d'Exécution Complet - ADAN Trading Bot

## 📋 Table des Matières

1. [Prérequis](#prérequis)
2. [Installation et Configuration](#installation-et-configuration)
3. [Pipeline d'Exécution CPU](#pipeline-dexécution-cpu)
4. [Pipeline d'Exécution GPU](#pipeline-dexécution-gpu)
5. [Vérification et Validation](#vérification-et-validation)
6. [Troubleshooting](#troubleshooting)
7. [Monitoring et Analyse](#monitoring-et-analyse)

---

## 🔧 Prérequis

### Système Requis
- **CPU** : Ubuntu 20.04+ ou équivalent, 8GB+ RAM, Python 3.11
- **GPU** : Ubuntu 20.04+, 32GB+ RAM, GTX 1650+, CUDA 11.8+, Python 3.11

### Dépendances
```bash
# Installation de Miniconda (si nécessaire)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Initialisation de conda
source ~/.bashrc
conda init
```

---

## 🛠️ Installation et Configuration

### 1. Clone et Setup Initial
```bash
# Cloner le projet
git clone <ADAN_REPOSITORY_URL>
cd ADAN

# Créer l'environnement conda
conda create -n trading_env python=3.11 -y
conda activate trading_env

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Configuration Spécifique GPU (Uniquement pour GPU)
```bash
# Vérifier CUDA
nvidia-smi
nvcc --version

# Installer PyTorch avec support CUDA (ajuster selon votre version CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Vérifier l'installation GPU
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Aucun\"}')"
```

### 3. Vérification de l'Installation
```bash
# Test des imports critiques
python -c "
import pandas as pd
import numpy as np
import gymnasium as gym
import stable_baselines3 as sb3
import ccxt
import pandas_ta as ta
print('✅ Toutes les dépendances sont installées correctement')
"
```

---

## 💻 Pipeline d'Exécution CPU

### Étape 1 : Nettoyage (Optionnel)
```bash
# Activer l'environnement
conda activate trading_env
cd ADAN

# Nettoyage complet (optionnel pour nouveau démarrage)
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
rm -rf data/processed/* data/scalers_encoders/* 2>/dev/null || true
```

### Étape 2 : Collecte des Données
```bash
# Télécharger les données de marché
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/fetch_data_ccxt.py --exec_profile cpu
```
**Durée estimée :** 5-10 minutes  
**Vérification :** `ls -la data/raw/` doit contenir 15 fichiers .parquet (5 assets × 3 timeframes)

### Étape 3 : Traitement des Données
```bash
# Traiter les données (calcul des indicateurs techniques)
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/process_data.py --exec_profile cpu
```
**Durée estimée :** 2-5 minutes  
**Vérification :** `ls -la data/processed/` doit contenir 5 dossiers d'assets

### Étape 4 : Fusion des Données
```bash
# Fusionner les données multi-assets
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/merge_processed_data.py --exec_profile cpu
```
**Durée estimée :** 1-2 minutes  
**Vérification :** `ls -la data/processed/merged/` doit contenir 12 fichiers .parquet

### Étape 5 : Validation du Pipeline
```bash
# Test court pour valider le pipeline
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/train_rl_agent.py --exec_profile cpu --device cpu --initial_capital 15 --total_timesteps 100
```
**Durée estimée :** 1-2 minutes  
**Vérification :** Aucune erreur, logs indiquent "Training completed successfully!"

### Étape 6 : Entraînement Principal CPU
```bash
# Entraînement de production CPU
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/train_rl_agent.py --exec_profile cpu --device cpu --initial_capital 15 --total_timesteps 200000
```
**Durée estimée :** 2-4 heures  
**Fichiers générés :** `models/final_model.zip`, logs dans `reports/tensorboard_logs/`

---

## 🚀 Pipeline d'Exécution GPU

### Étape 1 : Nettoyage et Préparation GPU
```bash
# Activer l'environnement
conda activate trading_env
cd ADAN

# Nettoyage complet (recommandé pour GPU)
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
rm -rf data/processed/* data/scalers_encoders/* 2>/dev/null || true

# Vérification GPU finale
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Étape 2 : Collecte des Données GPU (Maximisée)
```bash
# Télécharger les données étendues pour GPU
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/fetch_data_ccxt.py --exec_profile gpu
```
**Durée estimée :** 15-30 minutes (plus de données historiques)  
**Vérification :** `ls -la data/raw/` avec fichiers plus volumineux

### Étape 3 : Traitement GPU
```bash
# Traitement optimisé GPU
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/process_data.py --exec_profile gpu
```
**Durée estimée :** 5-10 minutes  
**Vérification :** Vérifier l'absence d'erreurs dans les logs

### Étape 4 : Fusion GPU
```bash
# Fusion pour configuration GPU
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/merge_processed_data.py --exec_profile gpu
```
**Durée estimée :** 2-5 minutes  
**Vérification :** Fichiers plus volumineux dans `data/processed/merged/`

### Étape 5 : Test de Validation GPU
```bash
# Test court GPU
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/train_rl_agent.py --exec_profile gpu --device cuda --initial_capital 15 --total_timesteps 500
```
**Durée estimée :** 2-3 minutes  
**Vérification :** Utilisation GPU visible avec `nvidia-smi`

### Étape 6 : Entraînement Principal GPU (Production)
```bash
# Entraînement intensif GPU - 2M timesteps avec 8 workers
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/train_rl_agent.py --exec_profile gpu --device cuda --initial_capital 15 --total_timesteps 2000000
```
**Durée estimée :** 8-12 heures  
**Configuration :** 8 environnements parallèles, batch_size=512, fenêtre CNN=30

---

## ✅ Vérification et Validation

### Vérifications Critiques à Chaque Étape

#### Après Collecte des Données
```bash
# Vérifier les fichiers raw
python -c "
import pandas as pd
import os
assets = ['DOGEUSDT', 'XRPUSDT', 'LTCUSDT', 'SOLUSDT', 'ADAUSDT']
timeframes = ['1m', '1h', '1d']
for asset in assets:
    for tf in timeframes:
        file = f'data/raw/{asset}_{tf}_raw.parquet'
        if os.path.exists(file):
            df = pd.read_parquet(file)
            print(f'✅ {file}: {len(df)} lignes')
        else:
            print(f'❌ {file}: MANQUANT')
"
```

#### Après Traitement
```bash
# Vérifier la structure des données traitées
python -c "
import pandas as pd
import os
# Test sur un fichier traité
df = pd.read_parquet('data/processed/DOGEUSDT/DOGEUSDT_1h_train.parquet')
print(f'Shape: {df.shape}')
print(f'Colonnes: {list(df.columns)}')
# Doit avoir 19 colonnes : 5 OHLCV + 14 indicateurs
assert df.shape[1] == 19, f'Expected 19 columns, got {df.shape[1]}'
print('✅ Structure correcte')
"
```

#### Après Fusion
```bash
# Vérifier la fusion multi-assets
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/merged/1h_train_merged.parquet')
print(f'Shape: {df.shape}')
print(f'Colonnes (30 premières): {list(df.columns[:30])}')
# Doit avoir 95 colonnes : 19 features × 5 assets
assert df.shape[1] == 95, f'Expected 95 columns, got {df.shape[1]}'
print('✅ Fusion correcte - 95 colonnes')
"
```

### Validation StateBuilder (Critical)
```bash
# Test crucial - vérifier que StateBuilder trouve toutes les features
python -c "
import sys
sys.path.append('src')
from adan_trading_bot.common.utils import load_config, get_path
import pandas as pd
import os

# Charger la config
data_config = load_config(os.path.join(get_path('config'), 'data_config_cpu.yaml'))
base_features = data_config['base_market_features']
assets = data_config['assets']

# Charger données fusionnées
df = pd.read_parquet('data/processed/merged/1h_train_merged.parquet')

# Vérifier chaque feature pour chaque asset
missing_features = []
for feature in base_features:
    for asset in assets:
        column_name = f'{feature}_{asset}'
        if column_name not in df.columns:
            missing_features.append(column_name)

if missing_features:
    print(f'❌ Features manquantes: {missing_features}')
    exit(1)
else:
    print(f'✅ Toutes les {len(base_features) * len(assets)} features trouvées')
"
```

---

## 🚨 Troubleshooting

### Problèmes Courants

#### 1. Erreur "FEATURE NON TROUVÉE"
```bash
# Diagnostic
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/DOGEUSDT/DOGEUSDT_1h_train.parquet')
print('Colonnes dans le fichier traité:')
for i, col in enumerate(df.columns):
    print(f'{i+1}. {col}')
"

# Solution : Vérifier base_market_features dans config/data_config_cpu.yaml
```

#### 2. Erreur CUDA/GPU
```bash
# Diagnostic GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Si False, réinstaller PyTorch avec CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Erreur Rich (affichage corrompu)
```bash
# Solution : Toujours utiliser les variables d'environnement
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/...
```

#### 4. Erreur de mémoire GPU
```bash
# Réduire batch_size dans config/agent_config_gpu.yaml
# batch_size: 512 -> 256 -> 128
# ou réduire n_steps: 4096 -> 2048
```

#### 5. Problème de chemins
```bash
# Test des chemins
python -c "
import sys
sys.path.append('src')
from adan_trading_bot.common.utils import get_project_root, get_path
print(f'Root: {get_project_root()}')
print(f'Data: {get_path(\"data\")}')
"
```

---

## 📊 Monitoring et Analyse

### Monitoring en Temps Réel

#### Terminal 1 : Entraînement
```bash
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/train_rl_agent.py --exec_profile gpu --device cuda --initial_capital 15 --total_timesteps 2000000
```

#### Terminal 2 : Monitoring GPU (si GPU)
```bash
watch -n 1 nvidia-smi
```

#### Terminal 3 : TensorBoard
```bash
conda activate trading_env
cd ADAN
tensorboard --logdir reports/tensorboard_logs --host 0.0.0.0 --port 6006
```
**Accès :** http://localhost:6006

### Métriques Clés à Surveiller

#### TensorBoard
- **rollout/ep_rew_mean** : Récompense moyenne par épisode (doit augmenter)
- **rollout/ep_len_mean** : Durée moyenne des épisodes
- **train/policy_loss** : Perte de la politique (doit diminuer)
- **train/value_loss** : Perte de la fonction de valeur
- **train/entropy_loss** : Entropie (exploration vs exploitation)

#### Logs de Trading
- **Capital Evolution** : Progression du capital
- **Positions Prises** : Fréquence et types d'ordres
- **PnL par Trade** : Performance des trades individuels

### Analyse Post-Entraînement
```bash
# Vérifier les modèles générés
ls -la models/
# Fichiers attendus : final_model.zip, checkpoints/, best_model/

# Analyser les logs
tail -100 training_log.txt

# Évaluer les performances
python -c "
import os
model_files = [f for f in os.listdir('models/') if f.endswith('.zip')]
print(f'Modèles générés: {model_files}')
"
```

---

## 🎯 Commandes de Référence Rapide

### CPU - Pipeline Complet
```bash
conda activate trading_env && cd ADAN
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/fetch_data_ccxt.py --exec_profile cpu
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/process_data.py --exec_profile cpu
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/merge_processed_data.py --exec_profile cpu
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/train_rl_agent.py --exec_profile cpu --device cpu --initial_capital 15 --total_timesteps 200000
```

### GPU - Pipeline Complet
```bash
conda activate trading_env && cd ADAN
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/fetch_data_ccxt.py --exec_profile gpu
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/process_data.py --exec_profile gpu
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/merge_processed_data.py --exec_profile gpu
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/train_rl_agent.py --exec_profile gpu --device cuda --initial_capital 15 --total_timesteps 2000000
```

### Test Rapide (Validation)
```bash
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/train_rl_agent.py --exec_profile cpu --device cpu --initial_capital 15 --total_timesteps 100
```

---

## 📝 Notes Importantes

- **TOUJOURS** utiliser `TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1` pour éviter les problèmes d'affichage Rich
- **GPU** : Surveiller la VRAM avec `nvidia-smi` pendant l'entraînement
- **Interruption** : Ctrl+C sauvegarde automatiquement le modèle (`interrupted_model.zip`)
- **Reproductibilité** : Seed fixé à 42 dans les configurations
- **Checkpoints** : Sauvegarde automatique tous les 50k (CPU) / 200k (GPU) timesteps

---

**✅ En suivant ce guide, vous devriez obtenir un agent ADAN fonctionnel entraîné sur des données multi-assets avec indicateurs techniques spécifiques par timeframe.**