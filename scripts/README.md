# Scripts d'Exécution ADAN

Ce répertoire contient tous les scripts autonomes pour l'exécution complète du pipeline ADAN, de la collecte des données à l'entraînement de l'agent RL.

## 🚀 Guide de Référence Rapide

**⚠️ IMPORTANT :** Toujours utiliser les variables d'environnement pour éviter les problèmes d'affichage Rich :
```bash
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/[SCRIPT].py
```

## 📋 Scripts Principaux

### 🔄 Pipeline de Données

#### `fetch_data_ccxt.py`
Collecte les données historiques OHLCV depuis Binance via CCXT.

```bash
# CPU (données réduites pour tests)
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/fetch_data_ccxt.py --exec_profile cpu

# GPU (données maximisées pour production)
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/fetch_data_ccxt.py --exec_profile gpu
```

**Sorties :** `data/raw/[ASSET]_[TIMEFRAME]_raw.parquet` (15 fichiers total)

#### `process_data.py`
Traite les données brutes en calculant les indicateurs techniques spécifiques par timeframe.

```bash
# Traitement avec indicateurs par timeframe
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/process_data.py --exec_profile cpu
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/process_data.py --exec_profile gpu
```

**Fonctionnalités :**
- Indicateurs spécifiques par timeframe (Scalping 1m, Day Trading 1h, Swing Trading 1d)
- Normalisation des features (sauf OHLC)
- Suffixes de timeframe automatiques (`_1m`, `_1h`, `_1d`)

**Sorties :** `data/processed/[ASSET]/[ASSET]_[TIMEFRAME]_[SPLIT].parquet`

#### `merge_processed_data.py`
Fusionne les données multi-assets en un seul DataFrame avec 95 colonnes.

```bash
# Fusion multi-assets
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/merge_processed_data.py --exec_profile cpu
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/merge_processed_data.py --exec_profile gpu
```

**Structure finale :** 19 features × 5 assets = 95 colonnes
**Sorties :** `data/processed/merged/[TIMEFRAME]_[SPLIT]_merged.parquet`

### 🤖 Entraînement RL

#### `train_rl_agent.py`
Script principal d'entraînement de l'agent PPO avec CNN Feature Extractor.

```bash
# Configuration CPU (200k timesteps, 1 worker)
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/train_rl_agent.py --exec_profile cpu --device cpu --initial_capital 15 --total_timesteps 200000

# Configuration GPU (2M timesteps, 8 workers)
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/train_rl_agent.py --exec_profile gpu --device cuda --initial_capital 15 --total_timesteps 2000000

# Test de validation rapide
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/train_rl_agent.py --exec_profile cpu --device cpu --initial_capital 15 --total_timesteps 100
```

**Arguments disponibles :**
- `--exec_profile` : cpu/gpu (détermine les configs chargées)
- `--device` : cpu/cuda (device PyTorch)
- `--initial_capital` : Capital de départ (outrepasse config)
- `--total_timesteps` : Nombre de pas d'entraînement (outrepasse config)
- `--learning_rate` : Taux d'apprentissage (optionnel)

**Sorties :**
- `models/final_model.zip` : Modèle final
- `models/interrupted_model.zip` : Sauvegarde interruption
- `models/checkpoints/` : Sauvegardes périodiques
- `reports/tensorboard_logs/` : Métriques TensorBoard

### 🔍 Tests et Validation

#### `test_environment_with_merged_data.py`
Teste l'environnement de trading avec les données fusionnées.

```bash
# Test environnement complet
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/test_environment_with_merged_data.py --exec_profile cpu --initial_capital 10000 --timeframe 1h --split train --num_rows 100

# Test avec différents paramètres
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/test_environment_with_merged_data.py --exec_profile gpu --initial_capital 50000 --timeframe 1h --split val --num_rows 500
```

**Arguments :**
- `--exec_profile` : cpu/gpu
- `--initial_capital` : Capital de test
- `--timeframe` : 1m/1h/1d
- `--split` : train/val/test
- `--num_rows` : Nombre de lignes à tester

#### `test_exec_profiles.py`
Valide les configurations des profils CPU/GPU.

```bash
# Vérification profils
python scripts/test_exec_profiles.py --exec_profile cpu
python scripts/test_exec_profiles.py --exec_profile gpu
```

#### `test_trading_actions.py`
Tests spécifiques des actions de trading.

```bash
python scripts/test_trading_actions.py
```

### 📊 Évaluation (En Développement)

#### `evaluate_agent.py`
Backtesting avancé d'un agent entraîné.

```bash
# Évaluation d'un modèle (à venir)
python scripts/evaluate_agent.py --exec_profile cpu --model_path models/final_model.zip --timeframe 1h --split test
```

## 🔧 Profils d'Exécution

### Profil CPU (`--exec_profile cpu`)
**Configurations chargées :**
- `config/data_config_cpu.yaml`
- `config/agent_config_cpu.yaml`

**Optimisations :**
- Timesteps : 200,000
- Workers : 1
- Batch size : 64
- CNN window : 15
- Périodes de données réduites

### Profil GPU (`--exec_profile gpu`)
**Configurations chargées :**
- `config/data_config_gpu.yaml`
- `config/agent_config_gpu.yaml`

**Optimisations :**
- Timesteps : 2,000,000
- Workers : 8
- Batch size : 512
- CNN window : 30
- Périodes de données maximisées

## 📈 Pipeline Complet

### Exécution CPU Complète
```bash
conda activate trading_env
cd ADAN

# 1. Collecte (5-10 min)
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/fetch_data_ccxt.py --exec_profile cpu

# 2. Traitement (2-5 min)
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/process_data.py --exec_profile cpu

# 3. Fusion (1-2 min)
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/merge_processed_data.py --exec_profile cpu

# 4. Validation (1-2 min)
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/train_rl_agent.py --exec_profile cpu --device cpu --initial_capital 15 --total_timesteps 100

# 5. Entraînement (2-4h)
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/train_rl_agent.py --exec_profile cpu --device cpu --initial_capital 15 --total_timesteps 200000
```

### Exécution GPU Production
```bash
conda activate trading_env
cd ADAN

# Pipeline données étendues (20-40 min total)
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/fetch_data_ccxt.py --exec_profile gpu
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/process_data.py --exec_profile gpu
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/merge_processed_data.py --exec_profile gpu

# Test validation GPU (2-3 min)
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/train_rl_agent.py --exec_profile gpu --device cuda --initial_capital 15 --total_timesteps 500

# Entraînement intensif (8-12h)
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/train_rl_agent.py --exec_profile gpu --device cuda --initial_capital 15 --total_timesteps 2000000
```

## 🔍 Monitoring Parallèle

### Terminal 1 : Entraînement
```bash
TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python scripts/train_rl_agent.py --exec_profile gpu --device cuda --initial_capital 15 --total_timesteps 2000000
```

### Terminal 2 : GPU Monitoring
```bash
watch -n 1 nvidia-smi
```

### Terminal 3 : TensorBoard
```bash
tensorboard --logdir reports/tensorboard_logs --host 0.0.0.0 --port 6006
```

## 🚨 Troubleshooting

### Erreur "FEATURE NON TROUVÉE"
1. Vérifier `base_market_features` dans `config/data_config_*.yaml`
2. Régénérer : `process_data.py` puis `merge_processed_data.py`

### Erreur Rich (affichage corrompu)
- **Solution :** Toujours utiliser `TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1`

### Erreur GPU/CUDA
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Erreur de mémoire GPU
- Réduire `batch_size` dans `config/agent_config_gpu.yaml`
- Réduire `n_steps` ou `cnn_input_window_size`

## ✅ Vérifications Importantes

### Après fetch_data_ccxt.py
```bash
ls -la data/raw/ | wc -l  # Doit être 16 (15 fichiers + . et ..)
```

### Après process_data.py
```bash
python -c "import pandas as pd; df = pd.read_parquet('data/processed/DOGEUSDT/DOGEUSDT_1h_train.parquet'); print(f'Shape: {df.shape}')"  # Doit être (X, 19)
```

### Après merge_processed_data.py
```bash
python -c "import pandas as pd; df = pd.read_parquet('data/processed/merged/1h_train_merged.parquet'); print(f'Shape: {df.shape}')"  # Doit être (X, 95)
```

## 📝 Notes Importantes

- **Rich Fix :** Les variables d'environnement `TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1` sont **obligatoires**
- **Interruption :** Ctrl+C sauvegarde automatiquement le modèle
- **Reproductibilité :** Seed=42 fixé dans toutes les configurations
- **Validation :** Toujours faire un test avec `--total_timesteps 100` avant l'entraînement long

---

**📖 Pour un guide complet étape par étape, consultez [../execution.md](../execution.md)**