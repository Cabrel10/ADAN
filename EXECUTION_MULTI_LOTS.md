# Guide d'Exécution Multi-Lots - ADAN Trading Bot

## 📋 Vue d'Ensemble

ADAN supporte maintenant deux lots de données distincts avec des caractéristiques différentes :

- **Lot 1 (DOGE & Co)** : 5 actifs, timeframe 1h, ~18 features standard
- **Lot 2 (BTC & Co)** : 5 actifs, timeframe 1m, 47 features avancées

## 🎯 Profils d'Exécution Disponibles

### Profils de Données
- `cpu_lot1` / `gpu_lot1` : Lot DOGE & Co (1h, 18 features)
- `cpu_lot2` / `gpu_lot2` : Lot BTC & Co (1m, 47 features)

### Profils d'Agent
- `cpu` : 1 environnement, configurations légères
- `gpu` : 8 environnements parallèles, configurations optimisées

## 🚀 Installation et Prérequis

### Environnement de Base
```bash
# Activer l'environnement
conda activate trading_env
cd ADAN

# Vérifier l'installation
python -c "import torch, stable_baselines3, gymnasium; print('✅ Dépendances OK')"
```

### Vérification GPU (optionnel)
```bash
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
nvidia-smi  # Vérifier la mémoire GPU
```

## 🔄 Pipeline Complet - Lot 1 (DOGE & Co)

### Caractéristiques Lot 1
- **Actifs** : ADAUSDT, DOGEUSDT, LTCUSDT, SOLUSDT, XRPUSDT
- **Timeframe** : 1h (day trading)
- **Features** : ~17 (OHLCV + indicateurs avec suffixes automatiques)
- **Configuration** : Features construites dynamiquement via `indicators_by_timeframe`
- **Volume** : ~10k lignes
- **Durée d'entraînement** : 2-4h (CPU), 1-2h (GPU)

### Configuration des Features - Lot 1
Le Lot 1 utilise une **construction dynamique** des features :
- OHLCV de base : `open`, `high`, `low`, `close`, `volume`
- Indicateurs définis dans `indicators_by_timeframe["1h"]`
- Suffixes de timeframe ajoutés automatiquement (ex: `rsi_14_1h`)
- `base_market_features` n'est **pas utilisé** pour le Lot 1

### CPU - Lot 1
```bash
# 1. Traitement des données (génère les indicateurs avec suffixes)
python scripts/process_data.py --exec_profile cpu_lot1

# 2. Fusion des données
python scripts/merge_processed_data.py --exec_profile cpu_lot1

# 3. Test rapide (validation)
python scripts/train_rl_agent.py --exec_profile cpu_lot1 --device cpu --initial_capital 15 --total_timesteps 100

# 4. Entraînement complet
python scripts/train_rl_agent.py --exec_profile cpu_lot1 --device cpu --initial_capital 15 --total_timesteps 200000
```

### GPU - Lot 1 (8 environnements parallèles)
```bash
# Pipeline complet GPU
python scripts/process_data.py --exec_profile gpu_lot1
python scripts/merge_processed_data.py --exec_profile gpu_lot1
python scripts/train_rl_agent.py --exec_profile gpu_lot1 --device cuda --initial_capital 15 --total_timesteps 2000000
```

## 🔄 Pipeline Complet - Lot 2 (BTC & Co)

### Caractéristiques Lot 2
- **Actifs** : ADAUSDT, BNBUSDT, BTCUSDT, ETHUSDT, XRPUSDT
- **Timeframe** : 1m (scalping)
- **Features** : 47 (OHLCV + 42 indicateurs pré-calculés)
- **Configuration** : Features listées dans `base_market_features`
- **Volume** : ~500k lignes
- **Durée d'entraînement** : 4-8h (CPU), 2-4h (GPU)

### Configuration des Features - Lot 2
Le Lot 2 utilise des **features pré-calculées** :
- 47 features listées exactement dans `base_market_features`
- Noms correspondent aux colonnes dans `data/new/`
- Pas de construction dynamique
- `indicators_by_timeframe` n'est **pas utilisé** pour le Lot 2

### CPU - Lot 2
```bash
# 1. Traitement des données (avec optimisation chunks)
python scripts/process_data.py --exec_profile cpu_lot2

# 2. Fusion des données
python scripts/merge_processed_data.py --exec_profile cpu_lot2

# 3. Test rapide (validation)
python scripts/train_rl_agent.py --exec_profile cpu_lot2 --device cpu --initial_capital 15 --total_timesteps 100

# 4. Entraînement complet
python scripts/train_rl_agent.py --exec_profile cpu_lot2 --device cpu --initial_capital 15 --total_timesteps 200000
```

### GPU - Lot 2 (8 environnements parallèles)
```bash
# Pipeline complet GPU
python scripts/process_data.py --exec_profile gpu_lot2
python scripts/merge_processed_data.py --exec_profile gpu_lot2
python scripts/train_rl_agent.py --exec_profile gpu_lot2 --device cuda --initial_capital 15 --total_timesteps 2000000
```

## 📊 Configurations et Paramètres

### Tableaux Comparatifs

#### Caractéristiques Techniques
| Paramètre | Lot 1 (DOGE & Co) | Lot 2 (BTC & Co) |
|-----------|-------------------|-------------------|
| **Actifs** | ADA, DOGE, LTC, SOL, XRP | ADA, BNB, BTC, ETH, XRP |
| **Timeframe** | 1h | 1m |
| **Features par actif** | ~18 | 47 |
| **Colonnes totales** | ~85 (5×17) | ~235 (5×47) |
| **Période d'entraînement** | 2022-2025 | 2024-2025 |
| **Lignes d'entraînement** | ~10,921 | ~480,961 |

#### Paramètres d'Entraînement
| Configuration | CPU Lot 1 | CPU Lot 2 | GPU Lot 1 | GPU Lot 2 |
|---------------|-----------|-----------|-----------|-----------|
| **n_envs** | 1 | 1 | 8 | 8 |
| **Timesteps** | 200k | 200k | 2M | 2M |
| **Batch Size** | 64 | 64 | 512 | 512 |
| **CNN Features** | 32 | 32 | 256 | 256 |
| **Durée estimée** | 2-4h | 3-5h | 1-2h | 2-3h |

## 🧪 Tests et Validation

### Tests d'Intégrité des Données
```bash
# Vérifier les fichiers merged
python -c "
import pandas as pd
try:
    lot1 = pd.read_parquet('data/processed/merged/lot1/1h_train_merged.parquet')
    print(f'✅ Lot 1: {lot1.shape} (attendu: ~10k × ~85)')
except: print('❌ Lot 1: Fichier manquant')

try:
    lot2 = pd.read_parquet('data/processed/merged/lot2/1m_train_merged.parquet')
    print(f'✅ Lot 2: {lot2.shape} (attendu: ~480k × ~235)')
except: print('❌ Lot 2: Fichier manquant')
"
```

### Tests d'Environnement
```bash
# Test environnement Lot 1
python scripts/test_environment_with_merged_data.py --exec_profile cpu_lot1 --initial_capital 15

# Test environnement Lot 2
python scripts/test_environment_with_merged_data.py --exec_profile cpu_lot2 --initial_capital 15
```

### Tests de Parallélisation
```bash
# Test parallélisation (modifier temporairement agent_config_cpu.yaml: n_envs: 4)
python scripts/train_rl_agent.py --exec_profile cpu_lot1 --device cpu --initial_capital 15 --total_timesteps 50

# Observer htop pour vérifier l'utilisation multi-CPU
htop
```

## 🔧 Configurations Personnalisées

### Modification du Parallélisme
```yaml
# config/agent_config_cpu.yaml
agent:
  n_envs: 4  # Augmenter pour plus de parallélisme

# config/agent_config_gpu.yaml  
agent:
  n_envs: 8  # Optimal pour GPU
```

### Modification des Features
```yaml
# config/data_config_cpu_lot1.yaml
base_market_features: [
  "open", "high", "low", "close", "volume",
  "rsi_14", "ema_20", "ema_50", "macd",
  # Ajouter/supprimer des features ici
]
```

### Optimisation Chunks (Lot 2)
```yaml
# config/data_config_cpu_lot2.yaml
processing_chunk_size: 10000  # Réduire si problèmes mémoire
```

## 📈 Monitoring en Temps Réel

### TensorBoard
```bash
# Lancer TensorBoard
tensorboard --logdir reports/tensorboard_logs --host 0.0.0.0 --port 6006

# Accès via navigateur
http://localhost:6006
```

### Métriques Clés à Surveiller
- **rollout/ep_rew_mean** : Récompense moyenne par épisode
- **train/policy_loss** : Perte de la politique
- **train/value_loss** : Perte de la fonction de valeur
- **rollout/ep_len_mean** : Longueur moyenne des épisodes

### Surveillance Système
```bash
# CPU/RAM
htop

# GPU (si applicable)
watch -n 1 nvidia-smi

# Espace disque
df -h
```

## 🚨 Troubleshooting Multi-Lots

### Erreurs Courantes

#### "Feature non trouvée"
```bash
# Problème : Incohérence entre configuration et données
# Solution : Vérifier le profil utilisé
python -c "
from adan_trading_bot.common.utils import load_config
config = load_config('config/data_config_cpu_lot1.yaml')
print(f'Features configurées: {len(config[\"base_market_features\"])}')
"
```

#### "Colonnes manquantes dans le DataFrame"
```bash
# Problème : Mauvais lot chargé
# Solution : Vérifier que exec_profile correspond aux données disponibles
ls -la data/processed/merged/lot1/  # Pour cpu_lot1/gpu_lot1
ls -la data/processed/merged/lot2/  # Pour cpu_lot2/gpu_lot2
```

#### Erreur de Segmentation (Lot 2)
```bash
# Problème : Mémoire insuffisante pour gros volumes
# Solution : Réduire chunk_size dans data_config_*_lot2.yaml
# processing_chunk_size: 5000  # Au lieu de 10000
```

#### Parallélisation ne fonctionne pas
```bash
# Vérifier la configuration n_envs
grep -n "n_envs" config/agent_config_*.yaml

# Vérifier les processus pendant l'entraînement
ps aux | grep python
```

## 🎯 Cas d'Usage Recommandés

### Pour le Développement et Tests
```bash
# Utiliser Lot 1 (plus rapide)
python scripts/train_rl_agent.py --exec_profile cpu_lot1 --device cpu --initial_capital 15 --total_timesteps 1000
```

### Pour l'Entraînement de Production
```bash
# Utiliser Lot 2 avec GPU (données riches)
python scripts/train_rl_agent.py --exec_profile gpu_lot2 --device cuda --initial_capital 15 --total_timesteps 2000000
```

### Pour la Recherche de Stratégies Scalping
```bash
# Lot 2 en 1m avec features avancées
python scripts/train_rl_agent.py --exec_profile gpu_lot2 --device cuda --initial_capital 15 --total_timesteps 5000000
```

### Pour Day Trading
```bash
# Lot 1 en 1h avec features standard
python scripts/train_rl_agent.py --exec_profile gpu_lot1 --device cuda --initial_capital 15 --total_timesteps 3000000
```

## 📚 Fichiers de Configuration

### Structure des Configurations
```
config/
├── data_config_cpu_lot1.yaml    # CPU + Lot DOGE & Co
├── data_config_gpu_lot1.yaml    # GPU + Lot DOGE & Co  
├── data_config_cpu_lot2.yaml    # CPU + Lot BTC & Co
├── data_config_gpu_lot2.yaml    # GPU + Lot BTC & Co
├── agent_config_cpu.yaml        # Agent CPU (n_envs: 1)
├── agent_config_gpu.yaml        # Agent GPU (n_envs: 8)
├── environment_config.yaml      # Env trading (commun)
└── main_config.yaml            # Config principale
```

## 🎛️ Commandes de Référence Rapide

### Pipeline Complet Lot 1
```bash
python scripts/process_data.py --exec_profile cpu_lot1
python scripts/merge_processed_data.py --exec_profile cpu_lot1  
python scripts/train_rl_agent.py --exec_profile cpu_lot1 --device cpu --initial_capital 15 --total_timesteps 200000
```

### Pipeline Complet Lot 2
```bash
python scripts/process_data.py --exec_profile cpu_lot2
python scripts/merge_processed_data.py --exec_profile cpu_lot2
python scripts/train_rl_agent.py --exec_profile cpu_lot2 --device cpu --initial_capital 15 --total_timesteps 200000
```

### Test Rapide Multi-Lots
```bash
# Test rapide des deux lots
python scripts/train_rl_agent.py --exec_profile cpu_lot1 --device cpu --initial_capital 15 --total_timesteps 100
python scripts/train_rl_agent.py --exec_profile cpu_lot2 --device cpu --initial_capital 15 --total_timesteps 100
```

### Production GPU
```bash
# Entraînement production avec parallélisation
python scripts/train_rl_agent.py --exec_profile gpu_lot1 --device cuda --initial_capital 15 --total_timesteps 2000000
python scripts/train_rl_agent.py --exec_profile gpu_lot2 --device cuda --initial_capital 15 --total_timesteps 2000000
```

---

## 🚀 Prochaines Étapes

Une fois l'entraînement terminé avec succès :

1. **Analyse des Résultats** : Examiner TensorBoard et métriques
2. **Backtesting** : Tester le modèle sur données de test  
3. **Optimisation** : Ajuster hyperparamètres avec Optuna
4. **Déploiement** : Migration vers trading en temps réel

**🎯 ADAN Multi-Lots - Maximisez vos stratégies avec des données adaptées !**