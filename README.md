# 🚀 ADAN - Agent de Décision Algorithmique Neuronal

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

ADAN est une plateforme avancée de trading algorithmique basée sur l'apprentissage par renforcement (RL). Le système utilise un agent PPO avec extracteur CNN pour apprendre des stratégies de trading rentables sur des données multi-assets avec indicateurs techniques spécifiques par timeframe.

## 📋 Table des matières
- [Fonctionnalités](#-fonctionnalités)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Utilisation](#-utilisation)
- [Structure du Projet](#-structure-du-projet)
- [Contribution](#-contribution)
- [Licence](#-licence)

## ✨ Fonctionnalités

### 📊 Système Multi-Assets et Multi-Lots
- **Lot 1 (DOGE & Co)** : 5 cryptomonnaies (ADAUSDT, DOGEUSDT, LTCUSDT, SOLUSDT, XRPUSDT) - Timeframe 1h, ~18 features
- **Lot 2 (BTC & Co)** : 5 cryptomonnaies (ADAUSDT, BNBUSDT, BTCUSDT, ETHUSDT, XRPUSDT) - Timeframe 1m, 47 features avancées
- Gestion intelligente du portefeuille avec allocation dynamique
- Système de paliers adaptatif selon le capital disponible

### 📈 Indicateurs Techniques Multi-Niveaux
- **Lot 1 (Indicateurs Standard)** : OHLCV, RSI, EMA, MACD, Bollinger Bands, ATR (~18 features)
- **Lot 2 (Indicateurs Avancés)** : OHLCV + SMA, EMA, RSI, MACD, Bollinger, ATR, Stochastic, ADX, CCI, Momentum, ROC, Williams %R, TRIX, Ultimate Oscillator, DPO, OBV, VWMA, CMF, MFI, Parabolic SAR, Ichimoku, KAMA, VWAP, Stoch RSI, CMO, PPO, Fisher Transform, HMM Regime Detection (47 features)
- **Timeframes Flexibles** : 1m (scalping), 1h (day trading), 1d (swing trading)

### 🧠 Architecture IA Sophistiquée
- Agent PPO avec MultiInputPolicy
- CNN Feature Extractor pour l'analyse temporelle
- Support CPU et GPU avec configurations optimisées
- **Entraînement parallèle** : VecEnv avec n_envs configurable (1-8 environnements)
- **Traitement par chunks** pour gros volumes de données (évite les erreurs de mémoire)
- **Dynamic Behavior Engine (DBE)** : Gestion dynamique des paramètres de trading
- **Apprentissage continu** avec mise à jour en temps réel des stratégies

### 🔄 Pipeline de Données Robuste
- Collecte automatisée via CCXT (Binance)
- Traitement et normalisation des indicateurs
- Fusion multi-assets avec validation complète
- Support de plusieurs timeframes simultanés

### 🎓 Apprentissage en Ligne Avancé
- **Mémoire de Rejeu Prioritaire** : Échantillonnage intelligent des expériences les plus instructives
- **Elastic Weight Consolidation (EWC)** : Prévention de l'oubli catastrophique lors de l'apprentissage continu
- **Mise à jour du réseau cible** : Pour une stabilité accrue de l'apprentissage
- **Optimisation CPU/GPU** : Support natif pour les environnements CPU et GPU
- **Journalisation complète** : Suivi détaillé des métriques d'apprentissage via TensorBoard

## 🚀 Nouvelles Fonctionnalités (Sprint 4)

### 🎓 Apprentissage en Ligne Avancé
L'agent d'apprentissage en ligne permet d'améliorer continuellement le modèle en temps réel avec de nouvelles données de marché.

**Fonctionnalités clés** :
- **Mémoire de Rejeu Prioritaire** : Améliore l'efficacité de l'apprentissage en se concentrant sur les expériences les plus instructives
- **Elastic Weight Consolidation (EWC)** : Empêche l'oubli des connaissances précédentes lors de l'apprentissage de nouvelles stratégies
- **Mise à jour asynchrone** : Mise à jour du modèle en arrière-plan sans interrompre le trading
- **Sauvegarde automatique** : Sauvegarde périodique du modèle et du buffer d'expérience

**Utilisation** :
```bash
python scripts/run_online_learning.py \
  --model-path models/rl_agents/adan_ppo_latest.zip \
  --config config/online_learning_config.yaml \
  --env-config config/environment_config.yaml \
  --save-path models/online \
  --tensorboard-log logs/online \
  --steps 100000
```

**Configuration recommandée** :
- `experience_buffer_size`: 100000 (taille du buffer d'expérience)
- `batch_size`: 128 (taille des lots d'apprentissage)
- `learning_rate`: 1e-5 (taux d'apprentissage)
- `use_prioritized_replay`: true (activer la mémoire de rejeu prioritaire)
- `use_ewc`: true (activer EWC pour éviter l'oubli catastrophique)
- `target_update_freq`: 1000 (fréquence de mise à jour du réseau cible)

## 🚀 Nouvelles Fonctionnalités (Sprint 3)

### 🔄 Dynamic Behavior Engine (DBE)
Le DBE est un système expert qui adapte dynamiquement les stratégies de trading en fonction des conditions du marché et des performances du portefeuille.

**Fonctionnalités clés** :
- Ajustement automatique des stops-loss et take-profit
- Gestion dynamique de la taille des positions
- Détection des régimes de marché (tendance, range, volatilité)
- Adaptation en temps réel des stratégies

### 📚 Apprentissage Continu
- Mise à jour du modèle en temps réel avec les nouvelles données de marché
- Réentraînement périodique pour s'adapter aux changements de marché
- Sauvegarde et restauration de l'état de l'agent

## 🎯 Utilisation du DBE

### Configuration
Le DBE est configuré via le fichier `config/environment_config.yaml` dans la section `dynamic_behavior` :

```yaml
dynamic_behavior:
  # Paramètres de gestion du risque
  risk_management:
    initial_sl_pct: 0.02  # Stop-loss initial (2%)
    initial_tp_pct: 0.04  # Take-profit initial (4%)
    max_position_size: 0.1  # Taille maximale de position (10% du capital)
    
  # Détection des régimes de marché
  market_regime_detection:
    volatility_lookback: 20  # Période de lookback pour le calcul de la volatilité
    trend_lookback: 50      # Période de lookback pour la détection de tendance
    
  # Paramètres d'adaptation
  adaptation:
    learning_rate: 0.01     # Vitesse d'adaptation des paramètres
    min_volatility: 0.005   # Volatilité minimale pour éviter les divisions par zéro
    max_volatility: 0.10    # Volatilité maximale pour le calcul des paramètres
```

### Utilisation dans le code

```python
from src.adan_trading_bot.environment import MultiAssetEnv
from src.adan_trading_bot.environment.dbe import DynamicBehaviorEngine

# Créer l'environnement avec le DBE activé
env = MultiAssetEnv(config=config, use_dbe=True)

# Accéder au DBE
dbe = env.dbe

# Mettre à jour les métriques
dbe.update_metrics(
    portfolio_metrics={
        'equity': portfolio_value,
        'drawdown': current_drawdown,
        'sharpe': sharpe_ratio
    },
    market_data={
        'prices': current_prices,
        'volumes': current_volumes,
        'volatility': current_volatility
    }
)

# Obtenir les paramètres de trading actuels
params = dbe.get_trading_parameters()
print(f"Stop-Loss: {params['sl_pct']:.2%}")
print(f"Take-Profit: {params['tp_pct']:.2%}")
print(f"Taille de position: {params['position_size']:.2f}")
```

### Visualisation des métriques
Le DBE enregistre des métriques détaillées qui peuvent être visualisées avec TensorBoard :

```bash
tensorboard --logdir=./reports/dbe_metrics
```

### Tests unitaires
Pour exécuter les tests du DBE :

```bash
pytest tests/environment/test_dbe.py -v
```

## 🛠 Installation

### Prérequis
- Python 3.11
- CUDA 11.8+ (pour GPU)
- 8GB+ RAM (CPU) / 32GB+ RAM (GPU)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommandé)
- [Git](https://git-scm.com/)

### Méthode 1 : Installation avec Conda (Recommandé)

```bash
# 1. Cloner le dépôt
git clone https://github.com/votre-utilisateur/adan-trading-bot.git
cd adan-trading-bot

# 2. Créer et activer l'environnement conda
conda create -n adan python=3.11 -y
conda activate adan

# 3. Installer les dépendances de base
pip install -r requirements.txt

# 4. Pour l'accélération GPU (optionnel)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. Installer pre-commit (recommandé)
pip install pre-commit
pre-commit install
```

### Méthode 2 : Installation avec venv

```bash
# 1. Cloner le dépôt
git clone https://github.com/votre-utilisateur/adan-trading-bot.git
cd adan-trading-bot

# 2. Créer et activer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: .\venv\Scripts\activate

# 3. Mettre à jour pip et installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 4. Installer pre-commit (recommandé)
pip install pre-commit
pre-commit install
```

## ⚙️ Configuration

### Configuration de l'environnement

1. Copiez le fichier `.env.example` vers `.env` :
   ```bash
   cp .env.example .env
   ```

2. Modifiez le fichier `.env` avec vos paramètres :
   ```ini
   # Configuration de l'API Binance (obligatoire pour le mode live)
   BINANCE_API_KEY=votre_cle_api_ici
   BINANCE_API_SECRET=votre_secret_api_ici
   
   # Paramètres généraux
   LOG_LEVEL=INFO
   ENVIRONMENT=dev
   
   # Paramètres de trading
   INITIAL_CAPITAL=10000
   MAX_POSITION_SIZE=0.1
   MAX_DAILY_TRADES=10
   ```

### Structure des répertoires

```
ADAN/
├── config/               # Fichiers de configuration
├── data/                 # Données brutes et traitées
│   ├── raw/             # Données brutes
│   ├── processed/       # Données traitées
│   └── models/          # Modèles entraînés
├── docs/                # Documentation
├── notebooks/           # Notebooks d'analyse
├── scripts/             # Scripts utilitaires
├── src/                 # Code source
│   └── adan_trading_bot/
│       ├── agent/       # Définition de l'agent RL
│       ├── common/      # Utilitaires communs
│       ├── data_processing/  # Traitement des données
│       ├── environment/ # Environnement de trading
│       └── training/    # Scripts d'entraînement
└── tests/               # Tests unitaires et d'intégration
```

## 🚀 Utilisation

### Préparation des données

```bash
# Traitement des données brutes
python scripts/process_data.py --config config/data_config.yaml

# Fusion des données traitées
python scripts/merge_processed_data.py --config config/data_config.yaml
```

### Entraînement du modèle

```bash
# Entraînement sur CPU (Lot 1 - DOGE & Co)
python scripts/train_rl_agent.py --config config/agent_config_cpu.yaml --profile lot1

# Entraînement sur GPU (Lot 2 - BTC & Co)
python scripts/train_rl_agent.py --config config/agent_config_gpu.yaml --profile lot2
```

### Évaluation du modèle

```bash
# Évaluer les performances du modèle
python scripts/evaluate_agent.py --model_path models/best_model.zip --num_episodes 100
```

### Démarrage du trading en direct (mode simulation)

```bash
python scripts/live_trading.py --mode paper --config config/live_config.yaml
```

## 🧪 Tests

### Exécution des tests unitaires

```bash
pytest tests/unit -v
```

### Vérification de la couverture de code

```bash
pytest --cov=src tests/unit/
```

### Vérification de la qualité du code

```bash
# Vérification du formatage
black --check src/

# Vérification du linting
flake8 src/

# Vérification des types
mypy src/
```

## 🤝 Contribution

1. **Fork** le dépôt
2. Créez une **branche** pour votre fonctionnalité (`git checkout -b feature/AmazingFeature`)
3. **Commit** vos modifications (`git commit -m 'Add some AmazingFeature'`)
4. **Push** vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une **Pull Request**

### Bonnes pratiques de développement

- Écrivez des tests unitaires pour les nouvelles fonctionnalités
- Assurez-vous que tous les tests passent avant de soumettre une PR
- Mettez à jour la documentation si nécessaire
- Suivez les conventions de code du projet (PEP 8, docstrings, etc.)
- Utilisez des messages de commit clairs et descriptifs

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🙏 Remerciements

- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) - Pour l'implémentation des algorithmes de RL
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) - Pour les environnements de renforcement
- [CCXT](https://github.com/ccxt/ccxt) - Pour l'accès aux API d'exchanges
- [TA-Lib](https://github.com/TA-Lib/ta-lib-python) - Pour les indicateurs techniques

## Utilisation

### Démarrage Rapide

#### Lot 1 (DOGE & Co - 1h, 18 features)
```bash
conda activate trading_env
cd ADAN

# Pipeline complet Lot 1
python scripts/process_data.py --exec_profile cpu_lot1
python scripts/merge_processed_data.py --exec_profile cpu_lot1
python scripts/train_rl_agent.py --exec_profile cpu_lot1 --device cpu --initial_capital 15 --total_timesteps 200000
```

#### Lot 2 (BTC & Co - 1m, 47 features)
```bash
# Pipeline complet Lot 2
python scripts/process_data.py --exec_profile cpu_lot2
python scripts/merge_processed_data.py --exec_profile cpu_lot2
python scripts/train_rl_agent.py --exec_profile cpu_lot2 --device cpu --initial_capital 15 --total_timesteps 200000
```

### Exécution GPU (Production)
```bash
# Lot 1 - Configuration optimisée : 8 workers parallèles
python scripts/train_rl_agent.py --exec_profile gpu_lot1 --device cuda --initial_capital 15 --total_timesteps 2000000

# Lot 2 - Données riches 1m avec parallélisation
python scripts/train_rl_agent.py --exec_profile gpu_lot2 --device cuda --initial_capital 15 --total_timesteps 2000000
```

### Monitoring et Analyse

#### TensorBoard
```bash
tensorboard --logdir reports/tensorboard_logs --host 0.0.0.0 --port 6006
```
**Métriques clés :** rollout/ep_rew_mean, train/policy_loss, train/value_loss

#### GPU Monitoring
```bash
watch -n 1 nvidia-smi
```

### Configuration Avancée

#### Profils d'Exécution Multi-Lots

| Paramètre | Lot 1 (cpu_lot1) | Lot 2 (cpu_lot2) |
|-----------|-------------------|-------------------|
| **Timesteps** | 200,000 | 200,000 |
| **Workers (n_envs)** | 1 | 1 |
| **Batch Size** | 64 | 64 |
| **Features** | ~18 (1h) | 47 (1m) |
| **Durée** | 2-4h | 3-5h |

| Paramètre | Lot 1 (gpu_lot1) | Lot 2 (gpu_lot2) |
|-----------|-------------------|-------------------|
| **Timesteps** | 2,000,000 | 2,000,000 |
| **Workers (n_envs)** | 8 | 8 |
| **Batch Size** | 512 | 512 |
| **Features** | ~18 (1h) | 47 (1m) |
| **Durée** | 8-12h | 10-15h |

#### Personnalisation

- **Indicateurs Techniques** : Modifiez `config/data_config_*.yaml` section `indicators_by_timeframe` pour ajouter/supprimer des indicateurs.
- **Architecture Agent** : Ajustez `config/agent_config_*.yaml` pour modifier les dimensions du CNN, l'architecture MLP, et les hyperparamètres PPO.
- **Environnement Trading** : Configurez `config/environment_config.yaml` pour les paliers de capital, les pénalités, les récompenses et les règles de trading.

### Validation et Tests

#### Validation du Pipeline Multi-Lots
```bash
# Test court Lot 1 (DOGE & Co)
python scripts/train_rl_agent.py --exec_profile cpu_lot1 --device cpu --initial_capital 15 --total_timesteps 100

# Test court Lot 2 (BTC & Co)  
python scripts/train_rl_agent.py --exec_profile cpu_lot2 --device cpu --initial_capital 15 --total_timesteps 100

# Test environnement
python scripts/test_environment_with_merged_data.py --exec_profile cpu_lot1 --initial_capital 15
python scripts/test_environment_with_merged_data.py --exec_profile cpu_lot2 --initial_capital 15
```

#### Vérifications Critiques Multi-Lots
```bash
# Vérifier Lot 1 (doit être ~85 colonnes : 5 assets × ~17 features)
python -c "import pandas as pd; df = pd.read_parquet('data/processed/merged/lot1/1h_train_merged.parquet'); print(f'Lot 1 Shape: {df.shape}')"

# Vérifier Lot 2 (doit être ~235 colonnes : 5 assets × 47 features)  
python -c "import pandas as pd; df = pd.read_parquet('data/processed/merged/lot2/1m_train_merged.parquet'); print(f'Lot 2 Shape: {df.shape}')"

# Vérifier les features par lot
python -c "
import sys; sys.path.append('src')
from adan_trading_bot.common.utils import load_config
lot1_config = load_config('config/data_config_cpu_lot1.yaml')
lot2_config = load_config('config/data_config_cpu_lot2.yaml')
print(f'Lot 1 Features: {len(lot1_config["base_market_features"])} par asset')
print(f'Lot 2 Features: {len(lot2_config["base_market_features"])} par asset')
"
```

### Troubleshooting

#### Problèmes Courants

- **"FEATURE NON TROUVÉE"** : Vérifier `base_market_features` dans `config/data_config_*_lot*.yaml`, s'assurer du bon profil (lot1 vs lot2), et régénérer les données.
- **Erreur "Colonnes manquantes"** : Assurer que le lot choisi correspond aux données disponibles.
- **Erreur GPU/CUDA** : Vérifier l'installation CUDA avec `python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"`.
- **Affichage Rich Corrompu** : Utiliser `TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1`.

## 📋 Structure du Projet

```
ADAN/
├── config/                # Fichiers de configuration
│   ├── environment_config.yaml  # Configuration de l'environnement et du DBE
│   ├── agent_config_*.yaml      # Configurations des agents
│   └── data_config.yaml         # Configuration des données
├── data/                   # Données de marché
│   ├── raw/              # Données brutes
│   └── processed/        # Données traitées
├── models/               # Modèles entraînés
│   ├── saved_agents/     # Agents sauvegardés
│   └── dbe/              # États du Dynamic Behavior Engine
├── notebooks/            # Notebooks d'analyse
├── reports/              # Rapports et résultats
│   ├── backtests/        # Résultats des backtests
│   └── metrics/          # Métriques d'entraînement
├── scripts/              # Scripts d'exécution
│   ├── train_rl_agent.py # Entraînement initial
│   ├── online_learning_agent.py  # Apprentissage continu
│   └── evaluate_performance.py   # Évaluation et backtesting
├── src/                  # Code source
│   └── adan_trading_bot/
│       ├── common/       # Utilitaires communs
│       ├── data_processing/  # Traitement des données
│       ├── environment/  # Environnement de trading
│       │   └── dbe/      # Dynamic Behavior Engine
│       └── training/     # Scripts d'entraînement
└── tests/                # Tests unitaires et d'intégration
```

## Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push sur la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## Contact

- **Auteur :** Cabrel
- **GitHub :** [Cabrel10](https://github.com/Cabrel10)
- **Projet :** [ADAN Repository](https://github.com/Cabrel10/ADAN)

---

**🎯 ADAN - Transformez vos stratégies de trading avec l'IA !**