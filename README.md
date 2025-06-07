<<<<<<< HEAD
# ADAN - Agent de Décision Algorithmique Neuronal

## 🚀 Vue d'Ensemble

ADAN est une plateforme avancée de trading algorithmique basée sur l'apprentissage par renforcement (RL). Le système utilise un agent PPO avec extracteur CNN pour apprendre des stratégies de trading rentables sur des données multi-assets avec indicateurs techniques spécifiques par timeframe.

## ✨ Fonctionnalités Principales

### 🎯 **Système Multi-Assets et Multi-Lots**
- **Lot 1 (DOGE & Co)** : 5 cryptomonnaies (ADAUSDT, DOGEUSDT, LTCUSDT, SOLUSDT, XRPUSDT) - Timeframe 1h, ~18 features
- **Lot 2 (BTC & Co)** : 5 cryptomonnaies (ADAUSDT, BNBUSDT, BTCUSDT, ETHUSDT, XRPUSDT) - Timeframe 1m, 47 features avancées
- Gestion intelligente du portefeuille avec allocation dynamique
- Système de paliers adaptatif selon le capital disponible

### 📊 **Indicateurs Techniques Multi-Niveaux**
- **Lot 1 (Indicateurs Standard)** : OHLCV, RSI, EMA, MACD, Bollinger Bands, ATR (~18 features)
- **Lot 2 (Indicateurs Avancés)** : OHLCV + SMA, EMA, RSI, MACD, Bollinger, ATR, Stochastic, ADX, CCI, Momentum, ROC, Williams %R, TRIX, Ultimate Oscillator, DPO, OBV, VWMA, CMF, MFI, Parabolic SAR, Ichimoku, KAMA, VWAP, Stoch RSI, CMO, PPO, Fisher Transform, HMM Regime Detection (47 features)
- **Timeframes Flexibles** : 1m (scalping), 1h (day trading), 1d (swing trading)

### 🧠 **Architecture IA Sophistiquée**
- Agent PPO avec MultiInputPolicy
- CNN Feature Extractor pour l'analyse temporelle
- Support CPU et GPU avec configurations optimisées
- **Entraînement parallèle** : VecEnv avec n_envs configurable (1-8 environnements)
- **Traitement par chunks** pour gros volumes de données (évite les erreurs de mémoire)

### 🔧 **Pipeline de Données Robuste**
- Collecte automatisée via CCXT (Binance)
- Traitement et normalisation des indicateurs
- Fusion multi-assets avec validation complète
- Support de plusieurs timeframes simultanés

## 📁 Structure du Projet

```
ADAN/
├── config/                 # Configurations YAML
│   ├── main_config.yaml    # Configuration principale
│   ├── data_config_*.yaml  # Données (CPU/GPU + Lots)
│   ├── agent_config_*.yaml # Agent RL (CPU/GPU + n_envs)
│   └── environment_config.yaml # Environnement trading
├── data/                   # Données de marché
│   ├── raw/               # Données brutes OHLCV (Lot 1)
│   ├── new/               # Données riches pré-calculées (Lot 2)
│   ├── processed/         # Données + indicateurs
│   │   ├── lot1/          # Données traitées Lot 1
│   │   ├── lot2/          # Données traitées Lot 2
│   │   └── merged/        # Données fusionnées par lot
│   │       ├── lot1/      # Fichiers merged Lot 1
│   │       └── lot2/      # Fichiers merged Lot 2
│   └── scalers_encoders/  # Normalisation par lot
├── src/adan_trading_bot/   # Code source principal
│   ├── data_processing/   # Pipeline de données
│   ├── environment/       # Environnement Gymnasium
│   ├── agents/           # Agents RL et CNN
│   └── common/           # Utilitaires
├── scripts/              # Scripts d'exécution
├── models/               # Modèles entraînés
├── reports/              # Analyses et TensorBoard
└── execution.md          # Guide complet d'exécution
```

## 🚀 Démarrage Rapide

### 📋 Prérequis
- Python 3.11
- CUDA 11.8+ (pour GPU)
- 8GB+ RAM (CPU) / 32GB+ RAM (GPU)

### 🛠️ Installation
```bash
# Cloner et configurer
git clone <ADAN_REPOSITORY_URL>
cd ADAN
conda create -n trading_env python=3.11 -y
conda activate trading_env
pip install -r requirements.txt

# Pour GPU uniquement
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### ⚡ Exécution Multi-Lots

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

### 🚀 Exécution GPU (Production)
```bash
# Lot 1 - Configuration optimisée : 8 workers parallèles
python scripts/train_rl_agent.py --exec_profile gpu_lot1 --device cuda --initial_capital 15 --total_timesteps 2000000

# Lot 2 - Données riches 1m avec parallélisation
python scripts/train_rl_agent.py --exec_profile gpu_lot2 --device cuda --initial_capital 15 --total_timesteps 2000000
```

## 📊 Monitoring et Analyse

### TensorBoard
```bash
tensorboard --logdir reports/tensorboard_logs --host 0.0.0.0 --port 6006
```
**Métriques clés :** rollout/ep_rew_mean, train/policy_loss, train/value_loss

### GPU Monitoring
```bash
watch -n 1 nvidia-smi
```

## ⚙️ Configuration Avancée

### Profils d'Exécution Multi-Lots

#### CPU (Test/Développement)
| Paramètre | Lot 1 (cpu_lot1) | Lot 2 (cpu_lot2) |
|-----------|-------------------|-------------------|
| **Timesteps** | 200,000 | 200,000 |
| **Workers (n_envs)** | 1 | 1 |
| **Batch Size** | 64 | 64 |
| **Features** | ~18 (1h) | 47 (1m) |
| **Durée** | 2-4h | 3-5h |

#### GPU (Production)
| Paramètre | Lot 1 (gpu_lot1) | Lot 2 (gpu_lot2) |
|-----------|-------------------|-------------------|
| **Timesteps** | 2,000,000 | 2,000,000 |
| **Workers (n_envs)** | 8 | 8 |
| **Batch Size** | 512 | 512 |
| **Features** | ~18 (1h) | 47 (1m) |
| **Durée** | 8-12h | 10-15h |

### Personnalisation

#### Indicateurs Techniques
Modifiez `config/data_config_*.yaml` section `indicators_by_timeframe` pour ajouter/supprimer des indicateurs.

#### Architecture Agent
Ajustez `config/agent_config_*.yaml` pour modifier :
- Dimensions du CNN (`features_dim`, `conv_layers`)
- Architecture MLP (`net_arch`)
- Hyperparamètres PPO (`learning_rate`, `batch_size`)

#### Environnement Trading
Configurez `config/environment_config.yaml` pour :
- Paliers de capital et allocation
- Pénalités et récompenses
- Règles de trading

## 🔍 Validation et Tests

### Validation du Pipeline Multi-Lots
```bash
# Test court Lot 1 (DOGE & Co)
python scripts/train_rl_agent.py --exec_profile cpu_lot1 --device cpu --initial_capital 15 --total_timesteps 100

# Test court Lot 2 (BTC & Co)  
python scripts/train_rl_agent.py --exec_profile cpu_lot2 --device cpu --initial_capital 15 --total_timesteps 100

# Test environnement
python scripts/test_environment_with_merged_data.py --exec_profile cpu_lot1 --initial_capital 15
python scripts/test_environment_with_merged_data.py --exec_profile cpu_lot2 --initial_capital 15
```

### Vérifications Critiques Multi-Lots
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
print(f'Lot 1 Features: {len(lot1_config[\"base_market_features\"])} par asset')
print(f'Lot 2 Features: {len(lot2_config[\"base_market_features\"])} par asset')
"
```

## 🚨 Troubleshooting

### Problèmes Courants

#### "FEATURE NON TROUVÉE"
- Vérifier `base_market_features` dans `config/data_config_*_lot*.yaml` 
- Vérifier que vous utilisez le bon profil (lot1 vs lot2)
- Régénérer les données : `process_data.py --exec_profile [profil]` puis `merge_processed_data.py --exec_profile [profil]`

#### Erreur "Colonnes manquantes"
- Assurer que le lot choisi correspond aux données disponibles
- Lot 1 : Utilise `data/backup/` ou données collectées via `fetch_data_ccxt.py`
- Lot 2 : Utilise `data/new/` avec données pré-calculées

#### Erreur GPU/CUDA
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

#### Affichage Rich Corrompu
- **Solution :** Toujours utiliser `TERM=dumb PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1`

## 📈 Performances et Résultats

### Métriques de Performance
- **PnL Total** : Evolution du capital
- **Sharpe Ratio** : Rendement ajusté au risque
- **Max Drawdown** : Perte maximale
- **Win Rate** : Pourcentage de trades gagnants

### Fichiers Générés
- `models/final_model.zip` : Modèle final entraîné
- `models/best_model/` : Meilleur modèle selon validation
- `models/checkpoints/` : Sauvegardes périodiques
- `reports/tensorboard_logs/` : Métriques d'entraînement

## 📚 Documentation Complète

- **[Guide d'Exécution](execution.md)** : Instructions détaillées étape par étape
- **[Configuration Multi-Lots](config/)** : Référence des paramètres YAML par lot
- **[Documentation Technique](docs/)** : Architecture et détails d'implémentation

## 🎛️ Configurations Disponibles

### Profils de Données
- `cpu_lot1` / `gpu_lot1` : DOGE & Co, 1h, 18 features standard
- `cpu_lot2` / `gpu_lot2` : BTC & Co, 1m, 47 features avancées

### Profils d'Agent  
- `cpu` : 1 environnement, configurations légères
- `gpu` : 8 environnements parallèles, configurations optimisées

## 🤖 Fonctionnalités Avancées (Roadmap)

### En Développement
- [ ] Script de backtesting avancé (`scripts/evaluate_agent.py`)
- [ ] Notifications Telegram
- [ ] Live trading avec APIs exchange
- [ ] Optimisation automatique des hyperparamètres (Optuna)
- [ ] Boucle évolutive Darwinienne

### Futurs Indicateurs
- [ ] Ichimoku Cloud
- [ ] Points Pivots 
- [ ] Fibonacci Retracements
- [ ] Volume Profile

## 🛡️ Sécurité et Bonnes Pratiques

- **Gestion des API Keys** : Variables d'environnement uniquement
- **Validation des Données** : Vérifications automatiques à chaque étape
- **Sauvegarde Automatique** : Modèles et checkpoints réguliers
- **Logging Complet** : Traçabilité de toutes les opérations

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push sur la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👨‍💻 Contact

- **Auteur :** Cabrel
- **GitHub :** [Cabrel10](https://github.com/Cabrel10)
- **Projet :** [ADAN Repository](https://github.com/Cabrel10/ADAN)

---

**🎯 ADAN - Transformez vos stratégies de trading avec l'IA !**
=======
# ADAN
eva01
>>>>>>> 296a1e7e7811e4726ac5e90c67f96c4d621521a6
