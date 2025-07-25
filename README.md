# ADAN Trading Bot

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

ADAN (Adaptive Dynamic Agent Network) est un bot de trading avancé utilisant l'apprentissage par renforcement pour le trading de cryptomonnaies avec un système de comportement dynamique (DBE - Dynamic Behavior Engine).

## 🛠 Configuration Cursor

Ce projet est configuré pour une intégration optimale avec [Cursor](https://www.cursor.sh/), un éditeur de code alimenté par l'IA. Les configurations suivantes sont incluses :

- **Configuration Cursor** : `.cursor/config.json`
- **Configuration Agent** : `.cursor/agent.json`
- **Gestion des tâches** : `.todo2.json`
- **Spécifications** : `SPEC.md`
- **Tâches** : `todo.md`

### Prérequis

- [Cursor Editor](https://www.cursor.sh/) (version 1.5.0 ou supérieure)
- Extension Todo2 pour la gestion des tâches
- Python 3.9+ avec environnement Conda (`trading_env`)

### Configuration recommandée

1. **Installer l'extension Todo2** dans Cursor
2. **Ouvrir le dossier du projet** dans Cursor
3. **Activer l'environnement Conda** :
   ```bash
   conda activate trading_env
   ```
4. **Installer les dépendances** :
   ```bash
   uv pip install -r requirements.txt
   ```
5. **Configurer les clés API** dans `.env` (voir `.env.example`)

### Commandes utiles

| Commande | Description |
|----------|-------------|
| `uv pip install -r requirements.txt` | Installer les dépendances |
| `pytest tests/ -v` | Exécuter les tests |
| `black src/ tests/` | Formater le code avec Black |
| `mypy src/ tests/` | Vérifier les types |
| `flake8 src/ tests/` | Vérifier le style de code |

## 📋 Gestion des Tâches

Le projet utilise le système de gestion de tâches intégré à Cursor (Todo2) avec le fichier `.todo2.json` comme configuration principale. Les tâches sont organisées en colonnes :

- **Backlog** : Tâches planifiées
- **À Faire** : Tâches prêtes à être traitées
- **En Cours** : Tâches en cours de développement
- **En Revue** : Tâches en attente de revue
- **Terminé** : Tâches finalisées

### Workflow de développement

1. **Créer une branche** pour la tâche :
   ```bash
   git checkout -b feature/nom-de-la-fonctionnalite
   ```
2. **Travailler sur la tâche** en suivant les spécifications
3. **Tester les modifications** :
   ```bash
   pytest tests/ -v
   ```
4. **Valider les changements** :
   ```bash
   git add .
   git commit -m "type(scope): description concise"
   ```
5. **Pousser les modifications** et créer une Pull Request

## 🧪 Tests

Le projet inclut une suite de tests complète :

```bash
# Exécuter tous les tests
pytest tests/ -v

# Exécuter un test spécifique
pytest tests/test_state_builder.py -v

# Générer un rapport de couverture
pytest --cov=src tests/
```

## 🚀 État du Projet - OPÉRATIONNEL

✅ **Système Complet et Fonctionnel**
- Environnement de trading multi-assets opérationnel
- Dynamic Behavior Engine (DBE) intégré et actif
- Pipeline de données multi-timeframes optimisé
- Gestion de portfolio et des risques complète
- Entraînement parallèle avec 4 instances

## 🎯 Fonctionnalités Principales

### Core Trading System
- **Apprentissage par renforcement** : PPO (Proximal Policy Optimization) optimisé
- **Multi-timeframes** : Analyse simultanée 5m, 1h, 4h avec fusion intelligente
- **Multi-actifs** : Trading simultané BTC, ETH, SOL, XRP, ADA
- **Mémoire optimisée** : Chargement par chunks (141MB RAM pour 750k points)

### Dynamic Behavior Engine (DBE)
- **Adaptation en temps réel** : Ajustement automatique des paramètres de risque
- **Régimes de marché** : Détection et adaptation (Bull, Bear, Sideways, Volatile)
- **Gestion dynamique SL/TP** : Stop-loss et take-profit adaptatifs
- **Modulation des récompenses** : Optimisation continue de l'apprentissage

### Entraînement Parallèle
- **4 instances simultanées** avec stratégies différentes :
  - **Conservative** : Capital faible, risque serré
  - **Balanced** : Capital moyen, risque équilibré  
  - **Aggressive** : Capital élevé, risque large
  - **Adaptive** : DBE très actif, adaptation continue

## 📊 Performance et Optimisation

- **Mémoire** : ~141MB pour 5 assets × 15 chunks × 10k points
- **CPU** : Optimisé pour parallélisme (4 cores recommandés)
- **Données** : 750,000 points d'entraînement disponibles
- **Throughput** : ~50+ steps/seconde par instance

## 🏗️ Architecture

```
ADAN/
├── config/                 # Configurations (YAML)
├── data/final/            # Données traitées par asset
├── logs/                  # Logs et métriques
│   ├── dbe/              # Logs Dynamic Behavior Engine
│   └── tensorboard/      # Logs TensorBoard
├── models/               # Modèles entraînés
│   ├── checkpoints/      # Sauvegardes intermédiaires
│   └── parallel_*/       # Modèles parallèles
├── scripts/              # Scripts d'entraînement
│   ├── train_rl_agent.py        # Entraînement simple
│   ├── train_parallel_agents.py # Entraînement parallèle
│   └── test_*.py               # Scripts de test
└── src/adan_trading_bot/
    ├── environment/      # Environnement RL + DBE
    ├── portfolio/        # Gestion portfolio
    ├── risk_management/  # Calculs de risque
    ├── trading/         # Exécution ordres
    └── data_processing/ # Pipeline données
```

## 🚀 Démarrage Rapide

### Installation
```bash
# Cloner le repository
git clone <repo-url>
cd ADAN

# Installer les dépendances
pip install -r requirements.txt

# Créer les répertoires nécessaires
mkdir -p logs models/checkpoints models/best reports
```

### Entraînement Simple
```bash
# Entraînement avec une instance
python scripts/train_rl_agent.py --total_timesteps 10000 --initial_capital 1000

# Test rapide de l'environnement
python scripts/test_training_simple.py
```

### Entraînement Parallèle (Recommandé)
```bash
# Entraînement avec 4 instances parallèles
python scripts/train_parallel_agents.py --total_timesteps 50000 --n_envs 4

# Monitoring en temps réel
tensorboard --logdir logs/tensorboard
```

## ⚙️ Configuration

### Fichiers Principaux
- `main_config.yaml` : Configuration générale
- `data_config_cpu.yaml` : Optimisations CPU/mémoire
- `environment_config.yaml` : Paramètres environnement
- `dbe_config.yaml` : Dynamic Behavior Engine
- `risk_config.yaml` : Gestion des risques

### Paramètres Clés
```yaml
# Optimisation mémoire
chunk_size: 10000
max_chunks_in_memory: 1
aggressive_cleanup: true

# DBE Configuration
risk_parameters:
  base_sl_pct: 0.02
  base_tp_pct: 0.04
  drawdown_risk_multiplier: 2.0

# Entraînement parallèle
n_envs: 4
batch_size: 64
learning_rate: 3e-4
```

## 📈 Monitoring et Métriques

### TensorBoard
```bash
tensorboard --logdir logs/tensorboard
```

### Logs Disponibles
- `logs/training.log` : Logs d'entraînement
- `logs/dbe/` : Décisions du Dynamic Behavior Engine
- `logs/instance_*_monitor.csv` : Métriques par instance

### Métriques Suivies
- Portfolio value evolution
- Reward per episode
- Risk metrics (VaR, Sharpe, Drawdown)
- DBE decisions and adaptations
- Memory usage and performance

## 🧪 Tests et Validation

```bash
# Tests des composants principaux
python scripts/test_reward_calculator.py
python scripts/test_risk_calculator.py
python scripts/test_metrics_tracking.py

# Test complet de l'environnement
python scripts/test_training_simple.py

# Validation des configurations
python scripts/validate_configs.py
```

## 🔧 Développement

### Structure des Tests
- `tests/unit/` : Tests unitaires par composant
- `tests/integration/` : Tests d'intégration
- `scripts/test_*.py` : Tests fonctionnels

### Ajout de Nouvelles Fonctionnalités
1. Implémenter dans `src/adan_trading_bot/`
2. Ajouter tests dans `tests/`
3. Mettre à jour configuration si nécessaire
4. Tester avec `scripts/test_*.py`

## 📋 Roadmap

### Prochaines Améliorations
- [ ] Interface web de monitoring
- [ ] Support trading en temps réel
- [ ] Optimisations GPU
- [ ] Backtesting avancé
- [ ] API REST pour contrôle externe

### Optimisations Continues
- [ ] Fine-tuning hyperparamètres DBE
- [ ] Nouvelles stratégies de reward shaping
- [ ] Support d'assets supplémentaires
- [ ] Amélioration prédictions de régime

## 📈 Résumé des Progrès par Sprint

### Sprint 1 : Pipeline de Données Multi-Timeframe ✅
- Standardisation des configurations de données
- Scripts de traitement et fusion multi-timeframe
- Optimisation du ChunkedDataLoader pour gestion mémoire

### Sprint 2 : Environnement RL Multi-Canal ✅
- StateBuilder pour observations 3D normalisées
- ActionTranslator avec gestion frais et position sizing
- PortfolioManager multi-actifs avec métriques avancées
- RewardCalculator avec bonus de performance dynamique

### Sprint 3 : DBE et Trading Futures ✅
- RiskAssessor pour métriques de risque avancées
- PositionManager avec ordres de protection
- Support trading futures avec levier
- Logging détaillé des exécutions

### Sprint 4 : Apprentissage Continu ✅
- ExperienceBuffer avec Prioritized Experience Replay
- Apprentissage en ligne et mise à jour modèle
- Learning rate scheduling et versioning
- Monitoring temps réel

### Sprint 5 : Optimisation et Tests Finaux ✅
- Tests de performance et stress
- Optimisation ressources (mémoire/CPU)
- Monitoring production
- Documentation complète

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature
3. Ajouter tests pour nouvelles fonctionnalités
4. Valider avec les scripts de test
5. Soumettre une pull request

## 📄 Licence

MIT License - Voir `LICENSE` pour plus de détails.

---

**Status** : ✅ Production Ready | **Version** : 1.0.0 | **Last Update** : 2025-01-18