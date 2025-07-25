# ADAN Trading Bot - Source Code

Ce répertoire contient le code source principal du bot de trading ADAN avec Dynamic Behavior Engine.

## 🏗️ Architecture Modulaire

### 🎯 Core Components

#### Environment (`adan_trading_bot/environment/`)
- **`multi_asset_chunked_env.py`** : Environnement principal Gymnasium multi-assets
- **`dynamic_behavior_engine.py`** : DBE - Adaptation comportementale en temps réel
- **`state_builder.py`** : Construction observations 3D multi-timeframes
- **`reward_calculator.py`** : Calcul récompenses avec bonus performance
- **`action_translator.py`** : Traduction actions agent → ordres trading

#### Portfolio Management (`adan_trading_bot/portfolio/`)
- **`portfolio_manager.py`** : Gestion portfolio multi-assets avec métriques
- **`position_manager.py`** : Gestion positions avec ordres protection

#### Risk Management (`adan_trading_bot/risk_management/`)
- **`risk_assessor.py`** : Évaluation risques (volatilité, corrélation, VaR)
- **`risk_calculator.py`** : Calculs métriques risque (Sharpe, CVaR, drawdown)

#### Trading Engine (`adan_trading_bot/trading/`)
- **`order_manager.py`** : Exécution ordres avec validation
- **`action_validator.py`** : Validation actions avant exécution
- **`fee_manager.py`** : Gestion frais et slippage
- **`position_sizer.py`** : Dimensionnement positions basé risque

#### Data Processing (`adan_trading_bot/data_processing/`)
- **`chunked_loader.py`** : Chargement données par chunks (optimisation mémoire)
- **`state_builder.py`** : Construction états multi-timeframes
- **`observation_validator.py`** : Validation observations avant agent

### 🔧 Support Components

#### Common Utilities (`adan_trading_bot/common/`)
- **`metrics_tracker.py`** : Suivi métriques temps réel
- **`reward_logger.py`** : Logging récompenses et décisions
- **`config_validator.py`** : Validation configurations YAML
- **`replay_logger.py`** : Logging décisions DBE pour analyse

#### Agent (`adan_trading_bot/agent/`)
- **`ppo_agent.py`** : Agent PPO optimisé pour trading
- **`feature_extractors.py`** : Extracteurs features pour CNN 3D

#### Training (`adan_trading_bot/training/`)
- **`trainer.py`** : Boucle d'entraînement principale
- **`callbacks.py`** : Callbacks personnalisés pour monitoring
- **`hyperparam_modulator.py`** : Modulation hyperparamètres dynamique

#### Live Trading (`adan_trading_bot/live_trading/`)
- **`safety_manager.py`** : Gestion sécurité trading live
- **`experience_buffer.py`** : Buffer expériences pour apprentissage continu
- **`online_reward_calculator.py`** : Calcul récompenses temps réel

#### Exchange API (`adan_trading_bot/exchange_api/`)
- **`connector.py`** : Connecteur exchanges (CCXT)

#### Evaluation (`adan_trading_bot/evaluation/`)
- **`evaluator.py`** : Évaluation performance modèles

## 🚀 Flux de Données

```
Data Pipeline:
ChunkedLoader → StateBuilder → MultiAssetEnv → Agent → ActionTranslator → OrderManager
                     ↓                ↑
              ObservationValidator    DBE (Dynamic Behavior Engine)
                     ↓                ↑
              MetricsTracker ← PortfolioManager → RiskAssessor
```

## 🎛️ Dynamic Behavior Engine (DBE)

Le DBE est le cœur adaptatif du système :

### Fonctionnalités
- **Détection régimes marché** : Bull, Bear, Sideways, Volatile
- **Adaptation paramètres risque** : SL/TP dynamiques
- **Modulation récompenses** : Optimisation apprentissage
- **Gestion drawdown** : Protection capital automatique

### Intégration
```python
# Le DBE s'intègre automatiquement dans l'environnement
dbe_modulation = self.dbe.compute_dynamic_modulation()
# Appliqué aux calculs de récompense et gestion risque
```

## 📊 Optimisations Performance

### Mémoire
- **Chunked Loading** : Chargement par blocs (10k points)
- **Aggressive Cleanup** : Nettoyage mémoire automatique
- **Single Chunk Strategy** : Un seul chunk en mémoire

### CPU
- **Vectorisation NumPy** : Calculs optimisés
- **Lazy Loading** : Chargement à la demande
- **Parallel Processing** : Support multi-instances

## 🧪 Tests et Validation

### Tests Unitaires (`../tests/unit/`)
- Chaque module a ses tests spécifiques
- Validation des calculs de risque
- Tests des transformations de données

### Tests d'Intégration (`../tests/integration/`)
- Tests bout-en-bout de l'environnement
- Validation pipeline complet
- Tests performance et mémoire

## 🔌 Points d'Extension

### Nouveaux Assets
1. Ajouter dans `data_config.yaml`
2. Mettre à jour `ChunkedLoader.assets_list`
3. Tester avec `test_environment_quick.py`

### Nouvelles Métriques
1. Implémenter dans `RiskCalculator`
2. Ajouter au `MetricsTracker`
3. Intégrer dans `DBE` si nécessaire

### Nouveaux Indicateurs
1. Ajouter dans `StateBuilder`
2. Mettre à jour configuration features
3. Valider avec `ObservationValidator`

## 📈 Monitoring

### Logs Disponibles
- **Training** : `../logs/training.log`
- **DBE Decisions** : `../logs/dbe/dbe_replay_*.jsonl.gz`
- **Metrics** : `../logs/metrics_*.json`
- **TensorBoard** : `../logs/tensorboard/`

### Métriques Clés
- Portfolio value evolution
- Risk metrics (VaR, Sharpe, Drawdown)
- DBE adaptation decisions
- Memory usage patterns
- Trading execution quality

## 🛠️ Configuration

### Fichiers de Configuration
- **`../config/main_config.yaml`** : Configuration générale
- **`../config/dbe_config.yaml`** : Paramètres DBE
- **`../config/risk_config.yaml`** : Gestion risques
- **`../config/memory_config.yaml`** : Optimisations mémoire

### Variables d'Environnement
```bash
export ADAN_CONFIG_DIR=config/
export ADAN_DATA_DIR=data/final/
export ADAN_LOG_LEVEL=INFO
```

---

**Architecture** : Modulaire | **Performance** : Optimisée | **Status** : Production Ready