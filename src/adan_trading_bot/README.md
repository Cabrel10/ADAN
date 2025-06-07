# Package `adan_trading_bot`

Ce package constitue le cœur fonctionnel de l'agent de trading ADAN. Il est structuré en plusieurs sous-modules, chacun responsable d'une partie spécifique de la logique de l'agent.

## Arborescence des Sous-Modules

*   `common/`: Utilitaires partagés, constantes (codes d'action, noms de colonnes), et configuration du logger.
    *   Tâches : Définir des constantes globales, fournir des fonctions utilitaires (ex: gestion des dates, sauvegarde/chargement de fichiers), initialiser un logger standardisé.
*   `data_processing/`: Modules pour le chargement, le nettoyage, la transformation des données et l'ingénierie des features.
    *   Tâches : Charger les données brutes (Parquet, CSV), calculer les indicateurs techniques, normaliser/standardiser les features, diviser les données en ensembles d'entraînement/validation/test.
*   `environment/`: Définition de l'environnement d'apprentissage par renforcement personnalisé (`MultiAssetEnv`).
    *   Tâches : Implémenter la logique de `step()`, `reset()`, la gestion des ordres (MARKET, LIMIT, STOP), le calcul des frais, la construction de l'état (observation) et le calcul de la fonction de récompense.
*   `agent/`: Implémentation et configuration de l'agent d'apprentissage par renforcement (ex: PPO utilisant Stable-Baselines3).
    *   Tâches : Définir l'architecture du réseau de neurones (politique et valeur), wrapper l'agent SB3, gérer le chargement et la sauvegarde des modèles d'agent.
*   `training/`: Scripts et pipelines pour l'entraînement de l'agent RL.
    *   Tâches : Orchestrer le processus d'entraînement, initialiser l'environnement et l'agent, lancer la boucle d'apprentissage, utiliser des callbacks pour la sauvegarde et l'évaluation périodique.
*   `evaluation/`: Outils pour le backtesting de l'agent entraîné et le calcul des métriques de performance.
    *   Tâches : Exécuter l'agent sur des données historiques non vues, calculer le PnL, le ratio de Sharpe, le drawdown maximum, et d'autres métriques de performance.
*   `main.py`: Point d'entrée optionnel si l'agent est conçu pour être exécuté comme une application autonome (ex: pour le paper trading ou le live trading).

## État actuel du développement

### Composants implémentés

#### 1. Chargement et préparation des données
- `data_loader.py` : Chargement des données brutes et fusionnées
  - Fonction `load_merged_data` pour charger directement les fichiers de données fusionnées
  - Support pour différents timeframes (1m, 1h, 1d) et splits (train, val, test)

- `feature_engineer.py` : Calcul des indicateurs techniques et préparation des données pour l'entraînement
  - Pipeline de préparation des données adapté pour les données fusionnées
  - Calcul dynamique des indicateurs techniques selon le timeframe

#### 2. Environnement de trading
- `multi_asset_env.py` : Environnement de trading multi-actifs conforme à Gymnasium
  - Implémentation des méthodes `step()`, `reset()` et `render()`
  - Système de paliers pour l'allocation de capital et les récompenses
  - Gestion des positions et des ordres de trading

- `order_manager.py` : Gestion des ordres de trading
  - Exécution des ordres MARKET
  - Support préliminaire pour les ordres LIMIT, STOP_LOSS, etc.
  - Vérification des règles d'ordre (capital suffisant, taille minimale, etc.)

- `reward_calculator.py` : Calcul des récompenses pour l'agent RL
  - Récompenses basées sur le log-return du portefeuille
  - Multiplicateurs de récompense selon les paliers
  - Pénalités pour les actions invalides

#### 3. Agent RL
- `state_builder.py` : Construction de l'état pour l'agent RL
  - Formatage des données de marché pour le CNN
  - Construction des features vectorielles pour l'agent

### Composants en cours de développement

#### 1. CNN Feature Extractor
- Implémentation d'un extracteur de features basé sur un CNN pour analyser les données de marché
- Intégration avec Stable Baselines3 pour l'entraînement de l'agent RL

#### 2. Ordres avancés
- Implémentation complète des ordres LIMIT, STOP_LOSS, TAKE_PROFIT
- Gestion des ordres en attente et de leur expiration

#### 3. Visualisation
- Interface de visualisation des performances de trading
- Génération de rapports détaillés sur les stratégies de l'agent

## Configuration

Le comportement de chaque composant est contrôlé par les fichiers de configuration dans le répertoire `config/` :

- `data_config.yaml` : Configuration des données (actifs, timeframes, features)
- `environment_config.yaml` : Configuration de l'environnement de trading (pénalités, règles d'ordre, paliers)
- `agent_config.yaml` : Configuration de l'agent RL (architecture, hyperparamètres)

Chaque sous-module contient son propre `README.md` pour plus de détails sur son contenu et son rôle spécifique.
