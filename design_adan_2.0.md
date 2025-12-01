# Conception (Design) : Plan de Reconstruction Complet ADAN 2.0

## 1. Architecture Générale

L'architecture d'ADAN 2.0 sera une refonte modulaire et robuste, visant à corriger les lacunes identifiées. Elle s'articulera autour de composants spécialisés et interconnectés, garantissant réalisme, reproductibilité et performance.

*   **Couche de Données :** Strictement séparée et enrichie.
*   **Couche Environnement :** Hautement réaliste, intégrant les frictions du marché réel.
*   **Couche Agent :** Composée d'experts individuels fusionnés progressivement.
*   **Couche de Contrôle :** Gestion centralisée de la reproductibilité, des risques et de la fréquence.
*   **Couche de Validation :** Protocole rigoureux et exhaustif.

## 2. Conception Détaillée

### Phase 0 : Audit Complet des Erreurs Critiques

*   **FR-0.1 (Analyse Post-Mortem des Échecs) :**
    *   **Design :** Intégration de logs structurés (JSONL) dans les composants clés (`PortfolioManager`, `MultiAssetChunkedEnv`, `RewardCalculator`) pour capturer les événements critiques (trades, changements de PnL, erreurs). Un script d'analyse (`scripts/analyze_critical_errors.py`) sera développé pour parser ces logs et générer un rapport synthétique.
*   **FR-0.2 (Vérification de la Reproductibilité) :**
    *   **Design :** Création du script `scripts/audit_reproducibility.py`. Ce script exécutera des runs courts avec des seeds spécifiques, vérifiera l'intégrité des splits de données, la consistance des outputs de l'environnement et le hachage des modèles entraînés pour détecter toute non-déterminisme.

### Phase 1 : Purification de l'Environnement (Données et Configuration)

*   **FR-1.1 (Reconstruction de l'Environnement Conda) :**
    *   **Design :** Création d'un fichier `environment_strict.yaml` listant toutes les dépendances Python avec leurs versions exactes. Ce fichier sera la source unique pour la création de l'environnement Conda.
*   **FR-1.2 (Gestion Globale des Graines Aléatoires) :**
    *   **Design :** Création du module `src/adan_trading_bot/common/reproducibility.py` contenant la classe `SeedManager`. Cette classe offrira des méthodes statiques (`set_global_seeds`, `seed_worker`) pour initialiser `random`, `numpy`, `torch` (CPU/CUDA) et les environnements `gym`/`stable-baselines3` avec une graine donnée. Tous les scripts d'entraînement et de validation devront appeler `SeedManager.set_global_seeds()` au démarrage.
*   **FR-1.3 (Nettoyage Radical des Artefacts) :**
    *   **Design :** Un script shell (`scripts/clean_project.sh`) sera créé pour supprimer les répertoires `data/processed`, `checkpoints`, `logs`, et `optuna.db`. Ce script sera appelé avant chaque exécution majeure du pipeline.
*   **FR-1.4 (Séparation Temporelle Stricte des Données) :**
    *   **Design :** Modification du `src/adan_trading_bot/data_processing/data_loader.py` pour implémenter une méthode `load_strict_split(start_train, end_train, start_test, end_test)`. Cette méthode garantira l'absence de chevauchement et la continuité temporelle des données.
*   **FR-1.5 (Correction et Vérification des Features) :**
    *   **Design :** Modification de `src/adan_trading_bot/data_processing/feature_engineer.py` pour ajouter les calculs des indicateurs `ATR_20`, `ATR_50` et `ADX` et s'assurer de leur présence dans les DataFrames traités.

### Phase 2 : Corrections Architecturales Profondes (Environnement et Récompense)

*   **FR-2.1 (Environnement d'Entraînement Réaliste - `RealisticTradingEnv`) :**
    *   **Design :** Création d'une nouvelle classe `src/adan_trading_bot/environment/realistic_trading_env.py` qui héritera de `MultiAssetChunkedEnv`. Cette classe intégrera les composants suivants :
        *   `BinanceFeeModel` (nouveau module) : Calcul des frais de maker/taker.
        *   `AdaptiveSlippage` (nouveau module) : Modélisation du slippage basé sur la volatilité et la taille de l'ordre.
        *   `LatencySimulator` (nouveau module) : Introduction d'un délai aléatoire dans l'exécution des ordres.
        *   `LiquidityModel` (nouveau module) : Modélisation de l'impact de l'ordre sur le prix en fonction de la liquidité du carnet d'ordres.
    *   La méthode `step()` de `RealisticTradingEnv` appliquera ces modèles séquentiellement.
*   **FR-2.2 (Contrôle de la Fréquence des Trades - `TradeFrequencyController`) :**
    *   **Design :** Création d'une classe `src/adan_trading_bot/environment/trade_frequency_controller.py`. Cette classe sera injectée dans `RealisticTradingEnv` et ses méthodes (`can_open_trade`, `can_close_trade`) seront appelées avant toute exécution de trade. Elle gérera :
        *   `min_trade_interval` (steps minimum entre trades).
        *   `daily_trade_limit` (trades max par jour).
        *   `cooldown_periods` (par actif).
*   **FR-2.3 (Fonction de Récompense Stabilisée - `StableRewardCalculator`) :**
    *   **Design :** Création d'une classe `src/adan_trading_bot/environment/stable_reward_calculator.py` qui remplacera l'actuel `RewardCalculator`. Elle implémentera :
        *   Normalisation du PnL.
        *   Calcul de la contribution Sharpe.
        *   Pénalités de Drawdown, de fréquence de trades.
        *   Bonus de consistance.
        *   La récompense finale sera clippée entre -1.0 et 1.0.
*   **FR-2.4 (Intégration Obligatoire de `VecNormalize`) :**
    *   **Design :** Le script d'entraînement (`train_adan_corrected.py`) enveloppera systématiquement l'instance de `RealisticTradingEnv` dans `stable_baselines3.common.vec_env.VecNormalize(norm_obs=True, norm_reward=True)`.
*   **FR-2.5 (Sauvegarde Automatique de `vecnormalize.pkl`) :**
    *   **Design :** Le script d'entraînement sera modifié pour s'assurer que l'objet `VecNormalize` est sauvegardé (méthode `env.save(path)`) dans le même répertoire que le modèle PPO entraîné.
*   **FR-2.6 (Alignement du `Portfolio State`) :**
    *   **Design :** Modification du `src/adan_trading_bot/data_processing/state_builder.py` pour s'assurer qu'il fournit des données brutes (non normalisées) pour le cash et l'equity, laissant `VecNormalize` gérer la normalisation.

### Phase 3 : Pipeline ADAN Corrigé (Fusion et Validation)

*   **FR-3.1 (Architecture de Fusion Progressive - `AdanFusionPipeline`) :**
    *   **Design :** Création d'une classe `src/adan_trading_bot/model/adan_fusion_pipeline.py`. Cette classe orchestrera :
        *   L'entraînement des 4 experts individuels (utilisant `train_adan_corrected.py` en mode "expert").
        *   Une phase d'entraînement collaboratif où les experts interagissent.
        *   Une phase de fusion où les experts sont combinés (ex: via un réseau de gating ou un méta-modèle) en un modèle ADAN unifié.
        *   Une phase de fine-tuning du modèle ADAN fusionné.
    *   Le modèle ADAN final sera un seul fichier `.zip` encapsulant la logique de fusion.
*   **FR-3.2 (Validation Croisée Temporelle - `TemporalCrossValidation`) :**
    *   **Design :** Création d'une classe `src/adan_trading_bot/validation/temporal_cross_validation.py`. Cette classe gérera la logique de split des données en périodes d'entraînement et de validation glissantes, et orchestrera le réentraînement partiel et la validation du modèle ADAN sur chaque période.

### Phase 4 : Entraînement Supervisé avec Corrections

*   **FR-4.1 (Script d'Entraînement Corrigé - `train_adan_corrected.py`) :**
    *   **Design :** Création d'un nouveau script `scripts/train_adan_corrected.py`. Ce script sera le point d'entrée principal pour l'entraînement. Il intégrera :
        *   L'initialisation des seeds via `SeedManager`.
        *   Le chargement des données via `DataManager.load_strict_split()`.
        *   L'instanciation de `RealisticTradingEnv` (enveloppé dans `VecNormalize`).
        *   L'intégration du `TradeFrequencyController` et du `RiskManager` (via l'environnement).
        *   L'orchestration de l'entraînement via `AdanFusionPipeline`.
        *   Des boucles d'entraînement par phases avec des validations intermédiaires.
        *   Des mécanismes pour appliquer des corrections automatiques au modèle si les métriques de validation le justifient.
*   **FR-4.2 (Checkpoints Intelligents - `IntelligentCheckpoint`) :**
    *   **Design :** Création d'une classe `src/adan_trading_bot/common/intelligent_checkpoint.py` qui héritera de `stable_baselines3.common.callbacks.BaseCallback`. Ce callback sera utilisé dans `train_adan_corrected.py` et implémentera la logique pour :
        *   Évaluer la performance, la stabilité et l'overfitting.
        *   Vérifier la progression de la fusion.
        *   Sauvegarder un checkpoint complet de l'état de fusion (experts + état de fusion) uniquement si les conditions sont remplies.

### Phase 5 : Validation Exhaustive

*   **FR-5.1 (Batterie de Tests Complète - `validation_suite`) :**
    *   **Design :** Création d'un répertoire `tests/validation_suite/` contenant des modules de tests unitaires et d'intégration spécifiques pour chaque aspect (reproductibilité, robustesse, comportement). Utilisation de `pytest`.
*   **FR-5.2 (Rapport de Validation Détaillé - `DetailedValidationReport`) :**
    *   **Design :** Création d'une classe `src/adan_trading_bot/validation/detailed_validation_report.py`. Cette classe collectera les métriques de toutes les phases de validation et générera un rapport HTML/PDF structuré, incluant des graphiques et des recommandations.
*   **FR-5.3 (Backtest "Out-of-Sample") :**
    *   **Design :** Le script `scripts/evaluate_model_comprehensive.py` sera modifié pour exécuter des backtests sur le `Test Set` (2024-Aujourd'hui) et vérifier les critères de succès (Sharpe > 1.0, Return positif, Drawdown < 30%).
*   **FR-5.4 (Script de "Check de Saturation") :**
    *   **Design :** Création d'un script `scripts/check_model_saturation.py` qui chargera un modèle entraîné, exécutera des prédictions sur un échantillon de données et analysera la distribution des sorties pour détecter une saturation binaire (sorties majoritairement ±1.0).
*   **FR-5.5 (Vérification de la Normalisation) :**
    *   **Design :** Intégration d'une étape dans le `DetailedValidationReport` et/ou `scripts/evaluate_model_comprehensive.py` pour tenter de charger un modèle sans son `vecnormalize.pkl` et vérifier que les prédictions sont aberrantes, confirmant ainsi la dépendance à la normalisation.

## 3. Alternatives Considérées

*   **Approche par "patchs" successifs :**
    *   **Raison du Rejet :** Cette approche a été identifiée comme la cause de la dette technique actuelle et des problèmes non résolus. Une refonte complète est nécessaire pour garantir la stabilité et la performance à long terme.
*   **Entraînement d'un seul modèle monolithique :**
    *   **Raison du Rejet :** L'architecture des 4 experts est une force. L'objectif est de fusionner leurs forces, pas de les remplacer par un modèle unique moins spécialisé.

## 4. Impacts

*   **Performance :** L'entraînement sera potentiellement plus long en raison de la complexité accrue de l'environnement réaliste et du pipeline de fusion. Cependant, la validation sera plus rapide et plus fiable.
*   **Complexité :** Augmentation significative de la complexité du code, mais compensée par une meilleure modularité et une documentation plus claire.
*   **Maintenabilité :** Amélioration de la maintenabilité grâce à une architecture bien définie, des composants spécialisés et des tests exhaustifs.
*   **Fiabilité :** Augmentation drastique de la fiabilité des modèles et des rapports de performance.
*   **Coûts :** Potentiellement des coûts de calcul plus élevés pour l'entraînement, mais réduction des risques de pertes en production.
