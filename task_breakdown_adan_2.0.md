# Décomposition des Tâches (Task Breakdown) : Plan de Reconstruction Complet ADAN 2.0

Ce document détaille les tâches concrètes à réaliser pour chaque composant de la conception du plan de reconstruction ADAN 2.0.

## 1. Phase 0 : Audit Complet des Erreurs Critiques

*   **Tâche 0.1 : Analyse Post-Mortem des Échecs**
    *   [ ] **Implémenter la journalisation structurée (JSONL) :** Modifier `PortfolioManager`, `MultiAssetChunkedEnv`, `RewardCalculator` pour émettre des logs JSONL pour les événements critiques (trades, PnL, erreurs, changements de régime).
    *   [ ] **Développer le script d'analyse :** Créer `scripts/analyze_critical_errors.py` pour parser les logs JSONL et générer un rapport synthétique des erreurs critiques.

*   **Tâche 0.2 : Vérification de la Reproductibilité**
    *   [ ] **Développer le script d'audit :** Créer `scripts/audit_reproducibility.py` pour :
        *   Exécuter des runs courts avec des seeds spécifiques.
        *   Vérifier l'intégrité des splits de données.
        *   Comparer les outputs de l'environnement pour détecter le non-déterminisme.
        *   Calculer et comparer les hachages des modèles entraînés.

## 2. Phase 1 : Purification de l'Environnement (Données et Configuration)

*   **Tâche 1.1 : Reconstruction de l'Environnement Conda**
    *   [ ] **Créer `environment_strict.yaml` :** Lister toutes les dépendances Python avec leurs versions exactes.
    *   [ ] **Mettre à jour la documentation :** Inclure les instructions pour créer l'environnement Conda à partir de ce fichier.

*   **Tâche 1.2 : Gestion Globale des Graines Aléatoires**
    *   [ ] **Créer `src/adan_trading_bot/common/reproducibility.py` :** Implémenter la classe `SeedManager` avec les méthodes `set_global_seeds(seed)` et `seed_worker(worker_id)`.
    *   [ ] **Intégrer `SeedManager` :** Modifier tous les scripts d'entraînement et de validation pour appeler `SeedManager.set_global_seeds(42)` au démarrage.

*   **Tâche 1.3 : Nettoyage Radical des Artefacts**
    *   [ ] **Créer `scripts/clean_project.sh` :** Script shell pour supprimer `data/processed`, `checkpoints`, `logs`, et `optuna.db`.
    *   [ ] **Intégrer le nettoyage :** Ajouter l'appel à ce script au début des exécutions majeures du pipeline.

*   **Tâche 1.4 : Séparation Temporelle Stricte des Données**
    *   [ ] **Modifier `src/adan_trading_bot/data_processing/data_loader.py` :**
        *   Implémenter `load_strict_split(start_train, end_train, start_test, end_test)`.
        *   Ajouter des vérifications pour garantir l'absence de chevauchement et la continuité temporelle.
    *   [ ] **Mettre à jour les scripts de téléchargement/traitement des données :** S'assurer qu'ils respectent les nouvelles plages de dates (Train: 2021-2023, Test: 2024-Aujourd'hui).

*   **Tâche 1.5 : Correction et Vérification des Features**
    *   [ ] **Modifier `src/adan_trading_bot/data_processing/feature_engineer.py` :**
        *   Ajouter le calcul des indicateurs `ATR_20`, `ATR_50` et `ADX`.
        *   S'assurer que ces indicateurs sont inclus dans les DataFrames traités et sauvegardés.

## 3. Phase 2 : Corrections Architecturales Profondes (Environnement et Récompense)

*   **Tâche 2.1 : Environnement d'Entraînement Réaliste (`RealisticTradingEnv`)**
    *   [ ] **Créer `src/adan_trading_bot/environment/realistic_trading_env.py` :** Hériter de `MultiAssetChunkedEnv`.
    *   [ ] **Développer les modèles de friction :**
        *   [ ] `src/adan_trading_bot/trading_models/binance_fee_model.py`
        *   [ ] `src/adan_trading_bot/trading_models/adaptive_slippage.py`
        *   [ ] `src/adan_trading_bot/trading_models/latency_simulator.py`
        *   [ ] `src/adan_trading_bot/trading_models/liquidity_model.py`
    *   [ ] **Intégrer les modèles de friction :** Modifier la méthode `step()` de `RealisticTradingEnv` pour appliquer ces modèles séquentiellement.

*   **Tâche 2.2 : Contrôle de la Fréquence des Trades (`TradeFrequencyController`)**
    *   [ ] **Créer `src/adan_trading_bot/environment/trade_frequency_controller.py` :** Implémenter la logique de contrôle de fréquence (min_trade_interval, daily_trade_limit, cooldown_periods).
    *   [ ] **Intégrer le contrôleur :** Injecter une instance de `TradeFrequencyController` dans `RealisticTradingEnv` et appeler ses méthodes (`can_open_trade`, `can_close_trade`) avant toute exécution de trade.

*   **Tâche 2.3 : Fonction de Récompense Stabilisée (`StableRewardCalculator`)**
    *   [ ] **Créer `src/adan_trading_bot/environment/stable_reward_calculator.py` :** Remplacer l'actuel `RewardCalculator`.
    *   [ ] **Implémenter la logique de récompense :** Inclure PnL normalisé, contribution Sharpe, pénalités de Drawdown/fréquence, bonus de consistance.
    *   [ ] **Appliquer le clipping :** S'assurer que la récompense finale est clippée entre -1.0 et 1.0.
    *   [ ] **Intégrer le calculateur :** Modifier `RealisticTradingEnv` pour utiliser `StableRewardCalculator`.

*   **Tâche 2.4 : Intégration Obligatoire de `VecNormalize`**
    *   [ ] **Modifier `scripts/train_adan_corrected.py` (à créer) :** Envelopper l'instance de `RealisticTradingEnv` dans `stable_baselines3.common.vec_env.VecNormalize(norm_obs=True, norm_reward=True)`.

*   **Tâche 2.5 : Sauvegarde Automatique de `vecnormalize.pkl`**
    *   [ ] **Modifier `scripts/train_adan_corrected.py` :** Ajouter la logique pour sauvegarder l'objet `VecNormalize` (via `env.save(path)`) dans le même répertoire que le modèle PPO entraîné.

*   **Tâche 2.6 : Alignement du `Portfolio State`**
    *   [ ] **Modifier `src/adan_trading_bot/data_processing/state_builder.py` :** S'assurer que les données du portefeuille (Cash, Equity) sont fournies sous forme brute (non normalisée) à l'environnement.

## 4. Phase 3 : Pipeline ADAN Corrigé (Fusion et Validation)

*   **Tâche 3.1 : Architecture de Fusion Progressive (`AdanFusionPipeline`)**
    *   [ ] **Créer `src/adan_trading_bot/model/adan_fusion_pipeline.py` :**
        *   Implémenter la logique d'orchestration pour l'entraînement des experts individuels.
        *   Développer la phase d'entraînement collaboratif.
        *   Concevoir et implémenter la phase de fusion (ex: méta-modèle, réseau de gating).
        *   Implémenter la phase de fine-tuning du modèle ADAN unifié.
        *   S'assurer que le résultat est un seul fichier modèle ADAN unifié.

*   **Tâche 3.2 : Validation Croisée Temporelle (`TemporalCrossValidation`)**
    *   [ ] **Créer `src/adan_trading_bot/validation/temporal_cross_validation.py` :**
        *   Implémenter la logique de split des données en périodes d'entraînement et de validation glissantes.
        *   Orchestrer le réentraînement partiel et la validation du modèle ADAN sur chaque période.

## 5. Phase 4 : Entraînement Supervisé avec Corrections

*   **Tâche 4.1 : Script d'Entraînement Corrigé (`train_adan_corrected.py`)**
    *   [ ] **Créer `scripts/train_adan_corrected.py` :**
        *   Intégrer `SeedManager` pour l'initialisation.
        *   Utiliser `DataManager.load_strict_split()` pour le chargement des données.
        *   Instancier `RealisticTradingEnv` (enveloppé dans `VecNormalize`).
        *   Intégrer `TradeFrequencyController` et `RiskManager` (via l'environnement).
        *   Orchestrer l'entraînement via `AdanFusionPipeline`.
        *   Mettre en place des boucles d'entraînement par phases avec validations intermédiaires.
        *   Développer les mécanismes de correction automatique du modèle.

*   **Tâche 4.2 : Checkpoints Intelligents (`IntelligentCheckpoint`)**
    *   [ ] **Créer `src/adan_trading_bot/common/intelligent_checkpoint.py` :** Hériter de `stable_baselines3.common.callbacks.BaseCallback`.
    *   [ ] **Implémenter la logique de sauvegarde conditionnelle :** Évaluer performance, stabilité, overfitting, progression de la fusion.
    *   [ ] **Implémenter la sauvegarde de l'état de fusion :** Sauvegarder un checkpoint complet de l'état de fusion (experts + état de fusion).
    *   [ ] **Intégrer le callback :** Utiliser `IntelligentCheckpoint` dans `train_adan_corrected.py`.

## 6. Phase 5 : Validation Exhaustive

*   **Tâche 5.1 : Batterie de Tests Complète (`validation_suite`)**
    *   [ ] **Créer le répertoire `tests/validation_suite/` :**
    *   [ ] **Développer les tests unitaires/d'intégration :**
        *   [ ] Tests de reproductibilité (SeedManager, DataManager).
        *   [ ] Tests de robustesse (RealisticTradingEnv, TradeFrequencyController).
        *   [ ] Tests comportementaux (StableRewardCalculator, AdanFusionPipeline).

*   **Tâche 5.2 : Rapport de Validation Détaillé (`DetailedValidationReport`)**
    *   [ ] **Créer `src/adan_trading_bot/validation/detailed_validation_report.py` :**
        *   Implémenter la collecte des métriques de toutes les phases de validation.
        *   Développer la génération d'un rapport HTML/PDF structuré avec graphiques et recommandations.

*   **Tâche 5.3 : Backtest "Out-of-Sample"**
    *   [ ] **Modifier `scripts/evaluate_model_comprehensive.py` :**
        *   Adapter le script pour exécuter des backtests sur le `Test Set` (2024-Aujourd'hui).
        *   Intégrer la vérification des critères de succès (Sharpe > 1.0, Return positif, Drawdown < 30%).

*   **Tâche 5.4 : Script de "Check de Saturation"**
    *   [ ] **Créer `scripts/check_model_saturation.py` :**
        *   Charger un modèle entraîné.
        *   Exécuter des prédictions sur un échantillon de données.
        *   Analyser la distribution des sorties pour détecter une saturation binaire (sorties majoritairement ±1.0).

*   **Tâche 5.5 : Vérification de la Normalisation**
    *   [ ] **Intégrer la vérification :** Ajouter une étape dans `DetailedValidationReport` et/ou `scripts/evaluate_model_comprehensive.py` pour :
        *   Tenter de charger un modèle sans son `vecnormalize.pkl`.
        *   Vérifier que les prédictions sont aberrantes, confirmant la dépendance à la normalisation.

## 7. Checklist de Mise en Œuvre (Consolidée)

Cette checklist est une consolidation des tâches ci-dessus, organisée pour le suivi de l'exécution.

### ✅ PRÉPARATION
- [ ] Retour au commit stable 1959e76 (si applicable)
- [ ] **Tâche 1.1 :** Création de `environment_strict.yaml` et reconstruction de l'environnement conda verrouillé.
- [ ] **Tâche 1.3 :** Création de `scripts/clean_project.sh` et intégration du nettoyage radical.
- [ ] **Tâche 1.4 :** Modification de `data_loader.py` pour le split temporel strict des données.
- [ ] **Tâche 1.5 :** Modification de `feature_engineer.py` pour inclure `ATR_20`, `ATR_50`, `ADX`.
- [ ] **Tâche 1.2 :** Implémentation de `SeedManager` et intégration globale des graines aléatoires.

### ✅ CORRECTIONS FONDAMENTALES
- [ ] **Tâche 2.1 :** Implémentation de `RealisticTradingEnv` et des modèles de friction (fees, slippage, latence, liquidité).
- [ ] **Tâche 2.2 :** Implémentation et intégration de `TradeFrequencyController`.
- [ ] **Tâche 2.3 :** Implémentation et intégration de `StableRewardCalculator`.
- [ ] **Tâche 2.4 :** Intégration obligatoire de `VecNormalize` dans le script d'entraînement.
- [ ] **Tâche 2.5 :** Sauvegarde automatique de `vecnormalize.pkl` avec le modèle.
- [ ] **Tâche 2.6 :** Alignement du `StateBuilder` pour fournir des données brutes du portefeuille.

### ✅ PIPELINE ADAN CORRIGÉ
- [ ] **Tâche 3.1 :** Implémentation de `AdanFusionPipeline` (experts individuels, entraînement collaboratif, fusion, fine-tuning).
- [ ] **Tâche 3.2 :** Implémentation de `TemporalCrossValidation`.

### ✅ ENTRAÎNEMENT SUPERVISÉ
- [ ] **Tâche 4.1 :** Création de `scripts/train_adan_corrected.py` intégrant tous les nouveaux composants.
- [ ] **Tâche 4.2 :** Implémentation et intégration de `IntelligentCheckpoint`.

### ✅ VALIDATION EXHAUSTIVE
- [ ] **Tâche 5.1 :** Développement de la `validation_suite` (tests de reproductibilité, robustesse, comportement).
- [ ] **Tâche 5.2 :** Implémentation de `DetailedValidationReport`.
- [ ] **Tâche 5.3 :** Modification du script d'évaluation pour le backtest "Out-of-Sample".
- [ ] **Tâche 5.4 :** Implémentation de `scripts/check_model_saturation.py`.
- [ ] **Tâche 5.5 :** Intégration de la vérification de la normalisation.
