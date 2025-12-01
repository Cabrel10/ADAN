# Exigences (Requirements) : Plan de Reconstruction Complet ADAN 2.0

## 1. Problème Général

Le projet ADAN a rencontré des problèmes critiques qui ont conduit à des performances non fiables et à des diagnostics erronés. Ces problèmes incluent :
*   **Overfitting massif et fuite de données :** Entraînement et validation sur des données non représentatives ou contaminées.
*   **Divergence Train/Live :** L'environnement d'entraînement ne reflète pas la réalité du trading en direct.
*   **Instabilité du Modèle :** Comportement erratique dû à une fonction de récompense complexe et instable.
*   **Trades Intempestifs :** Fréquence de trading excessive ou inappropriée.
*   **Fusion Incorrecte :** Absence d'un pipeline de fusion intégré pour les modèles experts.
*   **Manque de Reproductibilité :** Incapacité à obtenir des résultats cohérents entre les exécutions.

## 2. Objectifs Généraux

L'objectif de ce plan de reconstruction est de créer un bot de trading ADAN 2.0 qui soit :
*   **Robuste :** Capable de fonctionner de manière fiable dans diverses conditions de marché.
*   **Reproductible :** Les résultats d'entraînement et de validation doivent être cohérents.
*   **Réaliste :** L'entraînement doit simuler fidèlement les conditions du trading en direct.
*   **Unifié :** Les modèles experts doivent être fusionnés en une entité ADAN cohérente.
*   **Stable :** Le comportement du modèle doit être prévisible et contrôlé.
*   **Performant :** Atteindre des critères de succès clairs en validation out-of-sample.

## 3. Exigences Fonctionnelles Détaillées

### Phase 0 : Audit Complet des Erreurs Critiques

*   **FR-0.1 : Analyse Post-Mortem des Échecs**
    *   Le système doit permettre d'identifier et de documenter les symptômes, causes et solutions des erreurs critiques passées (trades intempestifs, discrepancy train/live, data leakage, fusion incorrecte, reward instable).
*   **FR-0.2 : Vérification de la Reproductibilité**
    *   Un script d'audit doit être capable de vérifier la cohérence des seeds, des splits de données, de l'environnement et du hachage des modèles.

### Phase 1 : Purification de l'Environnement (Données et Configuration)

*   **FR-1.1 : Reconstruction de l'Environnement Conda**
    *   L'environnement de développement doit être reconstruit à l'aide d'un fichier `environment_strict.yaml` avec des versions de dépendances verrouillées pour assurer la reproductibilité.
*   **FR-1.2 : Gestion Globale des Graines Aléatoires**
    *   Un module `reproducibility.py` (avec une classe `SeedManager`) doit être implémenté pour fixer les graines aléatoires de `random`, `numpy`, `torch` (CPU et CUDA), et l'environnement Gym/Stable-Baselines3.
*   **FR-1.3 : Nettoyage Radical des Artefacts**
    *   Un processus doit être mis en place pour supprimer tous les dossiers `data/processed`, `checkpoints`, et `logs` avant chaque nouvelle exécution majeure.
*   **FR-1.4 : Séparation Temporelle Stricte des Données**
    *   Le pipeline de données doit garantir une séparation temporelle stricte :
        *   **Train Set :** 01/01/2021 -> 31/12/2023.
        *   **Test Set :** 01/01/2024 -> Aujourd'hui.
    *   Aucune chevauchée de dates entre les ensembles d'entraînement et de test ne doit être permise.
*   **FR-1.5 : Correction et Vérification des Features**
    *   Le fichier `feature_engineer.py` doit être modifié pour inclure explicitement les indicateurs `ATR_20`, `ATR_50` et `ADX` dans les données traitées (parquets).

### Phase 2 : Corrections Architecturales Profondes (Environnement et Récompense)

*   **FR-2.1 : Environnement d'Entraînement Réaliste (`RealisticTradingEnv`)**
    *   L'environnement d'entraînement doit intégrer des modèles réalistes pour :
        *   Les frais de transaction (`BinanceFeeModel`).
        *   Le slippage (`AdaptiveSlippage`).
        *   La latence réseau (`LatencySimulator`).
        *   La profondeur de marché (`LiquidityModel`).
    *   Ces éléments doivent être appliqués séquentiellement et de manière réaliste à chaque étape.
*   **FR-2.2 : Contrôle de la Fréquence des Trades (`TradeFrequencyController`)**
    *   Un contrôleur doit être implémenté pour :
        *   Appliquer un intervalle minimum entre les trades (ex: 10 steps).
        *   Limiter le nombre de trades par jour (ex: 24 trades max).
        *   Gérer des périodes de cooldown par actif.
*   **FR-2.3 : Fonction de Récompense Stabilisée (`StableRewardCalculator`)**
    *   La fonction de récompense doit être simplifiée, normalisée et inclure des pénalités claires pour :
        *   PnL normalisé.
        *   Contribution Sharpe.
        *   Pénalité de Drawdown.
        *   Pénalité de fréquence de trades.
        *   Bonus de consistance.
    *   La récompense finale doit être clippée (ex: entre -1.0 et 1.0) pour éviter l'explosion des gradients.
*   **FR-2.4 : Intégration Obligatoire de `VecNormalize`**
    *   L'environnement d'entraînement doit être systématiquement enveloppé dans `VecNormalize(norm_obs=True, norm_reward=True)`.
*   **FR-2.5 : Sauvegarde Automatique de `vecnormalize.pkl`**
    *   Le fichier `vecnormalize.pkl` doit être automatiquement sauvegardé à la fin de l'entraînement dans le même dossier que le modèle entraîné.
*   **FR-2.6 : Alignement du `Portfolio State`**
    *   Le `StateBuilder` doit être configuré pour envoyer des données brutes (Cash, Equity) à l'environnement, laissant `VecNormalize` gérer la mise à l'échelle.

### Phase 3 : Pipeline ADAN Corrigé (Fusion et Validation)

*   **FR-3.1 : Architecture de Fusion Progressive (`AdanFusionPipeline`)**
    *   Un pipeline doit être implémenté pour gérer la fusion des 4 experts :
        *   Phase 1 : Entraînement des experts individuels.
        *   Phase 2 : Entraînement collaboratif.
        *   Phase 3 : Fusion et fine-tuning pour produire un modèle ADAN unifié.
*   **FR-3.2 : Validation Croisée Temporelle (`TemporalCrossValidation`)**
    *   Un mécanisme de validation croisée temporelle doit être mis en place, utilisant des périodes d'entraînement et de validation glissantes (ex: 2018-2020/2020-2021, etc.).

### Phase 4 : Entraînement Supervisé avec Corrections

*   **FR-4.1 : Script d'Entraînement Corrigé (`train_adan_corrected.py`)**
    *   Le script d'entraînement doit :
        *   Utiliser `SeedManager` pour une initialisation reproductible.
        *   Charger des données strictement séparées (train/test).
        *   Utiliser l'environnement réaliste (`RealisticTradingEnv`).
        *   Intégrer le `TradeFrequencyController` et le `RiskManager`.
        *   Permettre un entraînement par phases avec validation continue.
        *   Intégrer des procédures de correction automatique basées sur les résultats de validation intermédiaires.
*   **FR-4.2 : Checkpoints Intelligents (`IntelligentCheckpoint`)**
    *   Le système de checkpoints doit être intelligent, sauvegardant les modèles basés sur :
        *   L'amélioration de la performance.
        *   Une stabilité acceptable.
        *   Un contrôle de l'overfitting.
        *   La progression de la fusion.
    *   Les checkpoints doivent inclure l'état complet de la fusion.

### Phase 5 : Validation Exhaustive

*   **FR-5.1 : Batterie de Tests Complète (`validation_suite`)**
    *   Une suite de tests doit couvrir :
        *   **Reproductibilité :** Cohérence des seeds, absence de fuite de données, déterminisme de l'environnement.
        *   **Robustesse :** Performance sous stress (crash marché, faible liquidité), impact du slippage et des frais.
        *   **Comportement :** Respect des limites de fréquence, gestion des risques (SL, position sizing), performance dans différents régimes de marché.
*   **FR-5.2 : Rapport de Validation Détaillé (`DetailedValidationReport`)**
    *   Un rapport doit être généré, incluant :
        *   Métriques de performance et de risque.
        *   Analyse comportementale.
        *   Score de reproductibilité.
        *   Efficacité de la fusion.
        *   Recommandations.
*   **FR-5.3 : Backtest "Out-of-Sample"**
    *   Le modèle entraîné (2021-2023) doit être validé sur les données 2024 avec les critères de succès suivants :
        *   Sharpe Ratio > 1.0.
        *   Return positif.
        *   Drawdown < 30%.
*   **FR-5.4 : Script de "Check de Saturation"**
    *   Un script automatique post-entraînement doit vérifier si les sorties du modèle sont binaires (±1.0) et rejeter le modèle si c'est le cas.
*   **FR-5.5 : Vérification de la Normalisation**
    *   Un test doit confirmer que charger le modèle sans son fichier `vecnormalize.pkl` associé donne des résultats incohérents, prouvant l'activité de la normalisation.

## 4. Exigences Non-Fonctionnelles

*   **NF-1 : Performance :** Le pipeline d'entraînement et de validation doit rester dans des limites de temps acceptables.
*   **NF-2 : Reproductibilité :** Toute exécution du pipeline avec les mêmes inputs doit produire les mêmes outputs (modèles, métriques).
*   **NF-3 : Robustesse :** Le système doit être résilient aux conditions de marché extrêmes et aux données bruitées.
*   **NF-4 : Maintenabilité :** Le code doit être modulaire, bien documenté et facile à comprendre.
*   **NF-5 : Sécurité :** Le système doit intégrer des mécanismes de sécurité pour éviter les explosions numériques et les comportements aberrants.

## 5. Exclusions

*   L'intégration avec des plateformes de trading en direct (live trading) n'est pas couverte par ce plan de reconstruction initial.
*   L'optimisation des hyperparamètres spécifiques de chaque expert n'est pas détaillée ici, mais sera intégrée dans le pipeline d'entraînement.
