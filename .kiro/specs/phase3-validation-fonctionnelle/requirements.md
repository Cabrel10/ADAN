# Phase 3 - Validation Fonctionnelle: Requirements

## Introduction

Phase 3 valide que le système ADAN corrigé (Phase 2) peut effectuer des prédictions cohérentes et fonctionner correctement en mode paper trading. Cette phase établit la confiance dans le système avant l'entraînement MVP et la validation out-of-sample.

L'objectif principal est de vérifier que:
1. Les 4 modèles PPO chargés font des prédictions valides
2. Le système peut exécuter des itérations de paper trading sans erreurs
3. Les décisions du modèle sont cohérentes et non aléatoires
4. L'état du système peut être sauvegardé et restauré correctement

## Glossary

- **VecNormalize**: Wrapper Stable-Baselines3 qui normalise les observations selon les statistiques d'entraînement
- **Worker**: Instance d'agent de trading (w1, w2, w3, w4) avec son propre modèle PPO
- **Paper Trading**: Simulation de trading sans appels API réels
- **Covariate Shift**: Divergence entre distribution d'entraînement et d'inférence (résolu en Phase 2)
- **Observation**: État du marché normalisé (5m, 1h, 4h + portfolio_state)
- **Action**: Décision du modèle (25 dimensions pour 25 paires de trading)
- **Dry-Run**: Exécution de test sans effets réels

## Requirements

### Requirement 1: Test d'Inférence Basique

**User Story:** En tant que développeur, je veux vérifier que les modèles PPO peuvent faire des prédictions valides après la correction Phase 2, afin de confirmer que le système d'inférence fonctionne correctement.

#### Acceptance Criteria

1. WHEN le monitor est initialisé THEN le système SHALL charger les 4 modèles PPO sans erreurs
2. WHEN le monitor est initialisé THEN le système SHALL charger les 4 fichiers VecNormalize avec training=False
3. WHEN build_observation() est appelé avec un worker_id valide THEN le système SHALL retourner une observation Dict avec les clés requises (5m, 1h, 4h, portfolio_state)
4. WHEN build_observation() est appelé THEN le système SHALL normaliser les observations en utilisant le VecNormalize du worker correspondant
5. WHEN model.predict() est appelé avec une observation valide THEN le système SHALL retourner une action de shape (25,) sans NaN
6. WHEN model.predict() est appelé THEN le système SHALL retourner des actions dans la plage [-1.1, 1.1]

### Requirement 2: Paper Trading Dry-Run

**User Story:** En tant que développeur, je veux exécuter 100 itérations de paper trading sans appels API réels, afin de valider que le système peut boucler correctement et générer des décisions.

#### Acceptance Criteria

1. WHEN le dry-run démarre THEN le système SHALL initialiser l'état du portfolio correctement
2. WHEN chaque itération s'exécute THEN le système SHALL charger les données de marché pour les 3 timeframes
3. WHEN chaque itération s'exécute THEN le système SHALL générer une observation pour chaque worker
4. WHEN chaque itération s'exécute THEN le système SHALL obtenir une prédiction de chaque modèle
5. WHEN 100 itérations sont complétées THEN le système SHALL terminer sans erreurs
6. WHEN le dry-run est complété THEN le système SHALL générer un rapport avec les statistiques d'exécution

### Requirement 3: Analyse des Décisions

**User Story:** En tant que développeur, je veux analyser les décisions générées par les modèles, afin de vérifier qu'elles ne sont pas aléatoires et reflètent la logique du modèle.

#### Acceptance Criteria

1. WHEN les décisions sont collectées THEN le système SHALL calculer la moyenne et l'écart-type pour chaque dimension d'action
2. WHEN les décisions sont analysées THEN le système SHALL vérifier que l'écart-type n'est pas proche de zéro (pas de décisions figées)
3. WHEN les décisions sont analysées THEN le système SHALL vérifier que la distribution n'est pas uniforme (pas de décisions aléatoires)
4. WHEN les décisions sont comparées entre workers THEN le système SHALL identifier les patterns cohérents vs divergents
5. WHEN les décisions sont analysées THEN le système SHALL générer un rapport de cohérence

### Requirement 4: Test Génération État JSON

**User Story:** En tant que développeur, je veux vérifier que l'état du système peut être sérialisé en JSON et restauré correctement, afin de valider la persistance des données.

#### Acceptance Criteria

1. WHEN l'état du système est généré THEN le système SHALL créer un objet JSON valide avec toutes les informations requises
2. WHEN l'état JSON est sauvegardé THEN le système SHALL écrire le fichier sans erreurs
3. WHEN l'état JSON est chargé THEN le système SHALL restaurer l'état du système correctement
4. WHEN l'état est restauré THEN le système SHALL vérifier que les données correspondent aux données originales
5. WHEN l'état JSON est validé THEN le système SHALL vérifier la présence de tous les champs requis

### Requirement 5: Checkpoint de Validation

**User Story:** En tant que développeur, je veux avoir des checkpoints clairs pour valider chaque étape de Phase 3, afin de pouvoir identifier rapidement les problèmes.

#### Acceptance Criteria

1. WHEN Checkpoint 3.1 est exécuté THEN le système SHALL valider l'inférence basique et retourner un statut PASS/FAIL
2. WHEN Checkpoint 3.2 est exécuté THEN le système SHALL valider le dry-run et retourner un statut PASS/FAIL
3. WHEN Checkpoint 3.3 est exécuté THEN le système SHALL valider l'analyse des décisions et retourner un statut PASS/FAIL
4. WHEN Checkpoint 3.4 est exécuté THEN le système SHALL valider la génération d'état JSON et retourner un statut PASS/FAIL
5. WHEN tous les checkpoints sont PASS THEN le système SHALL générer un rapport final indiquant que Phase 3 est COMPLÈTE
