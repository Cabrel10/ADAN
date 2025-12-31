# Phase 3 - Validation Fonctionnelle: Design

## Overview

Phase 3 valide le système ADAN corrigé en Phase 2 à travers 4 checkpoints progressifs:

1. **Checkpoint 3.1**: Inférence basique - Vérifier que les modèles font des prédictions valides
2. **Checkpoint 3.2**: Paper trading dry-run - Exécuter 100 itérations sans API
3. **Checkpoint 3.3**: Analyse des décisions - Vérifier la cohérence des prédictions
4. **Checkpoint 3.4**: Génération d'état JSON - Valider la persistance des données

Chaque checkpoint est indépendant mais progressif, permettant d'identifier rapidement les problèmes.

## Architecture

```
Phase 3 Validation Fonctionnelle
├── Checkpoint 3.1: Test d'Inférence Basique
│   ├── Initialisation du monitor
│   ├── Vérification VecNormalize (4 workers)
│   ├── Chargement des données de test
│   ├── Test build_observation() pour chaque worker
│   ├── Test model.predict() pour chaque worker
│   └── Sauvegarde des résultats
│
├── Checkpoint 3.2: Paper Trading Dry-Run
│   ├── Initialisation de l'état du portfolio
│   ├── Boucle 100 itérations:
│   │   ├── Charger données de marché (5m, 1h, 4h)
│   │   ├── Générer observations (4 workers)
│   │   ├── Obtenir prédictions (4 modèles)
│   │   └── Mettre à jour l'état
│   ├── Collecte des statistiques
│   └── Génération du rapport
│
├── Checkpoint 3.3: Analyse des Décisions
│   ├── Charger les décisions collectées
│   ├── Calculer statistiques (mean, std, min, max)
│   ├── Vérifier cohérence (pas de figement, pas d'aléatoire)
│   ├── Comparer patterns entre workers
│   └── Génération du rapport de cohérence
│
└── Checkpoint 3.4: Génération État JSON
    ├── Créer objet état complet
    ├── Sérialiser en JSON
    ├── Sauvegarder le fichier
    ├── Charger et valider
    └── Vérifier intégrité des données
```

## Components and Interfaces

### 1. Test d'Inférence Basique (Checkpoint 3.1)

**Fichier**: `scripts/test_inference_basic.py` (déjà créé et validé ✅)

**Interface**:
```python
def test_monitor_initialization() -> RealPaperTradingMonitor
def test_vecnormalize_loading(monitor) -> bool
def test_data_loading() -> dict
def test_build_observation(monitor, data) -> dict
def test_model_predictions(monitor, observations) -> dict
def save_results(predictions) -> dict
```

**Résultat**: ✅ VALIDÉ - 4/4 workers fonctionnels

### 2. Paper Trading Dry-Run (Checkpoint 3.2)

**Fichier**: `scripts/test_paper_trading_dryrun.py` (À créer)

**Interface**:
```python
class PaperTradingDryRun:
    def __init__(self, num_iterations: int = 100)
    def initialize_portfolio() -> dict
    def run_iteration(iteration: int) -> dict
    def collect_statistics() -> dict
    def generate_report() -> dict
```

**Logique**:
- Initialiser l'état du portfolio (cash, positions, etc.)
- Boucler 100 fois:
  - Charger les données de marché
  - Générer observations pour chaque worker
  - Obtenir prédictions
  - Mettre à jour l'état
- Collecter les statistiques (actions, temps, erreurs)
- Générer un rapport

### 3. Analyse des Décisions (Checkpoint 3.3)

**Fichier**: `scripts/analyze_decisions.py` (À créer)

**Interface**:
```python
class DecisionAnalyzer:
    def __init__(self, decisions_file: str)
    def calculate_statistics() -> dict
    def check_coherence() -> dict
    def compare_workers() -> dict
    def generate_report() -> dict
```

**Logique**:
- Charger les décisions collectées
- Calculer mean, std, min, max pour chaque dimension
- Vérifier que std > 0.01 (pas figé)
- Vérifier que distribution n'est pas uniforme
- Comparer patterns entre workers
- Générer rapport

### 4. Génération État JSON (Checkpoint 3.4)

**Fichier**: `scripts/test_state_serialization.py` (À créer)

**Interface**:
```python
class StateSerializer:
    def __init__(self, monitor: RealPaperTradingMonitor)
    def generate_state() -> dict
    def serialize_to_json() -> str
    def save_to_file(filepath: str) -> bool
    def load_from_file(filepath: str) -> dict
    def validate_state(state: dict) -> bool
```

**Logique**:
- Créer objet état avec toutes les informations
- Sérialiser en JSON
- Sauvegarder et charger
- Valider que les données correspondent

## Data Models

### Observation Model
```python
observation = {
    '5m': np.ndarray(shape=(20, 15)),      # 20 barres 5m, 15 features
    '1h': np.ndarray(shape=(10, 15)),      # 10 barres 1h, 15 features
    '4h': np.ndarray(shape=(5, 15)),       # 5 barres 4h, 15 features
    'portfolio_state': np.ndarray(shape=(20,))  # État du portfolio
}
```

### Action Model
```python
action = np.ndarray(shape=(25,))  # 25 paires de trading, valeurs [-1, 1]
```

### Decision Record
```python
decision = {
    'iteration': int,
    'timestamp': str,
    'worker_id': str,
    'action': list,  # 25 dimensions
    'action_mean': float,
    'action_std': float,
    'action_min': float,
    'action_max': float
}
```

### State Model
```python
state = {
    'timestamp': str,
    'phase': str,
    'checkpoint': str,
    'portfolio': {
        'cash': float,
        'positions': dict,
        'total_value': float
    },
    'models': {
        'w1': {'status': str, 'predictions': int},
        'w2': {'status': str, 'predictions': int},
        'w3': {'status': str, 'predictions': int},
        'w4': {'status': str, 'predictions': int}
    },
    'statistics': {
        'total_iterations': int,
        'successful_iterations': int,
        'failed_iterations': int,
        'avg_execution_time': float
    }
}
```

## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property 1: Inférence Déterministe
*For any* observation valide et worker_id, appeler model.predict() deux fois avec deterministic=True SHALL retourner la même action.

**Validates: Requirements 1.5, 1.6**

### Property 2: Normalisation Cohérente
*For any* observation brute et worker_id, normaliser avec VecNormalize SHALL produire des valeurs dans la plage [-3, 3] (standard normal).

**Validates: Requirements 1.4**

### Property 3: Actions Valides
*For any* prédiction du modèle, l'action SHALL avoir shape (25,) et toutes les valeurs SHALL être dans [-1.1, 1.1].

**Validates: Requirements 1.5, 1.6**

### Property 4: Dry-Run Complète
*For any* exécution de 100 itérations, le système SHALL compléter sans erreurs et générer au moins 100 décisions.

**Validates: Requirements 2.5, 2.6**

### Property 5: Cohérence des Décisions
*For any* ensemble de décisions collectées, l'écart-type de chaque dimension SHALL être > 0.01 (pas figé) et < 0.5 (pas aléatoire).

**Validates: Requirements 3.2, 3.3**

### Property 6: Sérialisation Round-Trip
*For any* état du système, sérialiser en JSON puis désérialiser SHALL produire un état équivalent.

**Validates: Requirements 4.1, 4.2, 4.3, 4.4**

## Error Handling

### Checkpoint 3.1 Errors
- **Modèle non chargé**: Vérifier que le fichier .zip existe et est valide
- **VecNormalize incompatible**: Vérifier que les shapes correspondent
- **Observation invalide**: Vérifier que les données de marché sont disponibles
- **Prédiction NaN**: Vérifier que l'observation est normalisée correctement

### Checkpoint 3.2 Errors
- **Données manquantes**: Utiliser des données simulées si nécessaire
- **Erreur d'itération**: Logger l'erreur et continuer
- **Timeout**: Définir un timeout par itération (5 secondes)
- **Mémoire**: Monitorer l'utilisation mémoire

### Checkpoint 3.3 Errors
- **Fichier décisions manquant**: Générer les décisions à partir du dry-run
- **Données insuffisantes**: Nécessite au moins 50 décisions
- **Calcul statistique**: Gérer les cas edge (std=0, etc.)

### Checkpoint 3.4 Errors
- **Sérialisation JSON**: Vérifier que tous les types sont JSON-compatibles
- **Fichier non accessible**: Créer les répertoires si nécessaire
- **Validation échouée**: Comparer les champs clés

## Testing Strategy

### Unit Tests
- Test chaque fonction de normalisation
- Test la sérialisation JSON
- Test les calculs statistiques
- Test les validations

### Property-Based Tests
- **Property 1**: Inférence déterministe (100 itérations)
- **Property 2**: Normalisation cohérente (100 itérations)
- **Property 3**: Actions valides (100 itérations)
- **Property 4**: Dry-run complète (1 itération)
- **Property 5**: Cohérence des décisions (1 itération)
- **Property 6**: Sérialisation round-trip (100 itérations)

### Integration Tests
- Test complet du pipeline (Checkpoint 3.1 → 3.2 → 3.3 → 3.4)
- Test avec données réelles
- Test avec données simulées

## Success Criteria

Phase 3 est considérée comme **COMPLÈTE** si:

1. ✅ Checkpoint 3.1: 4/4 workers font des prédictions valides
2. ✅ Checkpoint 3.2: 100 itérations complétées sans erreurs
3. ✅ Checkpoint 3.3: Décisions cohérentes (std > 0.01, pas aléatoires)
4. ✅ Checkpoint 3.4: État JSON valide et round-trip réussi
5. ✅ Tous les tests passent
6. ✅ Rapport final généré

**Résultat Actuel**: Checkpoint 3.1 ✅ VALIDÉ
