# Tests du Projet ADAN

Ce répertoire contient les tests unitaires et d'intégration pour assurer la robustesse et la fiabilité du code du projet ADAN.

## Structure

* `unit/`: Tests unitaires pour les composants individuels
* `integration/`: Tests d'intégration pour vérifier l'interaction entre les différents modules
* `conftest.py`: Fixtures Pytest partagées entre les différents tests

## Exécution des Tests

Pour exécuter tous les tests:

```bash
# Depuis la racine du projet
pytest tests/

# Avec couverture de code
pytest tests/ --cov=src/adan_trading_bot
```

Pour exécuter un groupe spécifique de tests:

```bash
# Tests unitaires uniquement
pytest tests/unit/

# Tests d'un module spécifique
pytest tests/unit/test_environment.py
```

## Bonnes Pratiques

1. Chaque module du code source doit avoir des tests unitaires correspondants
2. Les tests doivent être indépendants et ne pas dépendre de l'ordre d'exécution
3. Utilisez des fixtures pour réutiliser la configuration et les données de test
4. Moquez les dépendances externes (API, bases de données) pour des tests plus rapides et plus fiables
5. Visez une couverture de code élevée, en particulier pour les composants critiques
