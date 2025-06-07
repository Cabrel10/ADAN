# Notebooks Jupyter

Ce répertoire contient les notebooks Jupyter utilisés pour l'exploration, le prototypage et la visualisation dans le cadre du projet ADAN.

## Contenu

* `0_setup_and_config.ipynb`: Configuration initiale et chargement des paramètres
* `1_data_exploration.ipynb`: Exploration et analyse des données brutes
* `2_feature_engineering.ipynb`: Test et validation des indicateurs techniques
* `3_environment_testing.ipynb`: Tests unitaires de l'environnement RL
* `4_agent_prototyping.ipynb`: Prototypage rapide d'un agent simple
* `5_results_visualization.ipynb`: Visualisation des résultats de backtest

## Conventions de Nommage

Les notebooks sont numérotés pour indiquer l'ordre logique d'exécution et suivent la convention:
`{NUMERO}_{DESCRIPTION}.ipynb`

## Bonnes Pratiques

1. **Reproductibilité**: Fixez les seeds aléatoires et documentez toutes les dépendances
2. **Modularité**: Importez les fonctions du package `src/adan_trading_bot/` plutôt que de dupliquer le code
3. **Documentation**: Incluez des explications claires pour chaque cellule et visualisation
4. **Nettoyage**: Exécutez "Restart and Run All" avant de sauvegarder pour vérifier la cohérence
5. **Versions**: Créez de nouveaux notebooks pour les changements majeurs plutôt que de modifier les existants

## Remarque

Ces notebooks sont principalement destinés à l'exploration et au prototypage. Le code stable et réutilisable doit être déplacé vers les modules appropriés dans `src/adan_trading_bot/`.
