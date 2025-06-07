# Modèles de Référence (Baselines)

Ce répertoire contient les modèles et résultats de référence utilisés pour comparer les performances de l'agent ADAN.

## Contenu Typique

* Stratégies simples (buy and hold, momentum, etc.)
* Modèles statistiques classiques
* Résultats de backtests de référence
* Benchmarks de performance

## Conventions de Nommage

Les fichiers suivent généralement la convention de nommage:
`{STRATEGY_TYPE}_{DESCRIPTION}_{VERSION}_{DATE}.{FORMAT}`

Exemples:
* `buy_and_hold_results_v1_20231020.json`
* `momentum_strategy_v1_20231105.pkl`
* `moving_average_crossover_v2_20231210.json`

## Importance

Ces modèles de référence sont essentiels pour:
1. Évaluer objectivement les performances de l'agent RL
2. Identifier les conditions de marché où l'agent surperforme ou sous-performe
3. Fournir un point de comparaison pour les améliorations incrémentales
4. Établir un seuil minimum de performance acceptable
