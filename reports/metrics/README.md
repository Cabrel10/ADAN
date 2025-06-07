# Métriques de Performance

Ce répertoire contient les fichiers de métriques quantitatives de performance pour le projet ADAN.

## Contenu Typique

* Résumés de backtest (ratio de Sharpe, rendement total, volatilité)
* Statistiques de trading (nombre de trades, win rate, profit factor)
* Métriques de drawdown (maximum drawdown, durée moyenne)
* Comparaisons avec les benchmarks
* Logs d'entraînement et courbes d'apprentissage

## Conventions de Nommage

Les fichiers suivent généralement la convention de nommage:
`{METRIC_TYPE}_{MODEL_VERSION}_{DATE}.{FORMAT}`

Exemples:
* `backtest_summary_v1_20231026.json`
* `training_metrics_ppo_v2_20231115.csv`
* `benchmark_comparison_v3_20231201.json`

## Format

Les métriques sont généralement stockées dans des formats structurés comme JSON ou CSV pour faciliter:
1. Le chargement et l'analyse ultérieure
2. La comparaison entre différentes versions de modèles
3. La génération automatique de rapports
4. Le suivi de l'évolution des performances au fil du temps
