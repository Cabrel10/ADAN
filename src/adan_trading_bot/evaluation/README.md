# Module `evaluation`

Ce module fournit les outils nécessaires pour évaluer les performances des agents entraînés via le backtesting et le calcul de métriques de performance.

## Contenu

* `backtester.py`: Logique de backtesting sur données historiques pour simuler le comportement de l'agent.
* `performance_metrics.py`: Calcul des métriques de performance financière (Sharpe, PnL, Drawdown, etc.).
* `plotting.py`: Fonctions de visualisation des résultats pour générer des graphiques et figures.

## Fonctionnalités Principales

### Backtesting
- Simulation du comportement de l'agent sur des données historiques non vues
- Gestion réaliste des ordres, des frais et du slippage
- Calcul du portefeuille et des positions au fil du temps
- Enregistrement détaillé des transactions et des décisions

### Métriques de Performance
- Calcul du ratio de Sharpe, Sortino, Calmar
- Analyse du PnL (Profit and Loss) total et annualisé
- Mesure du drawdown maximum et de sa durée
- Statistiques de trading (nombre de trades, win rate, profit factor)
- Comparaison avec des benchmarks (buy and hold, autres stratégies)

### Visualisation
- Courbes de PnL et de valeur du portefeuille
- Graphiques de drawdown
- Heatmaps des actions de l'agent
- Visualisation des trades sur les données de prix
- Distributions des rendements et des métriques

## Exemple d'Utilisation

```python
from adan_trading_bot.evaluation.backtester import Backtester
from adan_trading_bot.evaluation.performance_metrics import calculate_metrics
from adan_trading_bot.evaluation.plotting import plot_portfolio_performance

# Créer le backtester
backtester = Backtester(
    agent_path="models/rl_agents/ppo_agent_v1.zip",
    data_path="data/processed/test_data_v1_processed.parquet",
    config_path="config/environment_config.yaml"
)

# Exécuter le backtest
results = backtester.run()

# Calculer les métriques de performance
metrics = calculate_metrics(
    portfolio_values=results["portfolio_values"],
    benchmark_values=results["benchmark_values"],
    risk_free_rate=0.02
)

# Générer les visualisations
plot_portfolio_performance(
    results=results,
    metrics=metrics,
    save_path="reports/figures/backtest_results_v1.png"
)

# Sauvegarder les métriques
metrics.to_json("reports/metrics/backtest_metrics_v1.json")
```

## Intégration avec les Rapports

Ce module s'intègre avec le dossier `reports/` pour générer des rapports complets:
- Les figures sont sauvegardées dans `reports/figures/`
- Les métriques sont exportées dans `reports/metrics/`
- Les résultats détaillés peuvent être utilisés pour générer des rapports HTML ou PDF
