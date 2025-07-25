# Guide des Tiers de Capital

## Introduction
Ce document explique le système de gestion des capitaux par paliers (tiers) dans ADAN Trading Bot. Ce système permet d'adapter dynamiquement la gestion du risque en fonction de la taille du portefeuille.

## Configuration des Tiers

### Structure de configuration
Chaque tier est défini avec les paramètres suivants dans `config.yaml` :

```yaml
capital_tiers:
  - name: "Micro Capital"
    min_capital: 0
    max_capital: 30
    max_position_size_pct: 90.0
    leverage: 1.0
    risk_per_trade_pct: 2.0
    max_drawdown_pct: 15.0
  # ... autres tiers
```

### Paramètres
- `name` : Nom du tier
- `min_capital` / `max_capital` : Plage de capital pour ce tier
- `max_position_size_pct` : Taille de position maximale en % du capital
- `leverage` : Effet de levier autorisé
- `risk_per_trade_pct` : Risque maximal par trade en % du capital
- `max_drawdown_pct` : Drawdown maximum avant liquidation

## Simulation de Comportement

### Exécution d'une simulation
```bash
python scripts/simulate_capital_tiers.py
```

### Résultats attendus
- Graphique d'évolution du portefeuille
- Transitions entre les tiers
- Métriques de performance

## Intégration dans le Backtest

### Fichiers clés
- `portfolio/portfolio_manager.py` : Implémentation de la logique des tiers
- `tests/unit/portfolio/test_portfolio_manager.py` : Tests unitaires
- `scripts/simulate_capital_tiers.py` : Script de simulation

### Points d'extension
1. Ajouter des métriques personnalisées
2. Implémenter des stratégies de trading spécifiques
3. Adapter les règles de gestion du risque

## Dépannage

### Problèmes courants
1. **Tier non trouvé** : Vérifier les chevauchements dans `min_capital`/`max_capital`
2. **Liquidations fréquentes** : Ajuster `max_drawdown_pct` ou `risk_per_trade_pct`
3. **Taille de position trop petite** : Vérifier `min_trade_size` dans la configuration

## Améliorations futures
- [ ] Intégration avec le système de backtesting
- [ ] Optimisation des paramètres par tier
- [ ] Support multi-devises
