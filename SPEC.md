# Spécifications Techniques - ADAN Trading Bot

## 🎯 Objectifs du Projet
Développer un système de trading algorithmique avancé avec les caractéristiques suivantes :
- Support multi-actifs (cryptomonnaies, forex, actions)
- Stratégies basées sur du Deep Reinforcement Learning
- Gestion avancée du risque et du capital
- Backtesting robuste
- Intégration avec les principales plateformes d'échange

## 🏗 Architecture Technique

### Stack Technologique
- **Langage** : Python 3.9+
- **ML/DL** : PyTorch, Stable Baselines3
- **Traitement des données** : Pandas, NumPy, Dask
- **Backtesting** : Backtrader, VectorBT
- **Visualisation** : Plotly, Matplotlib
- **API** : FastAPI, WebSockets
- **Base de données** : TimescaleDB, Redis (cache)
- **Orchestration** : Prefect, Docker
- **Monitoring** : Prometheus, Grafana

## 🏗 Architecture Technique

### 1. StateBuilder
**Rôle** : Transforme les données brutes en observations pour l'agent RL

**Format d'entrée** :
```python
{
    "market_data": {
        "prices": np.ndarray,      # Prix normalisés [window_size, n_assets]
        "volumes": np.ndarray,     # Volumes normalisés [window_size, n_assets]
        "indicators": {            # 20+ indicateurs techniques par actif
            "rsi_14": np.ndarray,  # [window_size, n_assets]
            "bb_upper": np.ndarray,
            # ...
        }
    },
    "portfolio": {
        "cash": float,
        "positions": Dict[str, float],  # Positions actuelles par actif
        "value_history": np.ndarray,    # Historique des valeurs du portefeuille
        "returns": np.ndarray           # Rendements historiques
    }
}
```

**Format de sortie** :
- **Shape attendue** : `(3, window_size, n_features)`
  - **Dimension 1** : Timeframe (5m, 1h, 4h)
  - **Dimension 2** : Fenêtre temporelle (par défaut 30 pas)
  - **Dimension 3** : Features (prix, volumes + indicateurs techniques)

### 2. MultiAssetChunkedEnv
**Rôle** : Environnement de trading multi-actifs avec chargement par chunks

**Comportement clé** :
- Charge les données par blocs pour gérer la mémoire
- Gère le warm-up initial pour remplir la fenêtre d'observation
- Applique les règles de gestion du risque dynamique

**Problème actuel** : Incohérence entre `_setup_spaces()` et la sortie du StateBuilder

### Format de l'État (Observation)
```python
{
    "market_data": {
        "prices": np.ndarray,      # Prix normalisés
        "volumes": np.ndarray,     # Volumes normalisés
        "indicators": {            # Indicateurs techniques
            "rsi": np.ndarray,
            "macd": np.ndarray,
            # ...
        }
    },
    "portfolio": {
        "cash": float,
        "positions": Dict[str, float],
        "value_history": np.ndarray,
        "returns": np.ndarray
    },
    "current_step": int,
    "timestamp": "ISO8601"
}
```

## 🔄 Workflow de Développement

1. **Préparation des Données**
   - Collecte des données historiques
   - Nettoyage et normalisation
   - Feature engineering

2. **Développement des Stratégies**
   - Implémentation des indicateurs techniques
   - Définition des règles de trading
   - Intégration avec les modèles de ML/DL

3. **Backtesting**
   - Tests sur données historiques
   - Optimisation des paramètres
   - Analyse des performances

4. **Déploiement**
   - Mise en production
   - Monitoring en temps réel
   - Gestion des risques

## 🛠 Configuration Requise

### Prérequis
- Python 3.9+
- Conda/Miniconda
- Docker (optionnel pour le déploiement)
- Clés API pour les exchanges supportés

### Installation
```bash
# lancement de l'environnement Conda
conda activate trading_env



## 📈 Métriques de Performance
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Sortino Ratio
- Value at Risk (VaR)

## 🔒 Sécurité
- Gestion sécurisée des clés API
- Chiffrement des données sensibles
- Journalisation des opérations
- Authentification forte

## 📝 Documentation
- Documentation technique complète
- Guides d'utilisation
- Exemples de configuration
- FAQ

## 🚀 Roadmap
1. **Phase 1** : Mise en place de l'infrastructure de base
2. **Phase 2** : Implémentation des stratégies de base
3. **Phase 3** : Intégration du Deep Learning
4. **Phase 4** : Optimisation des performances
5. **Phase 5** : Déploiement et monitoring

## 📞 Support
Pour toute question ou problème, veuillez ouvrir une issue sur le dépôt du projet.
