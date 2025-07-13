# DBE Analysis Notebook

Ce notebook est conçu pour analyser les logs du Dynamic Behavior Engine (DBE) et comprendre comment il module le comportement de l'agent de trading.

## Structure du Notebook

### 1. Chargement des données
```python
def load_dbe_logs(log_dir='../logs/dbe'):
    # Charge les journaux du DBE depuis le répertoire spécifié
    # Retourne un DataFrame pandas avec les données de log
    pass
```

### 2. Nettoyage et préparation des données
```python
def prepare_data(df):
    # Nettoie et transforme les données brutes
    # - Extrait les paramètres de modulation
    # - Extrait les métriques de risque
    # - Convertit les types de données
    # Retourne un DataFrame nettoyé
    pass
```

### 3. Visualisations

#### 3.1 Évolution des paramètres de trading
```python
def plot_trading_params(df):
    # Affiche l'évolution des paramètres SL/TP au fil du temps
    # - Graphique du stop loss en fonction du temps
    # - Graphique du take profit en fonction du temps
    pass
```

#### 3.2 Analyse des modes de risque
```python
def plot_risk_modes(df):
    # Affiche la distribution des modes de risque
    # - Diagramme en barres des différents modes
    # - Durée passée dans chaque mode
    pass
```

#### 3.3 Corrélation drawdown/risque
```python
def plot_drawdown_risk_correlation(df):
    # Affiche la corrélation entre le drawdown et le mode de risque
    # - Nuage de points drawdown vs mode de risque
    # - Ligne de tendance
    pass
```

#### 3.4 Analyse des récompenses
```python
def plot_reward_analysis(df):
    # Analyse des récompenses et pénalités
    # - Distribution des reward_boost
    # - Fréquence des pénalités d'inaction
    pass
```

### 4. Analyse des performances
```python
def analyze_performance(df):
    # Calcule les métriques de performance globales
    # - Taux de réussite par mode de risque
    - Drawdown moyen/max par mode
    - Rendement par mode
    pass
```
