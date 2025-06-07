# Module `data_processing`

Ce module gère toutes les opérations liées au chargement, au nettoyage, à la transformation des données et à l'ingénierie des features pour le projet ADAN.

## Contenu

* `data_loader.py`: Fonctions et classes pour charger les données depuis différentes sources (fichiers Parquet, CSV, API d'exchanges).
* `feature_engineer.py`: Calcul des indicateurs techniques, normalisation et standardisation des features.
* `data_splitter.py`: Outils pour diviser les données en ensembles d'entraînement, de validation et de test, avec gestion des séries temporelles.

## Fonctionnalités Principales

### Chargement des Données
- Lecture de fichiers Parquet et CSV
- Gestion des données manquantes
- Alignement temporel des données multi-actifs
- Conversion des formats de date/heure

### Ingénierie des Features
- Calcul d'indicateurs techniques (moyennes mobiles, RSI, MACD, Bollinger Bands, etc.)
- Normalisation et standardisation des données
- Création de features avancées (retours logarithmiques, volatilité, etc.)
- Sauvegarde et chargement des transformateurs (scalers, encoders)

### Division des Données
- Méthodes de division temporelle pour les séries chronologiques
- Validation croisée adaptée aux séries temporelles
- Création de fenêtres glissantes pour l'entraînement et l'évaluation

## Exemple d'Utilisation

```python
from adan_trading_bot.data_processing.data_loader import load_ohlcv_data
from adan_trading_bot.data_processing.feature_engineer import compute_technical_indicators, normalize_features
from adan_trading_bot.data_processing.data_splitter import time_series_split

# Charger les données
raw_data = load_ohlcv_data("data/raw/BTCUSDT_1h_raw.parquet")

# Calculer les indicateurs techniques
data_with_features = compute_technical_indicators(raw_data)

# Normaliser les features
normalized_data, scaler = normalize_features(data_with_features)

# Diviser les données
train_data, val_data, test_data = time_series_split(normalized_data, train_ratio=0.7, val_ratio=0.15)
```

## Dépendances

Ce module dépend principalement des bibliothèques suivantes:
- pandas pour la manipulation des données
- numpy pour les calculs numériques
- ta (Technical Analysis) pour les indicateurs techniques
- scikit-learn pour les transformateurs (scalers)
