# Formats de Données ADAN

Ce document décrit en détail les formats de données utilisés dans le système ADAN.

## 1. Structure des Données de Base

### 1.1 Format des Données de Marché
Les données de marché sont structurées selon les timeframes suivants :

- **5 minutes (5m)**
- **1 heure (1h)**
- **4 heures (4h)**

Pour chaque timeframe, les colonnes requises sont :

| Colonne | Description | Type | Contraintes |
|---------|-------------|------|-------------|
| `OPEN` | Prix d'ouverture | float64 | > 0 |
| `HIGH` | Prix le plus haut | float64 | > 0 |
| `LOW` | Prix le plus bas | float64 | > 0 |
| `CLOSE` | Prix de clôture | float64 | > 0 |
| `VOLUME` | Volume échangé | float64 | ≥ 0 |

### 1.2 Features Techniques par Timeframe

#### 5 minutes (5m)
- Indicateurs de momentum : RSI_14, STOCHk_14_3_3, STOCHd_14_3_3
- Indicateurs de tendance : EMA_5, EMA_20
- Indicateurs de volatilité : CCI_20_0.015, ROC_9, MFI_14
- Indicateurs de support/résistance : SUPERTREND_14_2.0, PSAR_0.02_0.2

#### 1 heure (1h)
- Indicateurs de momentum : RSI_14
- Indicateurs de tendance : EMA_50, EMA_100, SMA_200
- Indicateurs de volatilité : CCI_20_0.015, MFI_14
- Indicateurs de support/résistance : ICHIMOKU_9_26_52, PSAR_0.02_0.2

#### 4 heures (4h)
- Indicateurs de momentum : RSI_14
- Indicateurs de tendance : EMA_50, SMA_200
- Indicateurs de volatilité : CCI_20_0.015, MFI_14
- Indicateurs de support/résistance : ICHIMOKU_9_26_52, SUPERTREND_14_3.0, PSAR_0.02_0.2

## 2. Structure des Observations

### 2.1 Format 3D des Observations
Les observations sont structurées en tenseurs 3D de forme (n_timeframes, window_size, n_features) :

- **n_timeframes**: 3 (5m, 1h, 4h)
- **window_size**: 20 (configurable)
- **n_features**: 15 (nombre maximum de features par timeframe)

### 2.2 Normalisation des Données
- Normalisation différente par timeframe :
  - 5m: MinMaxScaler (-1, 1)
  - 1h: StandardScaler
  - 4h: RobustScaler

## 3. Portfolio State

### 3.1 Structure du Portfolio
- **Balance**: float64 (≥ 0)
- **Positions**: dict {asset: float64}
- **Frais de trading**: 0.01 (1%)
- **Taille maximale de position**: 0.1 (10%)
- **Stop-loss**: 0.02 (2%)
- **Take-profit**: 0.05 (5%)

### 3.2 Contraintes de Risque
- **Drawdown maximum**: 0.1 (10%)
- **Position size maximum**: 0.1 (10%)

## 4. Format des Données d'Entraînement

### 4.1 Split des Données
- **Training**: 2023-01-01 à 2023-01-02
- **Validation**: 2023-01-02 à 2023-01-03
- **Test**: 2023-01-03 à 2023-01-04

### 4.2 Configuration du DataLoader
- **Batch size**: 32
- **Num workers**: 1
- **Chunk size**: 100
- **Max chunks in memory**: 1

## 5. Optimisation de la Mémoire

### 5.1 Paramètres de Performance
- **Taille de fenêtre**: 20
- **Taille de chunk**: 500
- **Nombre de threads**: 4
- **Seuil d'avertissement mémoire**: 5600 MB
- **Seuil critique mémoire**: 6300 MB

## 6. Validation des Données

### 6.1 Règles de Validation
1. **Types de données**
   - Toutes les colonnes numériques doivent être float64
   - Index doit être datetime

2. **Valeurs manquantes**
   - NaN doivent être remplacés par interpolation
   - Valeurs infinies doivent être remplacées

3. **Ordre temporel**
   - Données doivent être triées par ordre chronologique
   - Intervalle temporel constant entre les observations

4. **Valeurs extrêmes**
   - Prix : > 0
   - Volume : ≥ 0
   - Features : entre -1 et 1 après normalisation

## 7. Formats de Fichiers

### 7.1 Données Brutes
- Format: Parquet
- Localisation: `${paths.raw_data_dir}`
- Compression: Snappy

### 7.2 Données Traitées
- Format: Parquet
- Localisation: `${paths.processed_data_dir}`
- Compression: Snappy

### 7.3 Modèles Entraînés
- Format: Joblib
- Localisation: `${paths.trained_models_dir}`

## 8. Configuration des Assets

### 8.1 Assets Supportés
- BTC (Bitcoin)
- ETH (Ethereum)
- SOL (Solana)
- XRP (Ripple)
- ADA (Cardano)

### 8.2 Initialisation du Portfolio
- Solde initial: 100.0
- Frais de trading: 0.01 (1%)
- Maximum d'étapes: 100
