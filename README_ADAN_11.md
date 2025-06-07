# ADAN 1.1 - Version Optimisée

## 📋 Résumé

ADAN 1.1 est la version optimisée et compressée du projet ADAN (Autonomous Deep AI Navigator) pour l'entraînement de modèles de trading par renforcement. Cette version a été spécialement préparée pour un déploiement efficace sur une machine d'entraînement distante.

## 🚀 Caractéristiques ADAN 1.1

### Données Traitées
- **Volume** : 2,865,600 lignes de données (~2.9M)
- **Cryptomonnaies** : 5 paires (BTCUSDT, ETHUSDT, BNBUSDT, ADAUSDT, XRPUSDT)
- **Timeframe** : 1 minute
- **Période** : 01/01/2024 à 01/02/2025
- **Features** : 49 indicateurs techniques par cryptomonnaie

### Optimisations Appliquées
- ✅ **Normalisation** : RobustScaler pour résistance aux outliers
- ✅ **Types optimisés** : float32 au lieu de float64 (-50% mémoire)
- ✅ **Valeurs aberrantes** : Clipping automatique (IQR method)
- ✅ **Valeurs manquantes** : Forward fill
- ✅ **Splits temporels** : 70% train / 15% validation / 15% test

### Structure des Données

```
data/
├── processed/                    # Données prêtes pour l'entraînement
│   ├── BTCUSDT_train.parquet    # Données d'entraînement par asset
│   ├── BTCUSDT_validation.parquet
│   ├── BTCUSDT_test.parquet
│   ├── combined_train.parquet   # Datasets combinés multi-assets
│   ├── combined_validation.parquet
│   └── combined_test.parquet
└── scalers_encoders/
    └── adan_11_scalers.joblib   # Scalers normalisés
```

## 📊 Statistiques des Données

| Metric | Valeur |
|--------|---------|
| **Total lignes** | 2,865,600 |
| **Mémoire totale** | ~820 MB (compressé) |
| **Features techniques** | 42 indicateurs |
| **Entraînement** | 2,005,920 lignes |
| **Validation** | 429,840 lignes |
| **Test** | 429,840 lignes |

## 🛠️ Installation et Utilisation

### Prérequis
- Python 3.9+
- Conda (recommandé)
- 8 GB RAM minimum
- GPU recommandé pour l'entraînement

### Installation
```bash
# Extraire l'archive ADAN 1.1
unzip ADAN_1.1_*.zip
cd ADAN

# Créer l'environnement conda
conda create -n trading_env python=3.9
conda activate trading_env

# Installer les dépendances
pip install -r requirements.txt
```

### Démarrage Rapide

```bash
# Activer l'environnement
conda activate trading_env

# Lancer l'entraînement
python src/adan_trading_bot/main.py --mode training --config config/main_config.yaml
```

## 🏗️ Architecture

### Features Disponibles
- **OHLCV** : open, high, low, close, volume
- **Moyennes mobiles** : SMA_short, SMA_long, EMA_short, EMA_long
- **Oscillateurs** : RSI, STOCH, Williams_%R, CMO
- **Tendance** : MACD, ADX, TRIX, DPO
- **Volatilité** : ATR, Bollinger Bands (BBU, BBM, BBL)
- **Volume** : OBV, CMF, MFI, VWMA, VWAP
- **Avancés** : Parabolic SAR, Ichimoku, PPO, Fisher Transform
- **Régimes HMM** : hmm_regime, hmm_prob_0, hmm_prob_1, hmm_prob_2

### Configuration ADAN 1.1
```yaml
version: '1.1'
data_info:
  timeframe: 1m
  normalization: RobustScaler
  splits:
    train: 70
    validation: 15
    test: 15
processing:
  workers: 3
  missing_values: forward_fill
  outliers: clip_iqr
  dtype_optimization: true
```

## ⚡ Optimisations Performances

### Parallélisation
- **Workers** : 3 processus par défaut (configurable)
- **Traitement parallèle** : Activé pour le preprocessing
- **Mémoire optimisée** : Types de données réduits

### Compression
- **Archive finale** : ~820 MB (vs ~1.2 GB original)
- **Format** : Parquet avec compression GZIP
- **Exclusions** : Cache, logs, fichiers temporaires supprimés

## 📈 Métriques de Qualité

### Validation des Données
- ✅ **Cohérence temporelle** : Intervalles de 1 minute vérifiés
- ✅ **Intégrité OHLCV** : Validation des barres de prix
- ✅ **Outliers** : Détection et traitement automatique
- ✅ **Valeurs manquantes** : 0 valeurs manquantes après traitement

### Splits Temporels
```
Train:      2024-01-01 → 2024-10-05  (70%)
Validation: 2024-10-05 → 2024-12-04  (15%)
Test:       2024-12-04 → 2025-02-01  (15%)
```

## 🔧 Scripts Utilitaires

### Vérification des Données
```bash
python scripts/verify_new_data.py --data-dir data/processed
```

### Re-préparation (si nécessaire)
```bash
./scripts/launch_adan_11_prep.sh 3 run
```

## 📝 Logs et Monitoring

- **Logs de préparation** : `adan_11_preparation.log`
- **Configuration** : `config/adan_11_config.yaml`
- **Scalers** : `data/scalers_encoders/adan_11_scalers.joblib`

## 🚨 Notes Importantes

1. **Environnement** : Utiliser `trading_env` via conda exclusivement
2. **Mémoire** : Minimum 8 GB RAM pour le training complet
3. **GPU** : Fortement recommandé pour l'entraînement RL
4. **Sauvegarde** : Les scalers sont essentiels pour la production

## 📞 Support

Cette version ADAN 1.1 est optimisée pour un déploiement rapide et efficace. Tous les fichiers non essentiels ont été déplacés vers `data/unused_for_later/` pour une utilisation future si nécessaire.

---

**Version** : 1.1  
**Date de création** : 2025-05-29  
**Taille optimisée** : 820 MB  
**Status** : Prêt pour l'entraînement 🚀