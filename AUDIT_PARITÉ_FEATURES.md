# 🔍 AUDIT DE PARITÉ FEATURES - ENTRAÎNEMENT vs INFÉRENCE

**Date**: 2 Janvier 2026  
**Statut**: ⚠️ **AUDIT CRITIQUE EN COURS**

---

## 📋 CONFIGURATION EXACTE DES FEATURES

### Trouvée dans `src/adan_trading_bot/data_processing/state_builder.py` (lignes 200-250)

#### **5m Timeframe (15 features)**
```python
[
    "OPEN",                    # [0]
    "HIGH",                    # [1]
    "LOW",                     # [2]
    "CLOSE",                   # [3]
    "VOLUME",                  # [4]
    "RSI_14",                  # [5]
    "STOCHk_14_3_3",          # [6]
    "STOCHd_14_3_3",          # [7]
    "CCI_20_0.015",           # [8]
    "ROC_9",                  # [9]
    "MFI_14",                 # [10]
    "EMA_5",                  # [11]
    "EMA_20",                 # [12]
    "SUPERTREND_14_2.0",      # [13]
    "PSAR_0.02_0.2",          # [14]
]
```

#### **1h Timeframe (15 features)**
```python
[
    "OPEN",                    # [0]
    "HIGH",                    # [1]
    "LOW",                     # [2]
    "CLOSE",                   # [3]
    "VOLUME",                  # [4]
    "RSI_14",                  # [5]
    "MACD_12_26_9",           # [6]
    "MACD_HIST_12_26_9",      # [7]
    "CCI_20_0.015",           # [8]
    "MFI_14",                 # [9]
    "EMA_50",                 # [10]
    "EMA_100",                # [11]
    "SMA_200",                # [12]
    "ICHIMOKU_9_26_52",       # [13]
    "PSAR_0.02_0.2",          # [14]
]
```

#### **4h Timeframe (14 features)**
```python
[
    "OPEN",                    # [0]
    "HIGH",                    # [1]
    "LOW",                     # [2]
    "CLOSE",                   # [3]
    "VOLUME",                  # [4]
    "RSI_14",                  # [5]
    "MACD_12_26_9",           # [6]
    "CCI_20_0.015",           # [7]
    "MFI_14",                 # [8]
    "EMA_50",                 # [9]
    "SMA_200",                # [10]
    "ICHIMOKU_9_26_52",       # [11]
    "SUPERTREND_14_3.0",      # [12]
    "PSAR_0.02_0.2",          # [13]
]
```

---

## 📊 STRUCTURE DE L'OBSERVATION

### Dimensions Totales
```
5m:  20 fenêtres × 15 features = 300
1h:  10 fenêtres × 15 features = 150
4h:   5 fenêtres × 14 features =  70
Portfolio:                        = 17
─────────────────────────────────────
TOTAL:                          = 537
```

⚠️ **DISCREPANCE DÉTECTÉE**: Configuration dit 542, calcul donne 537

**Hypothèse**: Portfolio = 22 features (pas 17)
- Recalcul: 300 + 150 + 70 + 22 = 542 ✅

---

## 🔧 POINTS CRITIQUES À VÉRIFIER

### 1️⃣ **Paramètres des Indicateurs**

| Indicateur | Paramètre | Entraînement | Live | Status |
|-----------|-----------|-------------|------|--------|
| RSI | length | 14 | ? | ❌ À vérifier |
| Stoch | length | 14, 3, 3 | ? | ❌ À vérifier |
| CCI | length | 20 | ? | ❌ À vérifier |
| ROC | length | 9 | ? | ❌ À vérifier |
| MFI | length | 14 | ? | ❌ À vérifier |
| EMA | periods | 5, 20, 50, 100 | ? | ❌ À vérifier |
| SMA | periods | 200 | ? | ❌ À vérifier |
| MACD | params | 12, 26, 9 | ? | ❌ À vérifier |
| SUPERTREND | params | 14, 2.0 (5m) / 14, 3.0 (4h) | ? | ❌ À vérifier |
| PSAR | params | 0.02, 0.2 | ? | ❌ À vérifier |
| ICHIMOKU | params | 9, 26, 52 | ? | ❌ À vérifier |

### 2️⃣ **Normalisation par Feature**

**Question critique**: Chaque feature est-elle normalisée INDIVIDUELLEMENT ou par SEGMENT?

- ✅ Si par feature: Chaque feature a sa propre mean/std
- ❌ Si par segment: Toutes les features 5m partagent une mean/std (MAUVAIS)

**Vérification**: Comparer `vecnorm.obs_rms.mean` avec les données d'entraînement

### 3️⃣ **Ordre des Features dans le Vecteur**

**Structure du vecteur observation**:
```
[5m_window_1_feature_1, 5m_window_1_feature_2, ..., 5m_window_1_feature_15,
 5m_window_2_feature_1, 5m_window_2_feature_2, ..., 5m_window_2_feature_15,
 ...
 5m_window_20_feature_1, ..., 5m_window_20_feature_15,
 1h_window_1_feature_1, ..., 1h_window_10_feature_15,
 4h_window_1_feature_1, ..., 4h_window_5_feature_14,
 portfolio_feature_1, ..., portfolio_feature_22]
```

**Vérification**: L'ordre est-il IDENTIQUE à l'entraînement?

### 4️⃣ **Portfolio State (17 ou 22 features?)**

**Question**: Quelles sont les 17 (ou 22) features du portfolio?

Hypothèses:
- Balance, position_size, entry_price, current_price, pnl, pnl_percent, drawdown, sharpe_ratio, win_rate, avg_trade, trades_count, consecutive_wins, consecutive_losses, max_drawdown, volatility, time_in_trade, trade_duration

**Vérification**: Comparer avec `observation['portfolio_state']` dans le code

---

## 🚨 CHECKLIST DE VALIDATION

- [ ] **Paramètres RSI**: Vérifier que RSI(14) est utilisé partout
- [ ] **Paramètres Stoch**: Vérifier que Stoch(14, 3, 3) est utilisé
- [ ] **Paramètres MACD**: Vérifier que MACD(12, 26, 9) est utilisé
- [ ] **Paramètres EMA**: Vérifier que EMA(5, 20, 50, 100) sont utilisés
- [ ] **Paramètres SMA**: Vérifier que SMA(200) est utilisé
- [ ] **Paramètres SUPERTREND**: Vérifier 14, 2.0 (5m) et 14, 3.0 (4h)
- [ ] **Paramètres PSAR**: Vérifier 0.02, 0.2
- [ ] **Paramètres ICHIMOKU**: Vérifier 9, 26, 52
- [ ] **Ordre des features**: Vérifier que l'ordre est IDENTIQUE
- [ ] **Normalisation**: Vérifier que chaque feature est normalisée correctement
- [ ] **Portfolio features**: Vérifier que 17 ou 22 features sont présentes
- [ ] **Dimensions totales**: Vérifier que 542 features sont présentes

---

## 📝 FICHIERS À VÉRIFIER

1. **IndicatorCalculator**: `src/adan_trading_bot/indicators/calculator.py`
   - Vérifier les paramètres de chaque indicateur

2. **StateBuilder**: `src/adan_trading_bot/data_processing/state_builder.py`
   - Vérifier la construction du vecteur observation

3. **ObservationBuilder**: `src/adan_trading_bot/observation/builder.py`
   - Vérifier la construction du portfolio_state

4. **Données d'entraînement**: `data/processed/indicators/train/BTCUSDT/`
   - Vérifier les features réelles utilisées

---

## 🎯 PROCHAINES ÉTAPES

1. ✅ Lire la configuration exacte des features (FAIT)
2. ⏭️ Vérifier les paramètres des indicateurs dans IndicatorCalculator
3. ⏭️ Vérifier la construction du vecteur observation dans StateBuilder
4. ⏭️ Comparer avec les données d'entraînement
5. ⏭️ Valider la normalisation par feature
6. ⏭️ Tester avec un vecteur de test

---

## ⚠️ RISQUES IDENTIFIÉS

| Risque | Probabilité | Impact |
|--------|------------|--------|
| Paramètres indicateurs différents | HAUTE | 🚨 CATASTROPHIQUE |
| Ordre des features différent | MOYENNE | 🚨 CATASTROPHIQUE |
| Normalisation différente | HAUTE | 🚨 CATASTROPHIQUE |
| Portfolio features manquantes | MOYENNE | 🚨 ÉLEVÉE |
| Dimensions totales différentes | BASSE | 🚨 ÉLEVÉE |

---

**Status**: ⚠️ **AUDIT EN COURS - NE PAS DÉPLOYER AVANT VALIDATION COMPLÈTE**
