# 🚨 ALERTE CRITIQUE - DIVERGENCE FEATURES ENTRAÎNEMENT/INFÉRENCE

**Date**: 2 Janvier 2026  
**Gravité**: 🔴 **CRITIQUE - NE PAS DÉPLOYER**

---

## ⚠️ DÉCOUVERTE MAJEURE

**Les features utilisées en ENTRAÎNEMENT sont COMPLÈTEMENT DIFFÉRENTES de celles configurées en LIVE !**

---

## 📊 COMPARAISON DÉTAILLÉE

### **5m Timeframe**

#### Configuration LIVE (StateBuilder)
```
[0] OPEN
[1] HIGH
[2] LOW
[3] CLOSE
[4] VOLUME
[5] RSI_14
[6] STOCHk_14_3_3
[7] STOCHd_14_3_3
[8] CCI_20_0.015
[9] ROC_9
[10] MFI_14
[11] EMA_5
[12] EMA_20
[13] SUPERTREND_14_2.0
[14] PSAR_0.02_0.2
```

#### Données ENTRAÎNEMENT (5m.parquet)
```
[0] open
[1] high
[2] low
[3] close
[4] volume
[5] rsi_14
[6] macd_12_26_9          ← DIFFÉRENT (LIVE: STOCHk_14_3_3)
[7] bb_percent_b_20_2     ← DIFFÉRENT (LIVE: STOCHd_14_3_3)
[8] atr_14                ← DIFFÉRENT (LIVE: CCI_20_0.015)
[9] atr_20                ← DIFFÉRENT (LIVE: ROC_9)
[10] atr_50               ← DIFFÉRENT (LIVE: MFI_14)
[11] volume_ratio_20      ← DIFFÉRENT (LIVE: EMA_5)
[12] ema_20_ratio         ← DIFFÉRENT (LIVE: EMA_20)
[13] stoch_k_14_3_3       ← DIFFÉRENT (LIVE: SUPERTREND_14_2.0)
[14] price_action         ← DIFFÉRENT (LIVE: PSAR_0.02_0.2)
```

**DIVERGENCE**: 9/15 features différentes (60% de divergence!)

---

### **1h Timeframe**

#### Configuration LIVE (StateBuilder)
```
[0] OPEN
[1] HIGH
[2] LOW
[3] CLOSE
[4] VOLUME
[5] RSI_14
[6] MACD_12_26_9
[7] MACD_HIST_12_26_9
[8] CCI_20_0.015
[9] MFI_14
[10] EMA_50
[11] EMA_100
[12] SMA_200
[13] ICHIMOKU_9_26_52
[14] PSAR_0.02_0.2
```

#### Données ENTRAÎNEMENT (1h.parquet)
```
[0] open
[1] high
[2] low
[3] close
[4] volume
[5] rsi_21               ← DIFFÉRENT (LIVE: RSI_14)
[6] macd_21_42_9         ← DIFFÉRENT (LIVE: MACD_12_26_9)
[7] bb_width_20_2        ← DIFFÉRENT (LIVE: MACD_HIST_12_26_9)
[8] adx_14               ← DIFFÉRENT (LIVE: CCI_20_0.015)
[9] atr_20               ← DIFFÉRENT (LIVE: MFI_14)
[10] atr_50              ← DIFFÉRENT (LIVE: EMA_50)
[11] obv_ratio_20        ← DIFFÉRENT (LIVE: EMA_100)
[12] ema_50_ratio        ← DIFFÉRENT (LIVE: SMA_200)
[13] ichimoku_base       ← SIMILAIRE (LIVE: ICHIMOKU_9_26_52)
[14] fib_ratio           ← DIFFÉRENT (LIVE: PSAR_0.02_0.2)
[15] price_ema_ratio_50  ← EXTRA (LIVE: 15 features)
```

**DIVERGENCE**: 13/15 features différentes (87% de divergence!)

---

### **4h Timeframe**

#### Configuration LIVE (StateBuilder)
```
[0] OPEN
[1] HIGH
[2] LOW
[3] CLOSE
[4] VOLUME
[5] RSI_14
[6] MACD_12_26_9
[7] CCI_20_0.015
[8] MFI_14
[9] EMA_50
[10] SMA_200
[11] ICHIMOKU_9_26_52
[12] SUPERTREND_14_3.0
[13] PSAR_0.02_0.2
```

#### Données ENTRAÎNEMENT (4h.parquet)
```
[0] open
[1] high
[2] low
[3] close
[4] volume
[5] rsi_28               ← DIFFÉRENT (LIVE: RSI_14)
[6] macd_26_52_18        ← DIFFÉRENT (LIVE: MACD_12_26_9)
[7] supertrend_10_3      ← DIFFÉRENT (LIVE: CCI_20_0.015)
[8] atr_20               ← DIFFÉRENT (LIVE: MFI_14)
[9] atr_50               ← DIFFÉRENT (LIVE: EMA_50)
[10] volume_sma_20_ratio ← DIFFÉRENT (LIVE: SMA_200)
[11] ema_100_ratio       ← DIFFÉRENT (LIVE: ICHIMOKU_9_26_52)
[12] pivot_level         ← DIFFÉRENT (LIVE: SUPERTREND_14_3.0)
[13] donchian_width_20   ← DIFFÉRENT (LIVE: PSAR_0.02_0.2)
[14] market_structure    ← EXTRA
[15] volatility_ratio_14_50 ← EXTRA
```

**DIVERGENCE**: 14/14 features différentes (100% de divergence!)

---

## 🔴 IMPACT CRITIQUE

### **Scénario de Déploiement Actuel**

```
Modèles entraînés avec:
  5m: [open, high, low, close, volume, rsi_14, macd_12_26_9, bb_percent_b, atr_14, ...]
  1h: [open, high, low, close, volume, rsi_21, macd_21_42_9, bb_width, adx_14, ...]
  4h: [open, high, low, close, volume, rsi_28, macd_26_52_18, supertrend_10_3, ...]

Modèles reçoivent en LIVE:
  5m: [open, high, low, close, volume, rsi_14, stoch_k_14_3_3, stoch_d_14_3_3, cci_20, ...]
  1h: [open, high, low, close, volume, rsi_14, macd_12_26_9, macd_hist, cci_20, ...]
  4h: [open, high, low, close, volume, rsi_14, macd_12_26_9, cci_20, ...]

RÉSULTAT:
  ❌ Les modèles reçoivent des features COMPLÈTEMENT DIFFÉRENTES
  ❌ Les réseaux de neurones sont COMPLÈTEMENT DÉSORIENTÉS
  ❌ Les décisions sont ALÉATOIRES ou SYSTÉMATIQUEMENT MAUVAISES
  ❌ CATASTROPHE GARANTIE EN PRODUCTION
```

---

## 🚨 CAUSES PROBABLES

1. **Divergence de code**: StateBuilder a été modifié mais pas les données d'entraînement
2. **Versions différentes**: Les données d'entraînement utilisent une ancienne version
3. **Configuration oubliée**: La configuration d'entraînement n'a pas été mise à jour
4. **Deux pipelines différents**: Entraînement et inférence utilisent des pipelines différents

---

## 🎯 ACTIONS REQUISES IMMÉDIATEMENT

### **Option 1: Recréer les données d'entraînement** (Recommandé)
```bash
# Régénérer les données d'entraînement avec la configuration LIVE
python3 scripts/generate_training_data.py --config live
```

### **Option 2: Modifier la configuration LIVE** (Rapide)
```bash
# Modifier StateBuilder pour utiliser les features d'entraînement
# Éditer: src/adan_trading_bot/data_processing/state_builder.py
# Remplacer features_config par celle d'entraînement
```

### **Option 3: Réentraîner les modèles** (Complet)
```bash
# Réentraîner avec la configuration LIVE
python3 scripts/train_models.py --config live
```

---

## 📋 CHECKLIST DE CORRECTION

- [ ] **Décider**: Quelle configuration utiliser? (Entraînement ou Live?)
- [ ] **Aligner**: Mettre à jour StateBuilder OU les données d'entraînement
- [ ] **Valider**: Vérifier que les features sont IDENTIQUES
- [ ] **Tester**: Tester avec un vecteur de test
- [ ] **Réentraîner**: Si nécessaire, réentraîner les modèles
- [ ] **Déployer**: Seulement après validation complète

---

## ⚠️ RECOMMANDATION FINALE

**NE PAS DÉPLOYER AVANT CORRECTION COMPLÈTE**

Cette divergence est **CATASTROPHIQUE** et garantira des décisions aléatoires ou systématiquement mauvaises en production.

**Priorité**: CRITIQUE - À traiter IMMÉDIATEMENT avant tout déploiement.

---

**Status**: 🔴 **BLOQUANT - AUDIT ÉCHOUÉ**
