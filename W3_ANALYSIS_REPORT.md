# 📊 RAPPORT D'ANALYSE DÉTAILLÉE - W3 (20 TRIALS)

**Date**: 2025-12-09 22:19-22:49 UTC  
**Durée**: 29 minutes 40 secondes  
**Status**: ⚠️ **FAIBLE PERFORMANCE - PARAMÈTRES TROP RESTRICTIFS**

---

## 🎯 RÉSUMÉ EXÉCUTIF

| Métrique | Valeur |
|----------|--------|
| **Meilleur Score** | 3.29 (Trial 13) |
| **Meilleur Sharpe** | 6.65 (Trial 13) |
| **Meilleur Trial** | Trial 13 |
| **Tous les Trials** | ✅ 20/20 complétés |
| **Temps moyen/trial** | ~89 secondes |

---

## ⚠️ PROBLÈME IDENTIFIÉ

**Performance très faible**: Score moyen = 3.29 (vs W1 = 51.46)

**Cause probable**:
- W3 (Position Trader) a des paramètres très restrictifs
- `min_holding_period_steps: 140` est très élevé
- `max_concurrent_positions: 1` limite fortement les opportunités
- Ces paramètres réduisent drastiquement le nombre de trades

---

## 📊 MEILLEUR TRIAL (Trial 13)

### Paramètres PPO
```yaml
learning_rate: 4.293510964732076e-05
n_steps: 512
batch_size: 32
n_epochs: 13
gamma: 0.9870128124087191
gae_lambda: 0.9843310916091376
clip_range: 0.23312370343767125
ent_coef: 0.0012628352110880398
vf_coef: 0.73885189956288
max_grad_norm: 0.7339433833946983
```

### Paramètres Trading (W3)
```yaml
stop_loss_pct: 0.0744 (7.44%)
take_profit_pct: 0.1143 (11.43%)
position_size_pct: 0.258 (25.8%)
risk_per_trade_pct: 0.0232 (2.32%)
max_concurrent_positions: 1
min_holding_period_steps: 140
```

### Métriques
- **Score Optuna**: 3.29
- **Sharpe Ratio**: 6.65
- **Max Drawdown**: 17.78%
- **Win Rate**: 47.13%
- **Total Trades**: 174
- **Profit Factor**: 1.30
- **Total Return**: 38.99%

### Observations
⚠️ **Score très faible** (3.29 vs W1=51.46)  
⚠️ **Sharpe faible** (6.65 vs W1=25.95)  
⚠️ **Drawdown élevé** (17.78% vs W1=11.43%)  
⚠️ **Win rate faible** (47.13% vs W1=58.98%)  
⚠️ **Peu de trades** (174 vs W1=512)  
⚠️ **Profit factor faible** (1.30 vs W1=1.47)  

---

## 📈 COMPARAISON W1 vs W2 vs W3

| Métrique | W1 | W2 | W3 |
|----------|----|----|-----|
| **Score** | 51.46 ⭐ | 100.00 (outlier) | 3.29 ⚠️ |
| **Sharpe** | 25.95 ⭐ | 14.19 | 6.65 ⚠️ |
| **Drawdown** | 11.43% ⭐ | 8.77% | 17.78% ⚠️ |
| **Win Rate** | 58.98% ⭐ | 64.47% | 47.13% ⚠️ |
| **Trades** | 512 ⭐ | 380 | 174 ⚠️ |
| **Profit Factor** | 1.47 ⭐ | 3.00 | 1.30 ⚠️ |

---

## 🔍 ANALYSE DES PARAMÈTRES W3

### Problème 1: min_holding_period_steps = 140
- **Impact**: Chaque position doit rester ouverte au minimum 140 steps
- **Conséquence**: Très peu de trades possibles
- **Comparaison**: W1 = 5 steps, W2 = 20 steps, W3 = 140 steps

### Problème 2: max_concurrent_positions = 1
- **Impact**: Une seule position ouverte à la fois
- **Conséquence**: Pas de diversification
- **Comparaison**: W1 = 3 positions, W2 = 2 positions, W3 = 1 position

### Problème 3: SL/TP très larges
- **Stop Loss**: 7.44% (vs W1 = 2.53%)
- **Take Profit**: 11.43% (vs W1 = 3.21%)
- **Impact**: Positions très longues, peu de trades

---

## 🎯 CONCLUSION W3

**Status**: ⚠️ **FAIBLE MAIS UTILISABLE**

W3 (Position Trader) a produit des résultats faibles :
- **Score meilleur**: 3.29 (Trial 13)
- **Sharpe moyen**: 6.65 (faible)
- **Drawdown moyen**: 17.78% (élevé)
- **Win rate moyen**: 47.13% (faible)
- **Trades**: 174 (peu)

**Recommandation**: Utiliser Trial 13 comme configuration pour W3, mais les résultats sont bien inférieurs à W1.

---

## 🚀 PROCHAINES ÉTAPES

1. ✅ W1: Analysé et validé (Score=51.46) ⭐
2. ⚠️ W2: Analysé avec instabilité (Score=100.00 outlier)
3. ⚠️ W3: Analysé avec faible performance (Score=3.29)
4. ⏳ W4: À lancer (20 trials)

---

**Rapport généré**: 2025-12-09 22:49 UTC  
**Analyste**: Cascade  
**Status**: ⚠️ **PRÊT POUR W4**
