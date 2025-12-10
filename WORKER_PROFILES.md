# 📊 Profils des Workers - Documentation

Ce document décrit les profils des 4 workers actuellement configurés dans le système de trading.

## 🔵 W1 - Ultra-Stable (Trial 26)

### Caractéristiques Principales
- **Type** : Ultra-Conservateur
- **Objectif** : Préservation du capital, faible risque
- **Description** : Configuration optimisée pour la stabilité et la protection du capital

### Paramètres Clés
- **Batch Size** : 128
- **Clip Range** : 0.1616
- **Entropy Coefficient** : 0.0175
- **GAE Lambda** : 0.9309
- **Max Gradient Norm** : 0.8
- **N_epochs** : 10
- **N_steps** : 1024
- **Vf_coef** : 0.5

### Gestion du Risque
- **Risque par Trade** : 1.99% (Micro) à 5.48% (Enterprise)
- **Stop Loss** : 7.76% (par défaut)
- **Take Profit** : 10.56% (par défaut)
- **Position Max** : 1
- **Période de Détention Min** : 5 steps

---

## 🟢 W2 - Moderate Optimized

### Caractéristiques Principales
- **Type** : Équilibré
- **Objectif** : Équilibre entre risque et rendement
- **Description** : Configuration équilibrée pour des performances stables

### Paramètres Clés
- **Batch Size** : 128
- **Clip Range** : 0.3419
- **Entropy Coefficient** : 0.0195
- **GAE Lambda** : 0.9309
- **Max Gradient Norm** : 0.8
- **N_epochs** : 10
- **N_steps** : 1024
- **Vf_coef** : 0.5

### Gestion du Risque
- **Risque par Trade** : 1.8% (Micro) à 0.6% (Enterprise)
- **Stop Loss** : 3.5% (par défaut)
- **Take Profit** : 6.0% (par défaut)
- **Position Max** : 2
- **Période de Détention Min** : 20 steps

---

## 🟠 W3 - Aggressive Optimized

### Caractéristiques Principales
- **Type** : Agressif
- **Objectif** : Maximisation des rendements
- **Description** : Configuration agressive pour des rendements potentiellement plus élevés

### Paramètres Clés
- **Batch Size** : 128
- **Clip Range** : 0.1616
- **Entropy Coefficient** : 0.0175
- **GAE Lambda** : 0.9309
- **Max Gradient Norm** : 0.8
- **N_epochs** : 10
- **N_steps** : 1024
- **Vf_coef** : 0.5

### Gestion du Risque
- **Risque par Trade** : 3.5% (Micro) à 1.0% (Enterprise)
- **Stop Loss** : 7.44% (par défaut)
- **Take Profit** : 11.43% (par défaut)
- **Position Max** : 1
- **Période de Détention Min** : 140 steps

---

## 🔴 W4 - Sharpe Optimized

### Caractéristiques Principales
- **Type** : Optimisé pour le ratio de Sharpe
- **Objectif** : Meilleur ratio rendement/risque
- **Description** : Configuration optimisée pour maximiser le ratio de Sharpe

### Paramètres Clés
- **Batch Size** : 128
- **Clip Range** : 0.2
- **Entropy Coefficient** : 0.01
- **GAE Lambda** : 0.9
- **Max Gradient Norm** : 0.5
- **N_epochs** : 10
- **N_steps** : 2048
- **Vf_coef** : 0.5

### Gestion du Risque
- **Risque par Trade** : 2.5% (Micro) à 0.5% (Enterprise)
- **Stop Loss** : 2.09% (très serré)
- **Take Profit** : 3.94% (très serré)
- **Position Max** : 6
- **Période de Détention Min** : 3 steps

---

## 🔄 Comparaison des Profils

| Paramètre | W1 (Ultra-Stable) | W2 (Moderate) | W3 (Aggressive) | W4 (Sharpe) |
|-----------|-------------------|---------------|-----------------|-------------|
| **Type** | Ultra-Conservateur | Équilibré | Agressif | Optimisé Sharpe |
| **Risque/Trade** | 1.99-5.48% | 1.8-0.6% | 3.5-1.0% | 2.5-0.5% |
| **Stop Loss** | 7.76% | 3.5% | 7.44% | 2.09% |
| **Take Profit** | 10.56% | 6.0% | 11.43% | 3.94% |
| **Positions Max** | 1 | 2 | 1 | 6 |
| **Détention Min** | 5 steps | 20 steps | 140 steps | 3 steps |

## 📝 Notes d'Utilisation

1. **W1** : Recommandé pour les marchés très volatils ou pour les investisseurs averses au risque.
2. **W2** : Configuration par défaut pour un équilibre optimal entre risque et rendement.
3. **W3** : À utiliser avec précaution, uniquement pour les marchés avec tendance claire.
4. **W4** : Spécialisé pour les stratégies à court terme avec des objectifs de profit rapides.

## 🔄 Mise à Jour

Dernière mise à jour : 2025-12-09

*Note : Cette documentation reflète la configuration actuelle du fichier `config.yaml`.*
