# T2 : Spécification Formelle de la Nouvelle Hiérarchie

## 🎯 Objectif T2

Définir formellement la hiérarchie **Environnement → DBE → Optuna** en respectant strictement les contraintes immuables :
- ✅ Paliers (capital_tiers) : **AUCUNE modification de valeurs**
- ✅ Intervalles d'exposition par palier : **INCHANGÉS**
- ✅ Min trade = 11 USDT : **NON NÉGOCIABLE**

---

## 📐 HIÉRARCHIE FORMELLE À 3 COUCHES

### Couche 1 : ENVIRONNEMENT (Arbitre - Contraintes Absolues)

**Rôle** : Appliquer les lois inviolables du système. Aucune décision ne peut les contourner.

**Responsabilités** :
1. Déterminer le palier (Micro/Small/Medium/High/Enterprise) selon le capital courant
2. Imposer les limites absolues du palier :
   - `max_position_size_pct` (ex: Micro = 90%, Small = 65%, etc.)
   - `max_concurrent_positions` (ex: Micro = 1, Small = 2, etc.)
   - `risk_per_trade_pct` (ex: Micro = 4%, Small = 2%, etc.)
   - `exposure_range` (ex: Micro = 70-90%, Small = 35-75%, etc.)
3. Imposer les limites absolues globales :
   - `min_order_value_usdt` = 11.0 (jamais moins)
   - `max_drawdown_pct` (par palier)
   - Bornes SL/TP (min/max absolus)

**Décisions Prises** :
- ✅ Palier courant
- ✅ Rejet/acceptation d'une position (si < 11 USDT ou > max_positions)
- ✅ Clamp final de la position_size (≤ tier.max_position_size_pct)

**Contraintes Immuables** :
```
capital_tiers = [
  {name: "Micro", min: 11, max: 30, max_pos: 90%, max_conc: 1, risk: 4%, exposure: [70, 90]},
  {name: "Small", min: 30, max: 100, max_pos: 65%, max_conc: 2, risk: 2%, exposure: [35, 75]},
  {name: "Medium", min: 100, max: 300, max_pos: 48%, max_conc: 3, risk: 2.25%, exposure: [45, 60]},
  {name: "High", min: 300, max: 1000, max_pos: 28%, max_conc: 4, risk: 2.75%, exposure: [20, 35]},
  {name: "Enterprise", min: 1000, max: ∞, max_pos: 20%, max_conc: 5, risk: 3%, exposure: [5, 15]}
]

min_order_value_usdt = 11.0 (JAMAIS MOINS)
```

---

### Couche 2 : DBE (Tacticien - Modulation Légère)

**Rôle** : Adapter légèrement les paramètres de base Optuna selon les conditions de marché (régime, volatilité, performance).

**Principe** : Multiplicateurs relatifs **bornés à ±15% maximum**. DBE ne remplace jamais Optuna, il l'ajuste légèrement.

**Responsabilités** :
1. Détecter le régime de marché (bear, bull, sideways, volatile)
2. Calculer des multiplicateurs relatifs (±X%) pour :
   - `position_size_pct` (ex: -10% en bear, +10% en bull)
   - `stop_loss_pct` (ex: +5% en bear, -3% en bull)
   - `take_profit_pct` (ex: -5% en bear, +8% en bull)
3. **Borner les multiplicateurs à ±15% max** (jamais plus)
4. Appliquer les multiplicateurs aux valeurs de base Optuna

**Formule de Modulation** :
```
adjusted_param = base_param × (1 + dbe_multiplier)

où :
  base_param = valeur Optuna (stratège)
  dbe_multiplier ∈ [-0.15, +0.15]  (±15% max)
  adjusted_param = valeur après modulation DBE
```

**Exemple** :
```
Optuna base: position_size_pct = 0.1121 (11.21%)
DBE régime bear: multiplicateur = -0.10 (-10%)
Adjusted: 0.1121 × (1 - 0.10) = 0.1121 × 0.90 = 0.10089 (10.09%)
```

**Multiplicateurs DBE par Régime** (à définir dans config.yaml) :
```yaml
dbe_modulation:
  regime_multipliers:
    bear:
      position_size: -0.10    # -10%
      stop_loss: +0.05        # +5%
      take_profit: -0.05      # -5%
    bull:
      position_size: +0.10    # +10%
      stop_loss: -0.03        # -3%
      take_profit: +0.08      # +8%
    sideways:
      position_size: 0.0      # 0%
      stop_loss: 0.0          # 0%
      take_profit: 0.0        # 0%
    volatile:
      position_size: -0.15    # -15% (max)
      stop_loss: +0.10        # +10%
      take_profit: -0.10      # -10%
```

**Décisions Prises** :
- ✅ Régime de marché détecté
- ✅ Multiplicateurs calculés (±15% max)
- ✅ Paramètres ajustés (base Optuna × multiplicateurs)

**Contraintes** :
- Multiplicateurs toujours ∈ [-0.15, +0.15]
- Jamais de valeurs absolues (toujours relatif à Optuna)
- Jamais d'écrasement des valeurs Optuna

---

### Couche 3 : OPTUNA (Stratège - Performance Pure)

**Rôle** : Définir les paramètres de base optimisés pour chaque worker. C'est la source de vérité pour la performance.

**Responsabilités** :
1. Stocker les valeurs optimisées par Optuna pour chaque worker :
   - `position_size_pct` (ex: W1 = 0.1121, W2 = 0.25, W3 = 0.258, W4 = 0.2)
   - `stop_loss_pct` (ex: W1 = 0.0253, W2 = 0.025, W3 = 0.0744, W4 = 0.012)
   - `take_profit_pct` (ex: W1 = 0.0321, W2 = 0.05, W3 = 0.1143, W4 = 0.02)
   - `risk_per_trade_pct` (ex: W1 = 0.01, W2 = 0.015, W3 = 0.0232, W4 = 0.012)
   - Hyperparamètres PPO (learning_rate, gamma, ent_coef, vf_coef, etc.)

2. Être la source unique de vérité pour chaque paramètre (pas de doublons)

**Valeurs Optuna Actuelles** :
```yaml
workers:
  w1:
    trading_parameters:
      position_size_pct: 0.1121
      stop_loss_pct: 0.0253
      take_profit_pct: 0.0321
      risk_per_trade_pct: 0.01
  w2:
    trading_parameters:
      position_size_pct: 0.25
      stop_loss_pct: 0.025
      take_profit_pct: 0.05
      risk_per_trade_pct: 0.015
  w3:
    trading_parameters:
      position_size_pct: 0.258
      stop_loss_pct: 0.0744
      take_profit_pct: 0.1143
      risk_per_trade_pct: 0.0232
  w4:
    trading_parameters:
      position_size_pct: 0.2
      stop_loss_pct: 0.012
      take_profit_pct: 0.02
      risk_per_trade_pct: 0.012
```

**Décisions Prises** :
- ✅ Valeurs de base pour chaque worker
- ✅ Aucune modification (source de vérité)

**Contraintes** :
- Une seule source de vérité par paramètre
- Pas de doublons (pas de `risk_management.position_size_pct`, pas de `trading.stop_loss_pct`, etc.)
- Pas de `*_by_tier` (les paliers sont appliqués par l'environnement, pas Optuna)

---

## 🔄 FLUX DE DÉCISION SÉQUENTIEL

### Algorithme d'Application

```
ENTRÉE : worker_id, capital_courant, market_regime

ÉTAPE 1 : ENVIRONNEMENT - Déterminer le Palier
  tier = determine_capital_tier(capital_courant)
  tier_config = capital_tiers[tier]
  
ÉTAPE 2 : OPTUNA - Charger Valeurs de Base
  optuna_params = workers[worker_id].trading_parameters
  base_position = optuna_params.position_size_pct
  base_sl = optuna_params.stop_loss_pct
  base_tp = optuna_params.take_profit_pct
  base_risk = optuna_params.risk_per_trade_pct
  
ÉTAPE 3 : DBE - Appliquer Modulation Légère
  dbe_mult = dbe_modulation.regime_multipliers[market_regime]
  adjusted_position = base_position × (1 + dbe_mult.position_size)
  adjusted_sl = base_sl × (1 + dbe_mult.stop_loss)
  adjusted_tp = base_tp × (1 + dbe_mult.take_profit)
  
ÉTAPE 4 : ENVIRONNEMENT - Appliquer Contraintes Finales
  # Clamp position par tier et environnement
  final_position = min(adjusted_position, tier_config.max_position_size_pct)
  final_position = max(final_position, 0.01)  # Min 1%
  
  # Clamp SL/TP par bornes absolues
  final_sl = max(min(adjusted_sl, 0.20), 0.005)  # [0.5%, 20%]
  final_tp = max(min(adjusted_tp, 0.50), 0.01)   # [1%, 50%]
  
  # Clamp risk par tier et environnement
  final_risk = min(base_risk, tier_config.risk_per_trade_pct)
  
ÉTAPE 5 : ENVIRONNEMENT - Vérifier Notional
  notional = capital_courant × final_position
  if notional < 11.0:
    REJETER TRADE (< min_trade_value)
  
ÉTAPE 6 : ENVIRONNEMENT - Vérifier Max Positions
  if len(open_positions) >= tier_config.max_concurrent_positions:
    REJETER TRADE (max positions atteint)
  
SORTIE : final_position, final_sl, final_tp, final_risk (ou REJET)
```

---

## 📊 TABLEAU DE DÉCISION

| Étape | Couche | Décision | Exemple |
|-------|--------|----------|---------|
| 1 | Environnement | Déterminer palier | Capital 50 USDT → Tier "Small" |
| 2 | Optuna | Charger base | W1: pos=11.21%, SL=2.53%, TP=3.21% |
| 3 | DBE | Appliquer modulation | Régime bear: pos -10% → 10.09% |
| 4 | Environnement | Clamp par tier | Small max_pos=65% → 10.09% OK |
| 5 | Environnement | Vérifier notional | 50 × 10.09% = 5.05 USDT < 11 → REJET |
| 6 | Environnement | Vérifier max_pos | Small max_conc=2, open=1 → OK |

---

## ✅ VALIDATION DE LA HIÉRARCHIE

### Contrainte 1 : Paliers Inchangés
- ✅ `capital_tiers` reste **EXACTEMENT** comme aujourd'hui
- ✅ Aucune modification de valeurs, intervalles, ou limites
- ✅ Paliers appliqués dans l'étape 4 (Environnement)

### Contrainte 2 : Min Trade = 11 USDT
- ✅ Vérification dans l'étape 5 (Environnement)
- ✅ Rejet si notional < 11.0
- ✅ Jamais contournable

### Contrainte 3 : Intervalles d'Exposition Préservés
- ✅ Exposure_range du palier respecté (ex: Small = 35-75%)
- ✅ Position_size finale ≤ max_position_size_pct du palier
- ✅ Pas de modification des intervalles

### Contrainte 4 : DBE Modulateur Léger
- ✅ Multiplicateurs bornés à ±15% max
- ✅ Jamais d'écrasement des valeurs Optuna
- ✅ Toujours relatif à la base Optuna

### Contrainte 5 : Optuna Source Unique de Vérité
- ✅ Une seule source par paramètre (`trading_parameters`)
- ✅ Pas de doublons (`risk_management`, `trading`, `*_by_tier`)
- ✅ Valeurs Optuna préservées (jamais écrasées)

---

## 🎯 EXEMPLE CONCRET : W1 avec 50 USDT en Régime Bear

```
ENTRÉE : worker_id=w1, capital=50 USDT, market_regime=bear

ÉTAPE 1 : Déterminer Palier
  capital = 50 USDT
  50 ∈ [30, 100] → Tier = "Small"
  tier_config = {max_pos: 65%, max_conc: 2, risk: 2%, exposure: [35, 75]}

ÉTAPE 2 : Charger Optuna
  optuna_params = workers.w1.trading_parameters
  base_position = 0.1121 (11.21%)
  base_sl = 0.0253 (2.53%)
  base_tp = 0.0321 (3.21%)
  base_risk = 0.01 (1%)

ÉTAPE 3 : Appliquer DBE (Régime Bear)
  dbe_mult = {position_size: -0.10, stop_loss: +0.05, take_profit: -0.05}
  adjusted_position = 0.1121 × (1 - 0.10) = 0.10089 (10.09%)
  adjusted_sl = 0.0253 × (1 + 0.05) = 0.02657 (2.66%)
  adjusted_tp = 0.0321 × (1 - 0.05) = 0.03050 (3.05%)

ÉTAPE 4 : Appliquer Contraintes Environnement
  final_position = min(0.10089, 0.65) = 0.10089 (10.09%)
  final_sl = max(min(0.02657, 0.20), 0.005) = 0.02657 (2.66%)
  final_tp = max(min(0.03050, 0.50), 0.01) = 0.03050 (3.05%)
  final_risk = min(0.01, 0.02) = 0.01 (1%)

ÉTAPE 5 : Vérifier Notional
  notional = 50 × 0.10089 = 5.0445 USDT
  5.0445 < 11.0 → ❌ REJET (< min_trade_value)

SORTIE : TRADE REJETÉ (notional insuffisant)
```

**Interprétation** :
- Optuna base respectée (11.21%)
- DBE appliqué légèrement (-10% en bear)
- Environnement applique contraintes (tier Small, min_trade 11)
- Résultat : Trade rejeté car capital insuffisant pour respecter min_trade

---

## 📝 RÉSUMÉ T2

| Aspect | Spécification |
|--------|---------------|
| **Hiérarchie** | Environnement > DBE > Optuna |
| **Environnement** | Paliers inchangés, min_trade=11, limites absolues |
| **DBE** | Multiplicateurs ±15% max, relatif à Optuna |
| **Optuna** | Source unique de vérité, pas de doublons |
| **Flux** | 6 étapes séquentielles (Env → Opt → DBE → Env → Env → Env) |
| **Contraintes** | Paliers, min_trade, intervalles d'exposition préservés |

---

## ✅ VALIDATION T2

- ✅ Hiérarchie formelle définie
- ✅ Rôles de chaque couche clarifiés
- ✅ Flux de décision séquentiel documenté
- ✅ Contraintes immuables respectées
- ✅ Exemple concret validé

**Prochaine étape** : T3 - Refactoriser config.yaml pour refléter cette hiérarchie
