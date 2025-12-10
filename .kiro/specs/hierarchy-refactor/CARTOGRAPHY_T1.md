# T1 : Cartographie de la Hiérarchie Réelle Actuelle

## 📊 État de la Hiérarchie Actuelle

### 1. PALIERS (capital_tiers) - ENVIRONNEMENT

**Localisation** : `config/config.yaml` lignes 219-274

**Structure** :
```yaml
capital_tiers:
  - name: Micro Capital
    min_capital: 11.0
    max_capital: 30.0
    exposure_range: [70, 90]
    max_position_size_pct: 90
    max_concurrent_positions: 1
    risk_per_trade_pct: 4.0
    max_drawdown_pct: 4.0
    
  - name: Small Capital
    min_capital: 30.0
    max_capital: 100.0
    exposure_range: [35, 75]
    max_position_size_pct: 65
    max_concurrent_positions: 2
    risk_per_trade_pct: 2.0
    max_drawdown_pct: 3.75
    
  - name: Medium Capital
    min_capital: 100.0
    max_capital: 300.0
    exposure_range: [45, 60]
    max_position_size_pct: 48
    max_concurrent_positions: 3
    risk_per_trade_pct: 2.25
    max_drawdown_pct: 3.25
    
  - name: High Capital
    min_capital: 300.0
    max_capital: 1000.0
    exposure_range: [20, 35]
    max_position_size_pct: 28
    max_concurrent_positions: 4
    risk_per_trade_pct: 2.75
    max_drawdown_pct: 2.75
    
  - name: Enterprise
    min_capital: 1000.0
    max_capital: null
    exposure_range: [5, 15]
    max_position_size_pct: 20
    max_concurrent_positions: 5
    risk_per_trade_pct: 3.0
    max_drawdown_pct: 2.5
```

**Statut** : ✅ **INCHANGÉ** (contrainte immuable)

---

### 2. MIN TRADE = 11 USDT - ENVIRONNEMENT

**Localisations** :
- `config/config.yaml` ligne 1088 : `portfolio.min_order_value_usdt: 11.0`
- `config/config.yaml` ligne 1234 : `trading_rules.min_order_value_usdt: 11.0`
- `config/config.yaml` ligne 569 : `dbe.risk_management.min_trade_value: 11.0`
- `src/adan_trading_bot/portfolio/portfolio_manager.py` : appliqué dans `open_position()` et `calculate_trade_parameters()`

**Vérification** :
```python
# PortfolioManager.open_position() - ligne ~597-604
min_trade_value = self.config.get("portfolio", {}).get("min_order_value_usdt", 11.0)
if cost < min_trade_value:
    logger.warning(f"Trade rejected: cost {cost} < min_trade_value {min_trade_value}")
    return False
```

**Statut** : ✅ **RESPECTÉ PARTOUT** (contrainte immuable)

---

### 3. DBE (Dynamic Behavior Engine) - MODULATEUR

**Localisation** : `config/config.yaml` lignes 430-570

**Structure Actuelle** :
```yaml
dbe:
  aggressiveness_by_tier:
    Micro:
      bear:
        position_size_multiplier: 1.2
        sl_multiplier: 0.9
        tp_multiplier: 1.1
      bull:
        position_size_multiplier: 1.4
        sl_multiplier: 1.3
        tp_multiplier: 1.6
      # ... autres régimes
    Small:
      # ... multiplicateurs par régime
    # ... autres paliers
  
  aggressiveness_decay:
    micro: 1.1
    small: 1.03
    medium: 1.0
    high: 0.85
    enterprise: 0.6
  
  regime_parameters:
    bear:
      position_size_multiplier: 0.9
      sl_multiplier: 0.8
      tp_multiplier: 0.9
    bull:
      position_size_multiplier: 1.1
      sl_multiplier: 1.2
      tp_multiplier: 1.5
    # ... autres régimes
```

**Problème Identifié** :
- DBE a **deux sources de multiplicateurs** : `aggressiveness_by_tier` ET `regime_parameters`
- Les multiplicateurs peuvent être > 1.0 ou < 1.0 (ex: Micro bull = 1.4 = +40%)
- **Pas de limite explicite** sur l'ajustement (devrait être ±15% max selon le plan)
- DBE calcule aussi une `position_size_pct` propre, pas juste un multiplicateur

**Statut** : ⚠️ **À CLARIFIER** (multiplicateurs existent mais pas limités formellement)

---

### 4. OPTUNA (Stratège) - PARAMÈTRES DE BASE

**Localisation** : `config/config.yaml` lignes 1330-1750 (workers w1, w2, w3, w4)

**Valeurs Optuna Pures** :

#### W1 (Ultra-Stable Scalper)
```yaml
workers.w1:
  trading_parameters:
    position_size_pct: 0.1121
    stop_loss_pct: 0.0253
    take_profit_pct: 0.0321
    risk_per_trade_pct: 0.01
    max_concurrent_positions: 3
    min_holding_period_steps: 5
  
  agent_config:
    learning_rate: 1.0838581269344744e-05
    batch_size: 128
    gamma: 0.9745456241801775
    ent_coef: 0.010970372201012518
    vf_coef: 0.5159725093210579
```

#### W2 (Moderate Swing Trader)
```yaml
workers.w2:
  trading_parameters:
    position_size_pct: 0.25
    stop_loss_pct: 0.025
    take_profit_pct: 0.05
    risk_per_trade_pct: 0.015
    max_concurrent_positions: 3
    min_holding_period_steps: 10
```

#### W3 (Aggressive Position Trader)
```yaml
workers.w3:
  trading_parameters:
    position_size_pct: 0.258
    stop_loss_pct: 0.0744
    take_profit_pct: 0.1143
    risk_per_trade_pct: 0.0232
    max_concurrent_positions: 1
    min_holding_period_steps: 140
```

#### W4 (Sharpe Optimized Day Trader)
```yaml
workers.w4:
  trading_parameters:
    position_size_pct: 0.2
    stop_loss_pct: 0.012
    take_profit_pct: 0.02
    risk_per_trade_pct: 0.012
    max_concurrent_positions: 4
    min_holding_period_steps: 5
```

**Problème Identifié** :
- Les valeurs Optuna sont bien stockées dans `workers.wX.trading_parameters`
- **MAIS** il y a aussi des doublons :
  - `workers.wX.risk_management.position_size_pct` (différent de `trading_parameters.position_size_pct`)
  - `workers.wX.trading.stop_loss_pct` et `workers.wX.trading.take_profit_pct` (différents de `trading_parameters`)
  - `workers.wX.stop_loss_pct_by_tier` et `workers.wX.take_profit_pct_by_tier` (par palier)
  - `workers.wX.risk_per_trade_pct_by_tier` (par palier)

**Statut** : ⚠️ **CONFUS** (plusieurs sources de vérité pour les mêmes paramètres)

---

### 5. FLUX DE DÉCISION RÉEL (Traçage)

#### Étape 1 : Déterminer le Palier
```python
# PortfolioManager.get_current_tier() - ligne ~1300
def get_current_tier(self) -> Dict[str, Any]:
    capital = self.get_portfolio_value()
    for tier in self.config['capital_tiers']:
        if tier['min_capital'] <= capital < tier['max_capital']:
            return tier
    return self.config['capital_tiers'][-1]  # Enterprise par défaut
```

**Résultat** : Palier déterminé correctement selon capital courant ✅

---

#### Étape 2 : Charger Paramètres de Base
```python
# DynamicBehaviorEngine._get_tier_based_parameters() - ligne ~395
def _get_tier_based_parameters(self, worker_id, tier_name):
    worker = self.config['workers'][worker_id]
    
    # Cherche SL/TP de base
    if 'stop_loss_pct_by_tier' in worker:
        base_sl = worker['stop_loss_pct_by_tier'].get(tier_name)
    else:
        base_sl = worker.get('trading_parameters', {}).get('stop_loss_pct')
    
    # Cherche position_size de base
    if 'position_size_pct_by_tier' in worker:
        base_pos = worker['position_size_pct_by_tier'].get(tier_name)
    else:
        base_pos = worker.get('trading_parameters', {}).get('position_size_pct')
    
    # Applique aggressiveness_decay
    tier_adj_pos = base_pos * dbe.aggressiveness_decay[tier_name.lower()]
    
    return {
        'sl_pct': base_sl,
        'tp_pct': base_tp,
        'position_size_pct': tier_adj_pos
    }
```

**Problème** :
- Cherche d'abord `*_by_tier`, puis fallback sur `trading_parameters`
- Applique `aggressiveness_decay` **directement** (pas un multiplicateur DBE, mais une réduction de base)
- **Pas de modulation de régime** sur SL/TP (ils restent constants)

**Résultat** : Paramètres de base chargés, mais avec confusion sur la source ⚠️

---

#### Étape 3 : Appliquer Modulation DBE
```python
# DynamicBehaviorEngine.compute_dynamic_modulation() - ligne ~450
def compute_dynamic_modulation(self, market_regime, tier_name):
    base_params = self._get_tier_based_parameters(...)
    
    # Récupère multiplicateurs DBE par régime ET palier
    regime_mult = dbe.aggressiveness_by_tier[tier_name][market_regime]
    
    # Applique multiplicateurs
    modulated_pos = base_params['position_size_pct'] * regime_mult['position_size_multiplier']
    modulated_sl = base_params['sl_pct'] * regime_mult['sl_multiplier']
    modulated_tp = base_params['tp_pct'] * regime_mult['tp_multiplier']
    
    # Clamp par tier
    tier_cap = tier['max_position_size_pct']
    final_pos = min(modulated_pos, tier_cap)
    
    return {
        'position_size_pct': final_pos,
        'sl_pct': modulated_sl,
        'tp_pct': modulated_tp
    }
```

**Problème** :
- Multiplicateurs DBE peuvent être > 1.0 (ex: Micro bull = 1.4 = +40%)
- **Pas de limite formelle** sur l'ajustement (devrait être ±15% max)
- Clamp par tier appliqué, mais pas de clamp absolu (min/max SL/TP)

**Résultat** : Modulation appliquée, mais sans limite formelle ⚠️

---

#### Étape 4 : Appliquer Contraintes Finales
```python
# PortfolioManager.open_position() - ligne ~597
def open_position(self, ...):
    # Récupère paramètres finaux du DBE
    risk_params = dbe.calculate_trade_parameters(...)
    
    # Calcule notional
    notional = capital * risk_params['position_size_pct']
    
    # Vérifie min_trade
    if notional < 11.0:
        logger.warning(f"Trade rejected: notional {notional} < 11.0")
        return False
    
    # Vérifie max_positions
    if len(open_positions) >= tier['max_concurrent_positions']:
        return False
    
    # Ouvre la position
    position.open(...)
    return True
```

**Résultat** : Contraintes finales appliquées (min_trade, max_positions) ✅

---

## 🔴 POINTS DE CONFLIT IDENTIFIÉS

### Conflit 1 : Plusieurs Sources de Vérité pour les Paramètres Optuna
- `workers.wX.trading_parameters` (source principale)
- `workers.wX.risk_management.position_size_pct` (doublon)
- `workers.wX.trading.stop_loss_pct` (doublon)
- `workers.wX.stop_loss_pct_by_tier` (par palier)
- `workers.wX.risk_per_trade_pct_by_tier` (par palier)

**Impact** : Confusion sur quelle valeur utiliser, risque de divergence

---

### Conflit 2 : DBE Applique Multiplicateurs Sans Limite Formelle
- Multiplicateurs DBE peuvent être > 1.0 (ex: Micro bull = 1.4 = +40%)
- **Pas de limite explicite** (devrait être ±15% max selon le plan)
- Pas de clamp absolu sur SL/TP (min/max globaux)

**Impact** : DBE peut écrasement les valeurs Optuna au lieu de les moduler légèrement

---

### Conflit 3 : Aggressiveness_decay Appliqué Directement à la Base
- `aggressiveness_decay` réduit la position_size de base (ex: enterprise = 0.6 = -40%)
- **Pas clairement marqué** comme modulation DBE ou contrainte environnement
- Appliqué avant les multiplicateurs de régime

**Impact** : Hiérarchie peu claire (est-ce Optuna, DBE, ou environnement ?)

---

### Conflit 4 : Position_size Calculée Différemment Selon le Contexte
- Dans `_get_tier_based_parameters()` : `base_pos * aggressiveness_decay`
- Dans `calculate_trade_parameters()` : recalculée à partir de `exposure_range` + `desired_position_size`
- Dans `open_position()` : vérifiée contre `tier.max_position_size_pct`

**Impact** : Même paramètre calculé de 3 façons différentes, risque de divergence

---

## ✅ POINTS POSITIFS

1. **Paliers bien définis** : capital_tiers respecte les contraintes immuables
2. **Min_trade = 11 USDT respecté partout** : vérifications multiples en place
3. **Valeurs Optuna stockées** : `workers.wX.trading_parameters` existe
4. **Multiplicateurs DBE existent** : structure pour la modulation en place
5. **Clamp par tier appliqué** : position_size limitée par `max_position_size_pct`

---

## 📋 RÉSUMÉ DE LA CARTOGRAPHIE

| Couche | Statut | Problème |
|--------|--------|---------|
| **Environnement (Paliers)** | ✅ OK | Aucun (inchangé) |
| **Min Trade = 11 USDT** | ✅ OK | Aucun (respecté partout) |
| **DBE (Modulateur)** | ⚠️ À CLARIFIER | Multiplicateurs sans limite formelle, aggressiveness_decay confus |
| **Optuna (Stratège)** | ⚠️ CONFUS | Plusieurs sources de vérité, doublons |
| **Flux de Décision** | ⚠️ DIVERGENT | Position_size calculée de 3 façons différentes |

---

## 🎯 PROCHAINES ÉTAPES (T2-T3)

1. **Clarifier la hiérarchie** : Optuna → DBE (modulation ±15%) → Contraintes (env + paliers)
2. **Éliminer les doublons** : Une seule source de vérité pour chaque paramètre
3. **Limiter DBE** : Multiplicateurs bornés à ±15% max
4. **Centraliser le calcul** : Une seule fonction pour calculer les paramètres finaux
5. **Documenter clairement** : Chaque couche a un rôle unique et non-overlapping

---

## 📝 CONCLUSION T1

La hiérarchie **existe déjà partiellement**, mais elle est **confuse et divergente** :
- Les paliers et min_trade sont bien respectés (environnement OK)
- Les valeurs Optuna existent mais ont des doublons
- DBE a des multiplicateurs mais sans limite formelle
- Le flux de décision a plusieurs chemins pour le même paramètre

**Prochaine étape** : T2 - Définir formellement la nouvelle hiérarchie, puis T3 - Refactoriser config.yaml pour la refléter clairement.
