# T4 : Adapter DynamicBehaviorEngine pour Modulateur Relatif Pur

## 🎯 Objectif T4

Modifier le code DBE pour :
1. Lire `workers.wX.trading_parameters` comme base Optuna (source unique de vérité)
2. Appliquer multiplicateurs relatifs (±15% max) au lieu de valeurs absolues
3. Respecter les caps de palier et min_trade=11
4. Éliminer les doublons et les calculs divergents

---

## 📍 Fichiers à Modifier

### Fichier Principal
- `src/adan_trading_bot/portfolio/portfolio_manager.py`

### Méthodes Clés à Refactoriser
1. `_get_tier_based_parameters()` (ligne ~395)
2. `compute_dynamic_modulation()` (ligne ~450)
3. `calculate_trade_parameters()` (ligne ~1679)
4. `open_position()` (ligne ~491)

---

## 🔍 Analyse des Méthodes Actuelles

### Méthode 1 : `_get_tier_based_parameters()`

**Localisation** : Ligne ~395

**Comportement Actuel** :
```python
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

**Problèmes** :
- Cherche d'abord `*_by_tier`, puis fallback sur `trading_parameters`
- Applique `aggressiveness_decay` directement (pas un multiplicateur DBE clair)
- Pas de source unique de vérité

**Refactorisation Proposée** :
```python
def _get_tier_based_parameters(self, worker_id, tier_name):
    """
    Récupère les paramètres de base Optuna pour un worker et un palier.
    
    Hiérarchie :
    1. Charger trading_parameters (source unique de vérité Optuna)
    2. Retourner les valeurs pures (pas de modulation ici)
    
    La modulation DBE sera appliquée dans compute_dynamic_modulation()
    """
    worker = self.config['workers'][worker_id]
    
    # Source unique de vérité : trading_parameters
    trading_params = worker.get('trading_parameters', {})
    
    base_params = {
        'position_size_pct': trading_params.get('position_size_pct', 0.1),
        'stop_loss_pct': trading_params.get('stop_loss_pct', 0.02),
        'take_profit_pct': trading_params.get('take_profit_pct', 0.04),
        'risk_per_trade_pct': trading_params.get('risk_per_trade_pct', 0.01),
    }
    
    return base_params
```

---

### Méthode 2 : `compute_dynamic_modulation()`

**Localisation** : Ligne ~450

**Comportement Actuel** :
```python
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

**Problèmes** :
- Multiplicateurs DBE peuvent être > 1.0 (ex: Micro bull = 1.4 = +40%)
- Pas de limite formelle sur l'ajustement (devrait être ±15% max)
- Pas de clamp absolu sur SL/TP (min/max globaux)

**Refactorisation Proposée** :
```python
def compute_dynamic_modulation(self, market_regime, tier_name, worker_id):
    """
    Applique la modulation DBE aux paramètres de base Optuna.
    
    Formule : adjusted_param = base_param × (1 + dbe_multiplier)
    où dbe_multiplier ∈ [-0.15, +0.15]
    
    Hiérarchie :
    1. Charger base Optuna
    2. Appliquer multiplicateurs DBE (±15% max)
    3. Clamp par hard_constraints (min/max absolus)
    4. Clamp par tier (max_position_size_pct)
    """
    # Étape 1 : Charger base Optuna
    base_params = self._get_tier_based_parameters(worker_id, tier_name)
    
    # Étape 2 : Récupérer multiplicateurs DBE
    # Utiliser regime_parameters (global) au lieu de aggressiveness_by_tier (par tier)
    # Car la modulation DBE doit être légère et relative, pas dépendante du palier
    regime_mult = self.config['dbe'].get('regime_parameters', {}).get(market_regime, {})
    
    # Étape 3 : Appliquer multiplicateurs avec limite ±15%
    dbe_pos_mult = regime_mult.get('position_size_multiplier', 1.0)
    dbe_sl_mult = regime_mult.get('sl_multiplier', 1.0)
    dbe_tp_mult = regime_mult.get('tp_multiplier', 1.0)
    
    # Convertir multiplicateurs absolus en relatifs et borner à ±15%
    # Ex: 1.1 → +10%, 0.9 → -10%, 1.4 → +40% (clampé à +15%)
    pos_adjustment = min(max(dbe_pos_mult - 1.0, -0.15), 0.15)
    sl_adjustment = min(max(dbe_sl_mult - 1.0, -0.15), 0.15)
    tp_adjustment = min(max(dbe_tp_mult - 1.0, -0.15), 0.15)
    
    # Appliquer ajustements
    adjusted_pos = base_params['position_size_pct'] * (1 + pos_adjustment)
    adjusted_sl = base_params['stop_loss_pct'] * (1 + sl_adjustment)
    adjusted_tp = base_params['take_profit_pct'] * (1 + tp_adjustment)
    
    # Étape 4 : Clamp par hard_constraints (min/max absolus)
    hard_constraints = self.config['environment']['hard_constraints']
    
    adjusted_sl = max(min(adjusted_sl, hard_constraints['stop_loss_pct']['max']),
                      hard_constraints['stop_loss_pct']['min'])
    adjusted_tp = max(min(adjusted_tp, hard_constraints['take_profit_pct']['max']),
                      hard_constraints['take_profit_pct']['min'])
    
    # Étape 5 : Clamp par tier (max_position_size_pct)
    tier_config = self._get_tier_config(tier_name)
    adjusted_pos = min(adjusted_pos, tier_config['max_position_size_pct'])
    adjusted_pos = max(adjusted_pos, hard_constraints['min_position_size_pct'])
    
    return {
        'position_size_pct': adjusted_pos,
        'stop_loss_pct': adjusted_sl,
        'take_profit_pct': adjusted_tp,
        'risk_per_trade_pct': base_params['risk_per_trade_pct'],  # Pas de modulation
    }
```

---

### Méthode 3 : `calculate_trade_parameters()`

**Localisation** : Ligne ~1679

**Comportement Actuel** :
```python
def calculate_trade_parameters(self, ...):
    risk_params = compute_dynamic_modulation(...)
    
    # Calcule position_pct à partir de exposure_range + desired_position_size
    # (recalcul divergent)
    
    # Vérifie min_trade_value
    if notional < min_trade_value:
        # Remonte la taille
        ...
    
    return risk_params
```

**Problèmes** :
- Recalcule position_size de façon divergente
- Logique de min_trade_value confuse

**Refactorisation Proposée** :
```python
def calculate_trade_parameters(self, worker_id, capital, market_regime):
    """
    Calcule les paramètres finaux de trading en appliquant la hiérarchie séquentiellement.
    
    Flux :
    1. Déterminer palier (Environnement)
    2. Charger base Optuna (Optuna)
    3. Appliquer modulation DBE (DBE)
    4. Appliquer contraintes environnement (Environnement)
    5. Vérifier notional ≥ 11 USDT (Environnement)
    """
    # Étape 1 : Déterminer palier
    tier_name = self._get_tier_name(capital)
    tier_config = self._get_tier_config(tier_name)
    
    # Étape 2-4 : Appliquer hiérarchie (déjà fait dans compute_dynamic_modulation)
    modulated_params = self.compute_dynamic_modulation(market_regime, tier_name, worker_id)
    
    # Étape 5 : Vérifier notional ≥ 11 USDT
    notional = capital * modulated_params['position_size_pct']
    min_trade_value = self.config['environment']['hard_constraints']['min_order_value_usdt']
    
    if notional < min_trade_value:
        # Rejeter le trade (pas de remontée)
        logger.warning(f"Trade rejected: notional {notional} < min_trade_value {min_trade_value}")
        return None
    
    # Vérifier max_positions
    if len(self.open_positions) >= tier_config['max_concurrent_positions']:
        logger.warning(f"Trade rejected: max_concurrent_positions {tier_config['max_concurrent_positions']} reached")
        return None
    
    return modulated_params
```

---

## 📋 Checklist de Refactorisation T4

### Phase 1 : Préparation
- [ ] Lire complètement `portfolio_manager.py` (DynamicBehaviorEngine)
- [ ] Identifier tous les appels à `_get_tier_based_parameters()`, `compute_dynamic_modulation()`, `calculate_trade_parameters()`
- [ ] Identifier tous les doublons de paramètres (risk_management, trading, *_by_tier)

### Phase 2 : Refactorisation du Code
- [ ] Refactoriser `_get_tier_based_parameters()` pour lire uniquement `trading_parameters`
- [ ] Refactoriser `compute_dynamic_modulation()` pour appliquer multiplicateurs ±15% max
- [ ] Refactoriser `calculate_trade_parameters()` pour centraliser la hiérarchie
- [ ] Ajouter logging détaillé (quelle couche a modifié quoi)

### Phase 3 : Tests Unitaires
- [ ] Tester `_get_tier_based_parameters()` avec différents workers
- [ ] Tester `compute_dynamic_modulation()` avec différents régimes
- [ ] Tester que multiplicateurs sont bornés à ±15%
- [ ] Tester que min_trade=11 est respecté
- [ ] Tester que paliers sont respectés

### Phase 4 : Validation
- [ ] Vérifier que aucune régression dans les tests existants
- [ ] Vérifier que hiérarchie est respectée (Env > DBE > Optuna)
- [ ] Vérifier que contraintes immuables sont préservées

---

## 🎯 Résultat Attendu T4

Après T4, le code DBE doit :
1. ✅ Lire `workers.wX.trading_parameters` comme source unique de vérité
2. ✅ Appliquer multiplicateurs DBE ±15% max (relatifs, pas absolus)
3. ✅ Respecter hard_constraints (min_trade=11, bornes SL/TP, etc.)
4. ✅ Respecter capital_tiers (max_position_size_pct, max_concurrent_positions, etc.)
5. ✅ Centraliser la hiérarchie dans une seule fonction
6. ✅ Ajouter logging détaillé pour le débogage

---

## 📝 Notes Importantes

- **Pas de modification des valeurs** : Les multiplicateurs DBE dans config.yaml restent inchangés (pour l'instant)
- **Conversion multiplicateurs** : Les multiplicateurs absolus (ex: 1.4) doivent être convertis en relatifs (ex: +40% → clampé à +15%)
- **Logging** : Ajouter des logs pour tracer chaque étape de la hiérarchie
- **Tests** : Écrire des tests pour valider la hiérarchie

---

## ✅ Prochaines Étapes

Après T4 :
- T5 : Centraliser la décision finale dans PortfolioManager
- T6-T7 : Tests et validation
- T8-T10 : Relancer Optuna, injecter hyperparamètres, relancer entraînement


---

## 📊 ANALYSE DU CODE DBE ACTUEL

**Fichier** : `src/adan_trading_bot/environment/dynamic_behavior_engine.py` (2352 lignes)

**Méthodes Clés** :
- `_get_tier_based_parameters()` (ligne 395) : Récupère SL/TP/PosSize de base par tier
- `compute_dynamic_modulation()` (ligne 489) : Orchestre le calcul des paramètres
- `calculate_trade_parameters()` (ligne 1576) : Calcule les paramètres finaux de trade

**Problèmes Identifiés** :

1. **`_get_tier_based_parameters()` (ligne 395)** :
   - Cherche d'abord `stop_loss_pct_by_tier`, puis fallback sur `trading_parameters`
   - Pas de source unique de vérité
   - Applique `aggressiveness_decay` directement (pas un multiplicateur DBE clair)
   - Mélange les responsabilités (Optuna + DBE + Environnement)

2. **`compute_dynamic_modulation()` (ligne 489)** :
   - A un bypass "DISABLED_FOR_OPTUNA" (ligne 489)
   - Ignore la modulation de régime sur SL/TP
   - Mais applique quand même des multiplicateurs sur position_size
   - Logique confuse et non cohérente

3. **`calculate_trade_parameters()` (ligne 1576)** :
   - Recalcule position_size de façon divergente
   - Utilise `exposure_range` du palier
   - Applique `desired_position_size` de l'agent
   - Clamp entre 5-80%, puis par tier.max_position_size_pct
   - Remonte à min_trade_value si < 11 USDT
   - Plusieurs chemins pour le même paramètre

**Stratégie de Refactorisation** :
1. Centraliser la lecture des paramètres Optuna dans `_get_tier_based_parameters()`
2. Appliquer multiplicateurs DBE ±15% max dans `compute_dynamic_modulation()`
3. Centraliser la décision finale dans `calculate_trade_parameters()`
4. Ajouter logging détaillé pour tracer chaque étape

---

## 🔧 MODIFICATIONS CONCRÈTES À FAIRE

### Modification 1 : `_get_tier_based_parameters()` (ligne 395)

**Avant** :
```python
def _get_tier_based_parameters(self, worker_key: str, current_tier: Any) -> Tuple[float, float, float]:
    # Cherche d'abord *_by_tier, puis fallback
    base_sl = (
        worker_config.get("stop_loss_pct_by_tier", {}).get(tier_key)
        or worker_config.get("risk_management", {}).get("stop_loss_pct")
        or self.config.get("risk_parameters", {}).get("base_sl_pct", 0.02)
    )
    # Applique aggressiveness_decay directement
    tier_adjusted_pos_size = float(base_pos_size) * float(aggressiveness)
```

**Après** :
```python
def _get_tier_based_parameters(self, worker_key: str, current_tier: Any) -> Tuple[float, float, float]:
    # Source unique de vérité : trading_parameters
    trading_params = worker_config.get('trading_parameters', {})
    
    base_sl = trading_params.get('stop_loss_pct', 0.02)
    base_tp = trading_params.get('take_profit_pct', 0.04)
    base_pos = trading_params.get('position_size_pct', 0.1)
    
    # Pas d'aggressiveness_decay ici - sera appliqué dans compute_dynamic_modulation()
    return float(base_sl), float(base_tp), float(base_pos)
```

### Modification 2 : `compute_dynamic_modulation()` (ligne 489)

**Avant** :
```python
def compute_dynamic_modulation(self, env=None, risk_horizon: float = 0.0) -> Dict[str, Any]:
    # BYPASS REGIME MODULATION
    final_sl = base_sl
    final_tp = base_tp
    final_pos = tier_adj_pos
```

**Après** :
```python
def compute_dynamic_modulation(self, env=None, risk_horizon: float = 0.0) -> Dict[str, Any]:
    # Étape 1 : Charger base Optuna
    base_sl, base_tp, base_pos = self._get_tier_based_parameters(worker_key, current_tier)
    
    # Étape 2 : Appliquer multiplicateurs DBE ±15% max
    regime_mult = self.config.get('dbe', {}).get('regime_parameters', {}).get(regime, {})
    
    # Convertir multiplicateurs absolus en relatifs et borner à ±15%
    pos_adjustment = min(max(regime_mult.get('position_size_multiplier', 1.0) - 1.0, -0.15), 0.15)
    sl_adjustment = min(max(regime_mult.get('sl_multiplier', 1.0) - 1.0, -0.15), 0.15)
    tp_adjustment = min(max(regime_mult.get('tp_multiplier', 1.0) - 1.0, -0.15), 0.15)
    
    # Appliquer ajustements
    adjusted_pos = base_pos * (1 + pos_adjustment)
    adjusted_sl = base_sl * (1 + sl_adjustment)
    adjusted_tp = base_tp * (1 + tp_adjustment)
    
    # Étape 3 : Clamp par hard_constraints
    hard_constraints = self.config.get('environment', {}).get('hard_constraints', {})
    adjusted_sl = max(min(adjusted_sl, hard_constraints.get('stop_loss_pct', {}).get('max', 0.20)), 
                      hard_constraints.get('stop_loss_pct', {}).get('min', 0.005))
    adjusted_tp = max(min(adjusted_tp, hard_constraints.get('take_profit_pct', {}).get('max', 0.50)), 
                      hard_constraints.get('take_profit_pct', {}).get('min', 0.01))
    
    # Étape 4 : Clamp par tier
    tier_cap = current_tier.get('max_position_size_pct', 0.90) / 100.0
    adjusted_pos = min(adjusted_pos, tier_cap)
    
    return {
        'sl_pct': adjusted_sl,
        'tp_pct': adjusted_tp,
        'position_size_pct': adjusted_pos,
        'regime': regime,
        'regime_confidence': confidence,
    }
```

### Modification 3 : `calculate_trade_parameters()` (ligne 1576)

**Avant** :
```python
def calculate_trade_parameters(self, capital, worker_pref_pct, tier_config, ...):
    # Recalcule position_size de façon divergente
    position_pct = (
        min_position_pct
        + (max_position_pct - min_position_pct) * normalized_size
    )
    position_pct = np.clip(position_pct, 0.05, 0.80)
    max_position = tier_config.get("max_position_size_pct", 0.5) / 100.0
    position_pct = min(max_position, position_pct)
```

**Après** :
```python
def calculate_trade_parameters(self, capital, worker_pref_pct, tier_config, ...):
    # Utiliser directement les paramètres de compute_dynamic_modulation()
    risk_params = self.compute_dynamic_modulation()
    
    position_pct = risk_params.get('position_size_pct', 0.1)
    sl_pct = risk_params.get('sl_pct', 0.02)
    tp_pct = risk_params.get('tp_pct', 0.04)
    
    # Vérifier notional ≥ 11 USDT
    notional = capital * position_pct
    min_trade_value = self.config.get('environment', {}).get('hard_constraints', {}).get('min_order_value_usdt', 11.0)
    
    if notional < min_trade_value:
        if capital < min_trade_value:
            return {'feasible': False, 'reason': 'Capital insuffisant'}
        position_pct = min_trade_value / capital
    
    return {
        'feasible': True,
        'position_size_pct': position_pct,
        'sl_pct': sl_pct,
        'tp_pct': tp_pct,
        'regime': risk_params.get('regime', 'neutral'),
    }
```

---

## ✅ CHECKLIST DE REFACTORISATION

- [ ] Modifier `_get_tier_based_parameters()` pour lire uniquement `trading_parameters`
- [ ] Modifier `compute_dynamic_modulation()` pour appliquer multiplicateurs ±15% max
- [ ] Modifier `calculate_trade_parameters()` pour utiliser les paramètres de `compute_dynamic_modulation()`
- [ ] Ajouter logging détaillé (quelle couche a modifié quoi)
- [ ] Tester que min_trade=11 est respecté
- [ ] Tester que paliers sont respectés
- [ ] Vérifier aucune régression dans les tests existants

---

## 📝 PROCHAINES ÉTAPES

Après T4 :
- T5 : Centraliser la décision finale dans PortfolioManager
- T6-T7 : Tests et validation
- T8-T10 : Relancer Optuna, injecter hyperparamètres, relancer entraînement
