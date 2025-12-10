# T5 : Centraliser la Décision Finale dans PortfolioManager

## 🎯 Objectif

Créer une fonction centralisée `calculate_final_trade_parameters()` qui applique la hiérarchie complète :

```
Environnement (Arbitre) → DBE (Tacticien) → Optuna (Stratège) → Environnement (Arbitre)
```

## 📋 Spécification

### Fonction Principale : `calculate_final_trade_parameters()`

**Signature** :
```python
def calculate_final_trade_parameters(
    self,
    worker_id: int,
    capital: float,
    market_regime: str,
    current_step: int,
) -> Dict[str, float]:
    """
    Applique la hiérarchie complète pour calculer les paramètres finaux de trading.
    
    Hiérarchie :
    1. TIER 1 (Environnement) : Lire paliers et hard_constraints
    2. TIER 3 (Optuna) : Lire trading_parameters du worker
    3. TIER 2 (DBE) : Appliquer multiplicateurs ±15% selon régime
    4. TIER 1 (Environnement) : Appliquer contraintes finales
    
    Args:
        worker_id: ID du worker (w1, w2, w3, w4)
        capital: Capital actuel du portefeuille
        market_regime: Régime de marché (bull, bear, sideways, volatile)
        current_step: Étape actuelle de l'entraînement
    
    Returns:
        Dict avec clés : position_size_pct, stop_loss_pct, take_profit_pct, 
                         risk_per_trade_pct, notional_usdt, tier_name
    """
```

### Étapes de Calcul

#### Étape 1 : Lire Paliers et Hard Constraints (Environnement)

```python
# Lire capital_tiers pour déterminer le palier
tier_config = self._get_capital_tier(capital)
tier_name = tier_config['name']
max_position_pct = tier_config['max_position_size_pct'] / 100.0
exposure_range = tier_config['exposure_range']

# Lire hard_constraints
hard_constraints = self.config['environment']['hard_constraints']
min_trade = hard_constraints['min_order_value_usdt']
sl_bounds = hard_constraints['stop_loss_pct']
tp_bounds = hard_constraints['take_profit_pct']
```

#### Étape 2 : Lire Valeurs Optuna (Stratège)

```python
# Lire trading_parameters du worker (source unique)
worker_config = self.config['workers'][f'w{worker_id}']
trading_params = worker_config['trading_parameters']

base_position_pct = trading_params['position_size_pct']
base_sl_pct = trading_params['stop_loss_pct']
base_tp_pct = trading_params['take_profit_pct']
base_risk_pct = trading_params['risk_per_trade_pct']
```

#### Étape 3 : Appliquer DBE (Tacticien)

```python
# Lire multiplicateurs DBE pour le régime
regime_params = self.config['dbe']['regime_parameters'][market_regime]

pos_mult = regime_params['position_size_multiplier']
sl_mult = regime_params['sl_multiplier']
tp_mult = regime_params['tp_multiplier']

# Convertir en ajustements relatifs et borner à ±15%
pos_adj = min(max(pos_mult - 1.0, -0.15), 0.15)
sl_adj = min(max(sl_mult - 1.0, -0.15), 0.15)
tp_adj = min(max(tp_mult - 1.0, -0.15), 0.15)

# Appliquer ajustements
adjusted_position_pct = base_position_pct * (1 + pos_adj)
adjusted_sl_pct = base_sl_pct * (1 + sl_adj)
adjusted_tp_pct = base_tp_pct * (1 + tp_adj)
```

#### Étape 4 : Appliquer Contraintes Finales (Environnement)

```python
# Clamp SL/TP par hard_constraints
final_sl_pct = max(min(adjusted_sl_pct, sl_bounds['max']), sl_bounds['min'])
final_tp_pct = max(min(adjusted_tp_pct, tp_bounds['max']), tp_bounds['min'])

# Clamp position par palier
final_position_pct = min(adjusted_position_pct, max_position_pct)

# Vérifier notional ≥ min_trade
notional = capital * final_position_pct
if notional < min_trade:
    # Ajuster position pour atteindre min_trade
    final_position_pct = min_trade / capital
    if final_position_pct > max_position_pct:
        # Trade impossible, rejeter
        return None
```

### Logging Détaillé

Chaque étape doit être loggée pour traçabilité :

```
[TIER 1] Environnement: Palier=Medium, MaxPos=48%, MinTrade=11 USDT
[TIER 3] Optuna (w1): Pos=11.21%, SL=2.53%, TP=3.21%
[TIER 2] DBE (bull): Pos×1.10, SL×1.20, TP×1.50
[TIER 2] DBE ajusté: Pos=+10%, SL=+15% (borné), TP=+15% (borné)
[TIER 2] Après DBE: Pos=12.33%, SL=2.91%, TP=3.69%
[TIER 1] Après Env: Pos=12.33% (≤48%), SL=2.91%, TP=3.69%
[FINAL] Notional=18.50 USDT ≥ 11 USDT ✅
```

## 📁 Fichiers à Modifier

### 1. `src/adan_trading_bot/portfolio/portfolio_manager.py`

**Ajouter méthode** :
```python
def calculate_final_trade_parameters(
    self,
    worker_id: int,
    capital: float,
    market_regime: str,
    current_step: int,
) -> Optional[Dict[str, float]]:
    """Applique la hiérarchie complète pour calculer les paramètres finaux."""
    # Implémentation (voir détails ci-dessus)
```

**Modifier méthode** `open_position()` :
- Utiliser `calculate_final_trade_parameters()` au lieu de logique locale
- Passer `worker_id`, `capital`, `market_regime`, `current_step`

### 2. `src/adan_trading_bot/environment/dynamic_behavior_engine.py`

**Modifier méthode** `calculate_trade_parameters()` :
- Appeler `portfolio_manager.calculate_final_trade_parameters()`
- Retourner résultat centralisé

## ✅ Critères de Succès

1. ✅ Fonction `calculate_final_trade_parameters()` existe et fonctionne
2. ✅ Hiérarchie appliquée séquentiellement (Env → Optuna → DBE → Env)
3. ✅ Logging détaillé pour chaque étape
4. ✅ Min trade = 11 USDT garanti
5. ✅ Paliers respectés
6. ✅ DBE limité à ±15%
7. ✅ Aucune régression dans tests existants

## 🧪 Tests

Créer `tests/test_final_trade_parameters.py` :

```python
def test_hierarchy_applied_correctly():
    """Vérifier que la hiérarchie est appliquée correctement."""
    # Scénario 1 : W1 + 50 USDT + Bull
    # Scénario 2 : W2 + 150 USDT + Bear
    # Scénario 3 : W3 + 25 USDT + Volatile
    # Scénario 4 : W4 + 500 USDT + Sideways

def test_min_trade_guarantee():
    """Vérifier que min_trade=11 est garanti."""
    # Tester avec capital très faible

def test_tier_constraints():
    """Vérifier que les paliers sont respectés."""
    # Tester chaque palier

def test_dbe_bounds():
    """Vérifier que DBE est limité à ±15%."""
    # Tester chaque régime
```

## 📊 Progression

```
T1-T4 : ████████████████████ 100% ✅
T5    : ░░░░░░░░░░░░░░░░░░░░   0% ⏳ (EN COURS)
T6-T10: ░░░░░░░░░░░░░░░░░░░░   0% ⏳
```

## 🚀 Prochaines Étapes

1. Implémenter `calculate_final_trade_parameters()` dans PortfolioManager
2. Modifier `open_position()` pour utiliser la nouvelle fonction
3. Modifier DBE pour appeler la nouvelle fonction
4. Écrire tests d'intégration
5. Valider aucune régression

---

**Créé** : 10 décembre 2025
**Responsable** : Kiro (Agent IA)
**Statut** : 🔄 EN COURS
