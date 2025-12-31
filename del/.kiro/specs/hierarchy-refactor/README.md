# Chantier Refactorisation Hiérarchique - ADAN Trading Bot

## 🎯 Objectif Global

Restructurer la hiérarchie **Environnement → DBE → Optuna** pour éliminer les conflits et établir une autorité claire à chaque couche, en respectant strictement les contraintes immuables :
- ✅ Paliers (capital_tiers) : **AUCUNE modification de valeurs**
- ✅ Intervalles d'exposition par palier : **INCHANGÉS**
- ✅ Min trade = 11 USDT : **NON NÉGOCIABLE**

---

## 📋 Structure du Chantier

Ce chantier est organisé en **10 tâches** (T1-T10) avec une approche méthodique :

### Phase 1 : Analyse et Spécification (T1-T3) ✅ COMPLÉTÉE

| Tâche | Titre | Statut | Livrable |
|-------|-------|--------|----------|
| **T1** | Cartographier la hiérarchie réelle actuelle | ✅ | `CARTOGRAPHY_T1.md` |
| **T2** | Définir et valider la nouvelle hiérarchie formelle | ✅ | `HIERARCHY_SPECIFICATION_T2.md` |
| **T3** | Refactoriser config/config.yaml | ✅ | `T3_REFACTORING_SUMMARY.md` |

### Phase 2 : Refactorisation Code (T4-T5) ⏳ À FAIRE

| Tâche | Titre | Statut | Livrable |
|-------|-------|--------|----------|
| **T4** | Adapter DynamicBehaviorEngine | ⏳ | `T4_DBE_REFACTORING_PLAN.md` |
| **T5** | Centraliser la décision finale | ⏳ | À créer |

### Phase 3 : Tests et Validation (T6-T7) ⏳ À FAIRE

| Tâche | Titre | Statut | Livrable |
|-------|-------|--------|----------|
| **T6** | Écrire tests d'intégration de hiérarchie | ⏳ | À créer |
| **T7** | Exécuter batterie de tests existante | ⏳ | À créer |

### Phase 4 : Optuna et Entraînement (T8-T10) ⏳ À FAIRE

| Tâche | Titre | Statut | Livrable |
|-------|-------|--------|----------|
| **T8** | Relancer Optuna avec nouvelle hiérarchie | ⏳ | À créer |
| **T9** | Injecter hyperparamètres Optuna | ⏳ | À créer |
| **T10** | Relancer entraînement final | ⏳ | À créer |

---

## 📚 Documentation Disponible

### Spécifications Complètes

1. **`CARTOGRAPHY_T1.md`** - Analyse détaillée de la hiérarchie actuelle
   - État réel du système
   - Points de conflit identifiés
   - Conclusions et recommandations

2. **`HIERARCHY_SPECIFICATION_T2.md`** - Spécification formelle de la nouvelle hiérarchie
   - Rôles de chaque couche (Environnement, DBE, Optuna)
   - Flux de décision séquentiel (6 étapes)
   - Formules de calcul
   - Exemple concret validé

3. **`T3_REFACTORING_SUMMARY.md`** - Résumé des modifications config.yaml
   - Sections ajoutées/modifiées
   - Contraintes immuables vérifiées
   - Prochaines étapes

4. **`T4_DBE_REFACTORING_PLAN.md`** - Plan détaillé pour refactoriser le code DBE
   - Analyse des méthodes actuelles
   - Refactorisation proposée
   - Checklist de validation

5. **`PROGRESS_SUMMARY.md`** - Suivi global du chantier
   - État d'avancement
   - Fichiers créés
   - Prochaines actions

### Fichiers de Tâches

- **`tasks.md`** - Plan de 10 tâches avec checkboxes
- **`requirements.md`** - Spécifications des exigences (si applicable)

---

## 🏗️ Hiérarchie Formelle

### TIER 1 : ENVIRONNEMENT (Arbitre - Contraintes Absolues)

**Rôle** : Appliquer les lois inviolables du système.

**Responsabilités** :
- Déterminer le palier selon le capital courant
- Imposer les limites absolues du palier (max_position_size_pct, max_concurrent_positions, etc.)
- Imposer les limites absolues globales (min_trade=11, bornes SL/TP, etc.)
- Appliquer les contraintes FINALES (clamp, rejet si < 11 USDT, etc.)

**Contraintes Immuables** :
```yaml
capital_tiers:
  - Micro: min=11, max=30, max_pos=90%, max_conc=1, risk=4%, exposure=[70,90]
  - Small: min=30, max=100, max_pos=65%, max_conc=2, risk=2%, exposure=[35,75]
  - Medium: min=100, max=300, max_pos=48%, max_conc=3, risk=2.25%, exposure=[45,60]
  - High: min=300, max=1000, max_pos=28%, max_conc=4, risk=2.75%, exposure=[20,35]
  - Enterprise: min=1000, max=∞, max_pos=20%, max_conc=5, risk=3%, exposure=[5,15]

min_order_value_usdt = 11.0 (JAMAIS MOINS)
```

---

### TIER 2 : DBE (Tacticien - Modulation Légère)

**Rôle** : Adapter légèrement les paramètres Optuna selon les conditions de marché.

**Principe** : Multiplicateurs relatifs **bornés à ±15% maximum**.

**Formule** :
```
adjusted_param = base_param × (1 + dbe_multiplier)
où dbe_multiplier ∈ [-0.15, +0.15]
```

**Exemple** :
```
Optuna base: position_size_pct = 0.1121 (11.21%)
DBE régime bear: multiplicateur = -0.10 (-10%)
Adjusted: 0.1121 × (1 - 0.10) = 0.10089 (10.09%)
```

---

### TIER 3 : OPTUNA (Stratège - Performance Pure)

**Rôle** : Définir les paramètres de base optimisés pour chaque worker.

**Valeurs Actuelles** :
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
  # ... w3, w4
```

---

## 🔄 Flux de Décision Séquentiel

```
ENTRÉE : worker_id, capital_courant, market_regime

ÉTAPE 1 : ENVIRONNEMENT - Déterminer le Palier
  tier = determine_capital_tier(capital_courant)
  
ÉTAPE 2 : OPTUNA - Charger Valeurs de Base
  optuna_params = workers[worker_id].trading_parameters
  
ÉTAPE 3 : DBE - Appliquer Modulation Légère
  adjusted = optuna_params × (1 + dbe_multiplier)
  
ÉTAPE 4 : ENVIRONNEMENT - Appliquer Contraintes Finales
  final = clamp(adjusted, tier_max, env_max)
  
ÉTAPE 5 : ENVIRONNEMENT - Vérifier Notional
  if notional < 11.0: REJETER
  
ÉTAPE 6 : ENVIRONNEMENT - Vérifier Max Positions
  if len(open_positions) >= tier.max_concurrent_positions: REJETER
  
SORTIE : final_position, final_sl, final_tp, final_risk (ou REJET)
```

---

## 📊 Exemple Concret : W1 avec 50 USDT en Régime Bear

```
ENTRÉE : worker_id=w1, capital=50 USDT, market_regime=bear

ÉTAPE 1 : Déterminer Palier
  50 ∈ [30, 100] → Tier = "Small"
  tier_config = {max_pos: 65%, max_conc: 2, risk: 2%, exposure: [35, 75]}

ÉTAPE 2 : Charger Optuna
  base_position = 0.1121 (11.21%)
  base_sl = 0.0253 (2.53%)
  base_tp = 0.0321 (3.21%)

ÉTAPE 3 : Appliquer DBE (Régime Bear)
  dbe_mult = {position_size: -0.10, stop_loss: +0.05, take_profit: -0.05}
  adjusted_position = 0.1121 × 0.90 = 0.10089 (10.09%)
  adjusted_sl = 0.0253 × 1.05 = 0.02657 (2.66%)
  adjusted_tp = 0.0321 × 0.95 = 0.03050 (3.05%)

ÉTAPE 4 : Appliquer Contraintes Environnement
  final_position = min(0.10089, 0.65) = 0.10089 (10.09%)
  final_sl = clamp(0.02657, [0.005, 0.20]) = 0.02657 (2.66%)
  final_tp = clamp(0.03050, [0.01, 0.50]) = 0.03050 (3.05%)

ÉTAPE 5 : Vérifier Notional
  notional = 50 × 0.10089 = 5.0445 USDT
  5.0445 < 11.0 → ❌ REJET (< min_trade_value)

SORTIE : TRADE REJETÉ (notional insuffisant)
```

---

## 🚀 Prochaines Étapes

### Immédiat (T4-T5)

1. **Adapter le code DBE** (T4)
   - Lire `src/adan_trading_bot/portfolio/portfolio_manager.py`
   - Refactoriser `_get_tier_based_parameters()`
   - Refactoriser `compute_dynamic_modulation()`
   - Refactoriser `calculate_trade_parameters()`

2. **Centraliser la décision finale** (T5)
   - Créer fonction `calculate_final_trade_parameters()`
   - Appliquer hiérarchie séquentiellement

### Court terme (T6-T7)

3. **Écrire tests** (T6-T7)
   - Tests d'intégration de hiérarchie
   - Batterie de tests existante

### Moyen terme (T8-T10)

4. **Relancer Optuna et entraînement** (T8-T10)
   - Relancer Optuna pour chaque worker
   - Injecter hyperparamètres
   - Relancer entraînement final

---

## 📝 Directives Méthodologiques

- **Un module à la fois** : Terminer une tâche complètement avant la suivante
- **Tests à chaque étape** : Valider avant de passer au suivant
- **Pas de modification de valeurs immuables** : Paliers, min_trade, intervalles
- **Documentation claire** : Chaque étape doit être compréhensible
- **Utiliser tous les outils** : readFile, strReplace, getDiagnostics, executeBash, etc.
- **Commit réguliers** : Après chaque tâche majeure
- **Ne pas s'arrêter** : Continuer jusqu'à T10 complète

---

## 📈 Progression Globale

```
T1 : Cartographie          ████████████████████ 100% ✅
T2 : Spécification         ████████████████████ 100% ✅
T3 : Config Refactoring    ████████████████████ 100% ✅
T4 : DBE Refactoring       ░░░░░░░░░░░░░░░░░░░░   0% ⏳
T5 : Centralisation        ░░░░░░░░░░░░░░░░░░░░   0% ⏳
T6 : Tests Hiérarchie      ░░░░░░░░░░░░░░░░░░░░   0% ⏳
T7 : Tests Existants       ░░░░░░░░░░░░░░░░░░░░   0% ⏳
T8 : Relancer Optuna       ░░░░░░░░░░░░░░░░░░░░   0% ⏳
T9 : Injecter Optuna       ░░░░░░░░░░░░░░░░░░░░   0% ⏳
T10: Entraînement Final    ░░░░░░░░░░░░░░░░░░░░   0% ⏳

TOTAL : 30% ✅ | 70% ⏳
```

---

## 🎯 Résumé

**Travail Effectué** :
- ✅ Cartographie complète de la hiérarchie actuelle
- ✅ Spécification formelle de la nouvelle hiérarchie
- ✅ Refactorisation de config.yaml pour refléter la hiérarchie
- ✅ Plans détaillés pour les refactorisations de code

**Prochaine Étape** : **T4 - Adapter le code DBE**

**Temps Estimé** : 7-11 heures (T4-T10)

---

## 📞 Contact

**Responsable** : Kiro (Agent IA)
**Dernière Mise à Jour** : 10 décembre 2025, 11:56 UTC
**Statut** : 🔄 EN COURS - Prêt pour T4

---

## 📂 Fichiers du Chantier

```
.kiro/specs/hierarchy-refactor/
├── README.md                              (ce fichier)
├── tasks.md                               (plan de 10 tâches)
├── requirements.md                        (spécifications des exigences)
├── CARTOGRAPHY_T1.md                      (analyse hiérarchie actuelle)
├── HIERARCHY_SPECIFICATION_T2.md          (spécification nouvelle hiérarchie)
├── T3_REFACTORING_SUMMARY.md              (résumé refactoring config.yaml)
├── T4_DBE_REFACTORING_PLAN.md             (plan refactoring code DBE)
└── PROGRESS_SUMMARY.md                    (suivi global du chantier)
```

---

**Bon courage pour la suite du chantier ! 🚀**
