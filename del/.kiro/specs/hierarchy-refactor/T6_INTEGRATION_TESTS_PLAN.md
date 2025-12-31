# T6 : Écrire Tests d'Intégration de Hiérarchie

## 🎯 Objectif

Créer une suite de tests d'intégration qui valide le système complet (PortfolioManager + DBE + Config) fonctionnant ensemble dans des scénarios réalistes.

## 📋 Plan de Tests d'Intégration

### Test 1 : Flux Complet de Décision Hiérarchique

**Objectif** : Valider que la hiérarchie complète fonctionne de bout en bout.

**Scénarios** :
- W1 avec capital faible (15 USDT) en régime bear
- W2 avec capital moyen (150 USDT) en régime bull
- W3 avec capital élevé (500 USDT) en régime volatile
- W4 avec capital très élevé (2000 USDT) en régime sideways

**Validations** :
- ✅ Palier correct déterminé
- ✅ Optuna base chargé correctement
- ✅ DBE modulation appliquée (±15%)
- ✅ Contraintes finales respectées
- ✅ Notional ≥ 11 USDT

### Test 2 : Gestion des Cas Limites

**Objectif** : Valider le comportement aux limites du système.

**Cas Limites** :
- Capital exactement à la limite d'un palier (30.0, 100.0, 300.0, 1000.0)
- Capital juste au-dessus/en-dessous d'une limite
- Position calculée < 11 USDT (ajustement automatique)
- Position calculée > max_position_size_pct (clamp)

**Validations** :
- ✅ Palier correct sélectionné
- ✅ Ajustements automatiques appliqués
- ✅ Aucune exception levée
- ✅ Résultats cohérents

### Test 3 : Cohérence Multi-Régimes

**Objectif** : Valider que la hiérarchie fonctionne correctement pour tous les régimes.

**Régimes Testés** :
- bull (position +10%, SL +15%, TP +15%)
- bear (position -10%, SL -15%, TP -10%)
- sideways (position 0%, SL 0%, TP 0%)
- volatile (position -15%, SL +15%, TP -15%)

**Validations** :
- ✅ Ajustements corrects appliqués
- ✅ Limites ±15% respectées
- ✅ Résultats cohérents entre régimes
- ✅ Logging détaillé généré

### Test 4 : Cohérence Multi-Workers

**Objectif** : Valider que chaque worker a ses propres paramètres Optuna.

**Workers Testés** :
- W1 :