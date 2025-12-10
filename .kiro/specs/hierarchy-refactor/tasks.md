# Plan de Refactorisation Hiérarchique - ADAN Trading Bot

## Objectif Global
Restructurer la hiérarchie **Environnement → DBE → Optuna** en respectant strictement :
- ✅ Paliers (capital_tiers) : **AUCUNE modification de valeurs**
- ✅ Intervalles d'exposition par palier : **INCHANGÉS**
- ✅ Min trade = 11 USDT : **NON NÉGOCIABLE**

---

## 📊 Plan de 10 Tâches

### ✅ T1 : Cartographier la hiérarchie réelle actuelle
- [ ] 1.1 Analyser config/config.yaml (sections environment, capital_tiers, dbe, workers)
- [ ] 1.2 Tracer le flux de décision : où Optuna pose ses valeurs, où DBE les lit, où l'environnement impose les caps
- [ ] 1.3 Documenter les points de conflit actuels (si DBE écrase Optuna, si paliers ne sont pas respectés, etc.)
- [ ] 1.4 Valider que min_trade=11 est bien appliqué partout (PortfolioManager, DBE, trading_rules)
- **Livrable** : Rapport de cartographie (en mémoire, pas de fichier)

---

### ✅ T2 : Définir et valider la nouvelle hiérarchie formelle
- [ ] 2.1 Formaliser la hiérarchie : Environnement (hard_constraints) > DBE (modulateur ±X%) > Optuna (stratège)
- [ ] 2.2 Définir les rôles exacts de chaque couche (sans modifier les valeurs des paliers)
- [ ] 2.3 Valider que la hiérarchie respecte : paliers inchangés, min_trade=11, intervalles d'exposition préservés
- [ ] 2.4 Documenter les règles de calcul final : position_final = min(optuna_base × dbe_mult, tier_max, env_max)
- **Livrable** : Spécification formelle de la hiérarchie

---

### ✅ T3 : Refactoriser config/config.yaml pour refléter la hiérarchie
- [ ] 3.1 Ajouter section `environment.hard_constraints` (min_order_value_usdt: 11, bornes SL/TP, etc.)
- [ ] 3.2 Clarifier section `dbe` : multiplicateurs relatifs uniquement (±X%), pas de valeurs absolues
- [ ] 3.3 Marquer explicitement dans `workers.wX` : `trading_parameters` = base Optuna (stratège)
- [ ] 3.4 Vérifier que `capital_tiers` reste **INCHANGÉ** (valeurs, intervalles, min_trade)
- [ ] 3.5 Ajouter commentaires explicatifs sur la hiérarchie dans le YAML
- **Livrable** : config/config.yaml restructuré (hiérarchie claire, valeurs immuables préservées)

---

### ✅ T4 : Adapter DynamicBehaviorEngine pour modulateur relatif pur
- [ ] 4.1 Modifier `_get_tier_based_parameters()` pour lire `workers.wX.trading_parameters` comme base Optuna
- [ ] 4.2 Adapter `compute_dynamic_modulation()` pour appliquer multiplicateurs relatifs (±X%) au lieu de valeurs absolues
- [ ] 4.3 Vérifier que SL/TP restent constants (ou modulation légère si régime)
- [ ] 4.4 Garder les caps de palier et le respect de min_trade=11
- [ ] 4.5 Tester que DBE ne produit que des multiplicateurs, pas des valeurs écrasantes
- **Livrable** : DynamicBehaviorEngine refactorisé, tests unitaires DBE

---

### ✅ T5 : Centraliser la décision finale dans PortfolioManager
- [ ] 5.1 Créer fonction `calculate_final_trade_parameters(worker_id, capital, market_regime, ...)`
- [ ] 5.2 Implémenter la chaîne séquentielle : Optuna → DBE (modulation) → Contraintes (env + paliers)
- [ ] 5.3 Vérifier que position_final ≤ tier.max_position_size_pct et ≥ 11 USDT
- [ ] 5.4 Remplacer les endroits où la taille de position était recalculée de façon divergente
- [ ] 5.5 Ajouter logging détaillé : quelle couche a modifié quoi
- **Livrable** : PortfolioManager avec fonction centralisée, logs de décision

---

### ✅ T6 : Écrire tests d'intégration de hiérarchie
- [ ] 6.1 Créer `tests/test_decision_hierarchy.py` avec scénarios : capital/palier/régime/worker
- [ ] 6.2 Tester que position_final ∈ exposure_range du palier
- [ ] 6.3 Tester que notional ≥ 11 USDT ou trade rejeté
- [ ] 6.4 Tester que écart Optuna → final ≤ 15% (sauf clamp par palier)
- [ ] 6.5 Tester chaque worker (w1, w2, w3, w4) avec différents capitaux
- **Livrable** : Suite de tests complète, tous les tests passent

---

### ✅ T7 : Exécuter batterie de tests existante et corriger régressions
- [ ] 7.1 Lancer `pytest -q tests/` pour vérifier aucune régression
- [ ] 7.2 Lancer scripts de validation existants (validate_adan_final.py, etc.)
- [ ] 7.3 Corriger les éventuels bugs révélés par les tests
- [ ] 7.4 Vérifier que min_trade=11 est toujours respecté partout
- [ ] 7.5 Valider que paliers et intervalles d'exposition sont inchangés
- **Livrable** : Tous les tests passent, aucune régression

---

### ✅ T8 : Relancer Optuna avec nouvelle hiérarchie
- [ ] 8.1 Préparer scripts Optuna (optimize_hyperparams.py) avec nouvelle config
- [ ] 8.2 Relancer Optuna pour chaque worker (w1, w2, w3, w4)
- [ ] 8.3 Extraire meilleurs hyperparamètres (*_best_params.yaml)
- [ ] 8.4 Vérifier que métriques Optuna sont cohérentes (pas de dégradation)
- [ ] 8.5 Documenter résultats Optuna (scores, paramètres optimisés)
- **Livrable** : Nouveaux *_best_params.yaml pour chaque worker, rapport Optuna

---

### ✅ T9 : Injecter hyperparamètres Optuna dans config.yaml
- [ ] 9.1 Extraire valeurs Optuna pures des *_best_params.yaml
- [ ] 9.2 Injecter dans `config/config.yaml` section `workers.wX.trading_parameters`
- [ ] 9.3 Vérifier que paliers et min_trade restent **INCHANGÉS**
- [ ] 9.4 Valider que hiérarchie est respectée (Optuna → DBE → contraintes)
- [ ] 9.5 Faire un commit avec nouvelle config
- **Livrable** : config/config.yaml avec hyperparamètres Optuna injectés

---

### ✅ T10 : Relancer entraînement final et conclure
- [ ] 10.1 Lancer entraînement parallèle (scripts/train_parallel_agents.py)
- [ ] 10.2 Surveiller logs et dashboard (tensorboard, métriques centralisées)
- [ ] 10.3 Vérifier que hiérarchie est respectée pendant l'entraînement
- [ ] 10.4 Attendre fin de l'entraînement (ne pas interrompre)
- [ ] 10.5 Générer rapport final : hiérarchie validée, Optuna relancé, entraînement terminé
- **Livrable** : Entraînement terminé, rapport final, chantier clos

---

## 📌 Contraintes Immuables (À Respecter Absolument)

| Contrainte | Valeur | Statut |
|-----------|--------|--------|
| `min_order_value_usdt` | 11.0 | 🔒 INCHANGÉ |
| `capital_tiers` (valeurs) | Micro/Small/Medium/High/Enterprise | 🔒 INCHANGÉ |
| `exposure_range` par palier | Ex: Micro 70-90%, Small 35-75% | 🔒 INCHANGÉ |
| `max_position_size_pct` par palier | Ex: Micro 90%, Small 65% | 🔒 INCHANGÉ |
| `risk_per_trade_pct` par palier | Ex: Micro 4%, Small 2% | 🔒 INCHANGÉ |

---

## 🎯 État d'Avancement

| Tâche | Statut | Notes |
|-------|--------|-------|
| T1 | 🔄 EN COURS | Cartographie en cours |
| T2 | ⏳ À FAIRE | Après T1 |
| T3 | ⏳ À FAIRE | Après T2 |
| T4 | ⏳ À FAIRE | Après T3 |
| T5 | ⏳ À FAIRE | Après T4 |
| T6 | ⏳ À FAIRE | Après T5 |
| T7 | ⏳ À FAIRE | Après T6 |
| T8 | ⏳ À FAIRE | Après T7 |
| T9 | ⏳ À FAIRE | Après T8 |
| T10 | ⏳ À FAIRE | Après T9 |

---

## 📝 Notes Méthodologiques

- **Un module à la fois** : Terminer T1 complètement avant T2
- **Tests à chaque étape** : Valider avant de passer au suivant
- **Pas de modification de valeurs immuables** : Paliers, min_trade, intervalles
- **Documentation claire** : Chaque étape doit être compréhensible
- **Utiliser tous les outils** : readFile, strReplace, getDiagnostics, executeBash, etc.
- **Commit réguliers** : Après chaque tâche majeure
- **Ne pas s'arrêter** : Continuer jusqu'à T10 complète

---

## 🚀 Prochaine Étape

**Commencer T1 : Cartographier la hiérarchie réelle actuelle**
- Lire config/config.yaml (sections clés)
- Tracer le flux de décision
- Documenter les points de conflit
- Valider min_trade=11 partout
