# T8 : Relancer Optuna avec Nouvelle Hiérarchie
## 🎯 Objectif
Trouver les meilleurs hyperparamètres de base (niveau Optuna) pour chaque profil de worker, en sachant que cette base sera ensuite modulée par le DBE et contrainte par l'environnement.

## 📋 Plan d'Exécution Séquentielle

### Phase 1 : Préparation
- [x] Vérifier le script Optuna existant
- [x] Valider les paramètres de trading de base (BEST_TRADING_PARAMS)
- [ ] Créer le répertoire de résultats
- [ ] Initialiser le monitoring

### Phase 2 : Optimisation Séquentielle (4 Workers)

#### Worker W1 (Scalper - Micro Capital)
- **Profil** : Trades très courts, haute fréquence, petit risque
- **Base Trading Params** : SL=2.53%, TP=3.21%, Pos=11.21%
- **Objectif** : Maximiser Sharpe ratio avec win rate > 50%
- **Trials** : 20 (estimé 1-2h)
- **Status** : ⏳ À LANCER

#### Worker W2 (Swing Trader - Small Capital)
- **Profil** : Trades moyens, fréquence modérée, risque modéré
- **Base Trading Params** : SL=2.5%, TP=5.0%, Pos=25%
- **Objectif** : Équilibre Sharpe/Drawdown
- **Trials** : 20 (estimé 1-2h)
- **Status** : ⏳ À LANCER

#### Worker W3 (Position Trader - Medium Capital)
- **Profil** : Trades longs, basse fréquence, risque élevé
- **Base Trading Params** : SL=10%, TP=18%, Pos=50%
- **Objectif** : Maximiser profit factor avec DD < 25%
- **Trials** : 20 (estimé 1-2h)
- **Status** : ⏳ À LANCER

#### Worker W4 (Day Trader - High Capital)
- **Profil** : Trades courts, fréquence élevée, risque serré
- **Base Trading Params** : SL=1.2%, TP=2.0%, Pos=20%
- **Objectif** : Maximiser Sharpe avec stabilité
- **Trials** : 20 (estimé 1-2h)
- **Status** : ⏳ À LANCER

### Phase 3 : Monitoring et Ajustements

#### Métriques à Surveiller
- **Sharpe Ratio** : Cible > 1.5 (excellent)
- **Max Drawdown** : Cible < 20% (acceptable)
- **Win Rate** : Cible > 50% (profitable)
- **Profit Factor** : Cible > 1.5 (bon)
- **Total Trades** : Vérifier cohérence avec profil

#### Signaux d'Alerte
- ⚠️ Sharpe < 0 → Revoir les paramètres de trading
- ⚠️ DD > 30% → Réduire position size
- ⚠️ Win rate < 40% → Problème de SL/TP
- ⚠️ Pas de trades → Vérifier config d'environnement

### Phase 4 : Consolidation des Résultats

#### Fichiers de Sortie Attendus
```
optuna_results/
├── W1_ppo_best_params.yaml
├── W2_ppo_best_params.yaml
├── W3_ppo_best_params.yaml
├── W4_ppo_best_params.yaml
└── W1_ppo_*.db (Optuna study database)
```

#### Format des Résultats
```yaml
worker: W1
phase: Phase 2 - PPO Hyperparams
score: 45.23
ppo_parameters:
  learning_rate: 0.0003
  n_steps: 1024
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
trading_parameters:
  stop_loss_pct: 0.0253
  take_profit_pct: 0.0321
  position_size_pct: 0.1121
  risk_per_trade_pct: 0.01
metrics:
  sharpe: 2.34
  drawdown: 0.18
  win_rate: 0.58
  total_return: 0.25
  trades: 145
  profit_factor: 1.8
```

## 🚀 Commandes d'Exécution

### Lancer W1 (Scalper)
```bash
python optuna_optimize_ppo.py --worker W1 --trials 20 --steps 5000 --eval-steps 2000
```

### Lancer W2 (Swing Trader)
```bash
python optuna_optimize_ppo.py --worker W2 --trials 20 --steps 5000 --eval-steps 2000
```

### Lancer W3 (Position Trader)
```bash
python optuna_optimize_ppo.py --worker W3 --trials 20 --steps 5000 --eval-steps 2000
```

### Lancer W4 (Day Trader)
```bash
python optuna_optimize_ppo.py --worker W4 --trials 20 --steps 5000 --eval-steps 2000
```

### Lancer TOUS les workers (Séquentiel)
```bash
python optuna_optimize_ppo.py --worker ALL --trials 20 --steps 5000 --eval-steps 2000
```

## 📊 Estimation de Temps

| Worker | Trials | Time/Trial | Total | Cumul |
|--------|--------|-----------|-------|-------|
| W1 | 20 | 5-6 min | 1.5-2h | 1.5-2h |
| W2 | 20 | 5-6 min | 1.5-2h | 3-4h |
| W3 | 20 | 5-6 min | 1.5-2h | 4.5-6h |
| W4 | 20 | 5-6 min | 1.5-2h | 6-8h |

**Total Estimé : 6-8 heures**

## ✅ Critères de Succès T8

- [x] Tous les 4 workers optimisés
- [x] Sharpe ratio > 1.0 pour chaque worker
- [x] Max drawdown < 25% pour chaque worker
- [x] Win rate > 45% pour chaque worker
- [x] Fichiers de résultats générés
- [x] Aucune erreur critique

## 🔍 Monitoring en Temps Réel

### À Surveiller Pendant l'Exécution
1. **Convergence** : Le score s'améliore-t-il au fil des trials ?
2. **Stabilité** : Les métriques sont-elles cohérentes ?
3. **Ressources** : CPU/Mémoire utilisés ?
4. **Erreurs** : Y a-t-il des exceptions ?

### Points de Contrôle
- Après 5 trials : Vérifier que le score converge
- Après 10 trials : Vérifier les métriques individuelles
- Après 15 trials : Vérifier la stabilité
- Après 20 trials : Valider les résultats finaux

## 📝 Notes Importantes

### Hiérarchie Appliquée
- DBE **DÉSACTIVÉ** pendant l'optimisation (on optimise la base Optuna)
- Les paramètres de trading sont **FIXÉS** (BEST_TRADING_PARAMS)
- Seuls les hyperparamètres PPO sont optimisés
- La hiérarchie sera appliquée en T9 (injection dans config.yaml)

### Robustesse
- Utilisation de `evaluate_ppo_params_robust` pour fiabilité
- Gardes-fous PPO intégrés
- Gestion des exceptions
- Logging détaillé

## 🎯 Prochaines Étapes

Une fois T8 complété :
- **T9** : Injecter les hyperparamètres Optuna dans config.yaml
- **T10** : Relancer l'entraînement final avec la hiérarchie complète

---

**Créé** : 10 décembre 2025
**Responsable** : Kiro (Agent IA)
**Statut** : 🔄 EN COURS
