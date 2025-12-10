# 🚀 GUIDE DE RELANCE OPTUNA - APRÈS FIX ZÉRO METRICS

**Date**: 2025-12-09  
**Status**: ✅ **PRÊT À RELANCER**  
**Durée estimée**: 2-4h par worker (100 trials)  

---

## 📋 CHECKLIST PRÉ-LANCEMENT

- [x] Solution robuste implémentée (`evaluate_ppo_params_robust()`)
- [x] Tests d'intégration validés (60+ trades, sharpe != 0)
- [x] Mini Optuna testé (2 trials avec métriques correctes)
- [x] Limites de fréquence augmentées
- [x] Tous les imports vérifiés

**Status**: ✅ **TOUS LES PRÉREQUIS SATISFAITS**

---

## 🎯 STRATÉGIE DE RELANCE

### Phase 1: Validation Rapide (30 min)
```bash
# Test d'intégration (5 min)
cd /home/morningstar/Documents/trading/bot
source ~/miniconda3/bin/activate trading_env
python scripts/test_optuna_training_only.py

# Résultat attendu: ✅✅✅ TOUS LES TESTS PASSENT
```

### Phase 2: Mini Optuna par Worker (2h)
```bash
# 2 trials par worker pour validation rapide
python optuna_optimize_ppo.py --worker W1 --trials 2 --steps 3000
python optuna_optimize_ppo.py --worker W2 --trials 2 --steps 3000
python optuna_optimize_ppo.py --worker W3 --trials 2 --steps 3000
python optuna_optimize_ppo.py --worker W4 --trials 2 --steps 3000

# Vérifier les résultats
cat optuna_results/W*_ppo_best_params.yaml | grep -E "trades:|sharpe:|score:"
```

**Résultats attendus** :
```yaml
trades: >0 (au moins 50+)
sharpe: >2.0 (au moins 4+)
score: >0.5 (au moins 1.8+)
```

### Phase 3: Optuna Complet (4-8h)
```bash
# 100 trials par worker pour optimisation complète
python optuna_optimize_ppo.py --worker W1 --trials 100 --steps 5000
python optuna_optimize_ppo.py --worker W2 --trials 100 --steps 5000
python optuna_optimize_ppo.py --worker W3 --trials 100 --steps 5000
python optuna_optimize_ppo.py --worker W4 --trials 100 --steps 5000

# Monitoring en temps réel
tail -f /mnt/new_data/adan_logs/optuna_*.log
```

---

## 📊 MÉTRIQUES ATTENDUES

### Par Worker (Basé sur Mini Optuna)

| Worker | Sharpe | Trades | Max DD | Win Rate | Score |
|--------|--------|--------|--------|----------|-------|
| W1 | 4.4+ | 85+ | <15% | 42%+ | 1.8+ |
| W2 | 9.4+ | 123+ | <20% | 48%+ | 4.2+ |
| W3 | 5.0+ | 70+ | <18% | 45%+ | 2.0+ |
| W4 | 6.0+ | 100+ | <22% | 46%+ | 2.5+ |

**Note** : Ces valeurs sont des estimations basées sur les tests. Les valeurs réelles peuvent varier.

---

## 🔧 COMMANDES DÉTAILLÉES

### Test d'Intégration
```bash
cd /home/morningstar/Documents/trading/bot
source ~/miniconda3/bin/activate trading_env

# Lancer le test
python scripts/test_optuna_training_only.py

# Résultat attendu:
# ✅ metrics.trades > 0
# ✅ metrics.closed_positions > 0
# ✅ total_trades (returned) > 0
# ✅ sharpe_ratio != 0
# ✅✅✅ TOUS LES TESTS PASSENT
```

### Mini Optuna (Validation)
```bash
# W1 (Ultra-Stable 4h)
python optuna_optimize_ppo.py --worker W1 --trials 2 --steps 3000

# W2 (Moderate 1h)
python optuna_optimize_ppo.py --worker W2 --trials 2 --steps 3000

# W3 (Aggressive 5m)
python optuna_optimize_ppo.py --worker W3 --trials 2 --steps 3000

# W4 (Sharpe Optimized)
python optuna_optimize_ppo.py --worker W4 --trials 2 --steps 3000
```

### Optuna Complet (Production)
```bash
# Lancer en parallèle (4 workers simultanément)
python optuna_optimize_ppo.py --worker W1 --trials 100 --steps 5000 &
python optuna_optimize_ppo.py --worker W2 --trials 100 --steps 5000 &
python optuna_optimize_ppo.py --worker W3 --trials 100 --steps 5000 &
python optuna_optimize_ppo.py --worker W4 --trials 100 --steps 5000 &

# Ou séquentiellement (plus stable)
for worker in W1 W2 W3 W4; do
    echo "Lancement Optuna pour $worker..."
    python optuna_optimize_ppo.py --worker $worker --trials 100 --steps 5000
    echo "$worker terminé"
done
```

### Vérifier les Résultats
```bash
# Afficher les meilleurs scores
for worker in W1 W2 W3 W4; do
    echo "=== $worker ==="
    cat optuna_results/${worker}_ppo_best_params.yaml | grep -E "score:|sharpe:|trades:|drawdown:|win_rate:"
done

# Afficher les YAML complets
cat optuna_results/W*_ppo_best_params.yaml

# Vérifier les logs
tail -100 /mnt/new_data/adan_logs/optuna_*.log
```

---

## 📈 MONITORING EN TEMPS RÉEL

### Logs Optuna
```bash
# Suivre les logs en temps réel
tail -f /mnt/new_data/adan_logs/optuna_W1.log
tail -f /mnt/new_data/adan_logs/optuna_W2.log
tail -f /mnt/new_data/adan_logs/optuna_W3.log
tail -f /mnt/new_data/adan_logs/optuna_W4.log

# Ou tous les logs à la fois
tail -f /mnt/new_data/adan_logs/optuna_*.log
```

### Vérifier les Trials
```bash
# Afficher les trials en cours
python << 'EOF'
import optuna
import sqlite3

for worker in ['W1', 'W2', 'W3', 'W4']:
    db_path = f"optuna_results/{worker}_ppo_*.db"
    print(f"\n=== {worker} ===")
    # Afficher les meilleurs trials
    # (Implémentation dépend de la structure de la DB)
EOF
```

---

## ⚠️ POINTS D'ATTENTION

### 1. Limites de Ressources
- **CPU**: 4 cores (1 par worker)
- **RAM**: 8 GB (2 GB par worker)
- **Disque**: 50 GB libre pour logs et checkpoints

**Vérifier avant de lancer** :
```bash
free -h  # RAM disponible
df -h    # Espace disque
nproc    # Nombre de cores
```

### 2. Gestion des Erreurs
Si un trial échoue :
```bash
# Vérifier les logs
tail -100 /mnt/new_data/adan_logs/optuna_W*.log | grep ERROR

# Relancer le worker
python optuna_optimize_ppo.py --worker W2 --trials 100 --steps 5000
```

### 3. Arrêt Gracieux
```bash
# Arrêter les trials en cours
pkill -f "optuna_optimize_ppo.py"

# Ou avec signal SIGTERM
kill -TERM $(pgrep -f "optuna_optimize_ppo.py")
```

---

## 📊 RÉSULTATS ATTENDUS

### Après Mini Optuna
```
✅ Tous les workers ont des métriques non nulles
✅ Sharpe ratio > 2.0 pour tous les workers
✅ Trades > 50 pour tous les workers
✅ YAML générés avec paramètres PPO valides
```

### Après Optuna Complet
```
✅ Convergence visible dans les trials
✅ Meilleur trial pour chaque worker identifié
✅ Sharpe ratio > 3.0 pour au moins 2 workers
✅ Paramètres PPO optimisés et sauvegardés
```

---

## 🔄 WORKFLOW COMPLET

```
1. Test d'intégration (5 min)
   ↓
2. Mini Optuna par worker (2h)
   ↓
3. Vérifier les résultats
   ↓
4. Optuna complet (4-8h)
   ↓
5. Extraire les meilleurs paramètres
   ↓
6. Injecter dans config.yaml
   ↓
7. Lancer entraînement final
```

---

## 📝 NOTES IMPORTANTES

### Hyperparamètres Optuna
- **Learning Rate**: [1e-5, 5e-4] log scale
- **N Steps**: [512, 2048] (puissance de 2)
- **Batch Size**: [32, 128]
- **N Epochs**: [5, 15]
- **Gamma**: [0.95, 0.99]
- **GAE Lambda**: [0.9, 0.99]
- **Clip Range**: [0.1, 0.3]
- **Ent Coef**: [0.001, 0.1] log scale

### Critères de Pruning
- **Sharpe Ratio**: > -10.0
- **Max Drawdown**: < 1.0 (100%)
- **Win Rate**: > 0.0

---

## ✅ CHECKLIST POST-LANCEMENT

- [ ] Test d'intégration réussi
- [ ] Mini Optuna validé (2 trials par worker)
- [ ] Résultats conformes aux attentes
- [ ] YAML générés et vérifiés
- [ ] Optuna complet lancé
- [ ] Monitoring en cours
- [ ] Logs vérifiés régulièrement
- [ ] Meilleurs paramètres extraits
- [ ] Config.yaml mis à jour
- [ ] Entraînement final lancé

---

## 🚀 COMMANDE RAPIDE (COPY-PASTE)

```bash
cd /home/morningstar/Documents/trading/bot
source ~/miniconda3/bin/activate trading_env

# Phase 1: Validation
echo "=== PHASE 1: Test d'intégration ==="
python scripts/test_optuna_training_only.py

# Phase 2: Mini Optuna
echo "=== PHASE 2: Mini Optuna ==="
for worker in W1 W2 W3 W4; do
    echo "Lancement $worker..."
    python optuna_optimize_ppo.py --worker $worker --trials 2 --steps 3000
done

# Phase 3: Vérifier les résultats
echo "=== PHASE 3: Résultats ==="
for worker in W1 W2 W3 W4; do
    echo "=== $worker ==="
    cat optuna_results/${worker}_ppo_best_params.yaml | grep -E "score:|sharpe:|trades:"
done

echo "✅ Prêt pour Optuna complet!"
```

---

**Status**: ✅ **PRÊT À RELANCER OPTUNA**
