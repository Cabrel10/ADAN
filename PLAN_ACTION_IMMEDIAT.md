# 🎯 PLAN D'ACTION IMMÉDIAT

**Date**: 2025-12-08  
**Priorité**: 🔴 HAUTE  
**Durée estimée**: 2-4 heures

---

## 📊 SITUATION ACTUELLE

| Aspect | État | Problème |
|--------|------|---------|
| **Environnement** | ✅ Stable | Aucun |
| **Config.yaml** | ✅ OK | Hyperparamètres verrouillés (bon) |
| **Force Trade** | ✅ Activé | Aucun |
| **Logging** | ✅ Actif | Métriques persistantes |
| **Entraînement** | ⚠️ Lent | 0.05% progression/heure |
| **PnL** | ❌ Négatif | -$10.20 (W0 seulement) |
| **Rewards** | ❌ Négatif | -0.1961 (W0 seulement) |
| **Workers** | ⚠️ Invisibles | W1, W2, W3 ne loggent pas |

---

## 🔴 PROBLÈMES CRITIQUES À RÉSOUDRE

### 1. **PROGRESSION EXTRÊMEMENT LENTE** (0.05%/h)
**Cause probable**:
- Environnement trop complexe (3 timeframes, 20 features, DBE, etc.)
- Trop de calculs par step
- Pas de parallélisation efficace

**Solution**:
- [ ] Profiler l'environnement pour identifier les goulots
- [ ] Optimiser les calculs les plus lents
- [ ] Vérifier si les 4 workers tournent vraiment en parallèle

### 2. **PnL NÉGATIF** (-$10.20)
**Cause probable**:
- Reward function mal calibrée
- Stop loss trop serré
- Modèle apprend à perdre de l'argent

**Solution**:
- [ ] Analyser les trades en détail (W0)
- [ ] Vérifier les SL/TP appliqués
- [ ] Vérifier la reward function

### 3. **WORKERS INVISIBLES** (W1, W2, W3)
**Cause probable**:
- Logging incomplet
- Workers crashent silencieusement
- Ou ne loggent pas leurs données

**Solution**:
- [ ] Vérifier les logs de chaque worker
- [ ] Ajouter logging explicite pour W1, W2, W3
- [ ] Vérifier si les workers tournent

### 4. **OPTUNA ABSENT**
**Cause**: Pas d'optimisation lancée

**Solution**:
- [ ] Lancer optimisation pour w1, w2, w3, w4
- [ ] Utiliser hyperparamètres optimisés

---

## ✅ PLAN D'ACTION PAR PHASE

### PHASE 1: DIAGNOSTIC APPROFONDI (30 min)

#### 1.1 Vérifier les logs d'entraînement
```bash
# Voir les 100 dernières lignes
tail -100 /mnt/new_data/adan_logs/training_20251208_072851.log

# Chercher les erreurs
grep -i "error\|exception\|nan\|inf" /mnt/new_data/adan_logs/training_20251208_072851.log

# Chercher les données W0
grep "Worker 0" /mnt/new_data/adan_logs/training_20251208_072851.log | tail -20

# Chercher les données W1, W2, W3
grep "Worker [123]" /mnt/new_data/adan_logs/training_20251208_072851.log | tail -10
```

#### 1.2 Analyser les performances W0
```bash
# Extraire les métriques W0
python3 << 'EOF'
import re
from pathlib import Path

log_file = Path("/mnt/new_data/adan_logs/training_20251208_072851.log")
if log_file.exists():
    with open(log_file) as f:
        lines = f.readlines()
    
    # Chercher les lignes avec PnL, Rewards, Portfolio
    for line in lines[-100:]:
        if any(x in line for x in ["PnL", "Reward", "Portfolio", "Worker 0"]):
            print(line.strip())
EOF
```

#### 1.3 Vérifier les processus workers
```bash
# Voir les processus Python actifs
ps aux | grep train_parallel | grep -v grep

# Voir la consommation CPU/RAM
top -b -n 1 | grep python | head -5
```

### PHASE 2: CORRECTIONS CRITIQUES (1-2h)

#### 2.1 Ajouter logging explicite pour W1, W2, W3
**Fichier à modifier**: `scripts/train_parallel_agents.py`

```python
# Ajouter après chaque step:
if step % 100 == 0:
    logger.info(f"[Worker {worker_id}] Step: {step}, "
                f"Portfolio: {portfolio_value:.2f}, "
                f"Reward: {reward:.4f}, "
                f"PnL: {pnl:.2f}")
```

#### 2.2 Profiler l'environnement
**Créer**: `scripts/profile_environment.py`

```python
import cProfile
import pstats
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

# Profile 100 steps
profiler = cProfile.Profile()
profiler.enable()

# ... 100 steps ...

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

#### 2.3 Vérifier les hyperparamètres Optuna
**Vérifier**: `config/config.yaml` pour chaque worker

```bash
python3 << 'EOF'
import yaml
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

# Afficher les hyperparamètres de chaque worker
for worker in ["w1", "w2", "w3", "w4"]:
    if worker in config:
        print(f"\n{worker}:")
        ppo = config[worker].get("agent", {}).get("ppo", {})
        print(f"  learning_rate: {ppo.get('learning_rate', 'N/A')}")
        print(f"  max_grad_norm: {ppo.get('max_grad_norm', 'N/A')}")
        print(f"  clip_range: {ppo.get('clip_range', 'N/A')}")
EOF
```

### PHASE 3: OPTIMISATION OPTUNA (2-4h par worker)

#### 3.1 Lancer optimisation w1
```bash
cd /home/morningstar/Documents/trading/bot
timeout 14400 python scripts/optimize_hyperparams.py --worker w1 2>&1 | tee optuna_w1.log &
```

#### 3.2 Lancer optimisation w2 (après w1)
```bash
timeout 14400 python scripts/optimize_hyperparams.py --worker w2 2>&1 | tee optuna_w2.log &
```

#### 3.3 Lancer optimisation w3 (après w2)
```bash
timeout 14400 python scripts/optimize_hyperparams.py --worker w3 2>&1 | tee optuna_w3.log &
```

#### 3.4 Lancer optimisation w4 (après w3)
```bash
timeout 14400 python scripts/optimize_hyperparams.py --worker w4 2>&1 | tee optuna_w4.log &
```

### PHASE 4: REDÉMARRAGE ENTRAÎNEMENT (30 min)

#### 4.1 Arrêter entraînement actuel
```bash
pkill -f train_parallel_agents.py
```

#### 4.2 Appliquer hyperparamètres optimisés
```bash
python3 << 'EOF'
# Script pour injecter les meilleurs trials Optuna dans config.yaml
# À créer: scripts/inject_optuna_results.py
EOF
```

#### 4.3 Relancer entraînement
```bash
cd /home/morningstar/Documents/trading/bot
timeout 86400 python scripts/train_parallel_agents.py \
  --config-path config/config.yaml \
  --checkpoint-dir checkpoints \
  --resume \
  2>&1 | tee /mnt/new_data/adan_logs/training_$(date +%Y%m%d_%H%M%S).log &
```

---

## 📋 CHECKLIST D'EXÉCUTION

### Avant de commencer
- [ ] Environnement conda `trading_env` activé
- [ ] Logs vérifiés et analysés
- [ ] Processus workers vérifiés
- [ ] Backup de config.yaml créé

### Phase 1: Diagnostic
- [ ] Logs d'entraînement analysés
- [ ] Performances W0 extraites
- [ ] Processus workers vérifiés
- [ ] Goulots d'étranglement identifiés

### Phase 2: Corrections
- [ ] Logging W1, W2, W3 amélioré
- [ ] Environnement profilé
- [ ] Hyperparamètres vérifiés
- [ ] Reward function validée

### Phase 3: Optuna
- [ ] w1 optimisé
- [ ] w2 optimisé
- [ ] w3 optimisé
- [ ] w4 optimisé
- [ ] Résultats injectés dans config.yaml

### Phase 4: Redémarrage
- [ ] Entraînement arrêté proprement
- [ ] Hyperparamètres appliqués
- [ ] Entraînement relancé
- [ ] Logs vérifiés

---

## 🚨 AVERTISSEMENTS

⚠️ **NE PAS**:
- Modifier `palier_tiers.yaml` (données critiques)
- Désactiver `force_trade.enabled`
- Changer l'architecture du modèle sans tests
- Ignorer les erreurs DBE

✅ **À FAIRE**:
- Utiliser `timeout` pour les tests
- Documenter tous les changements
- Sauvegarder les configs avant modification
- Monitorer la progression en temps réel

---

## 📞 SUPPORT

Si vous rencontrez des problèmes:

1. **Vérifier les logs**: 
   ```bash
   tail -100 /mnt/new_data/adan_logs/training_*.log
   ```

2. **Vérifier les processus**:
   ```bash
   ps aux | grep train_parallel
   ```

3. **Vérifier la base de données**:
   ```bash
   ls -lh metrics.db optuna.db
   ```

4. **Redémarrer proprement**:
   ```bash
   pkill -f train_parallel_agents.py
   sleep 5
   # Puis relancer
   ```

---

## 🎯 OBJECTIFS FINAUX

Après exécution du plan:

| Métrique | Cible | Actuel |
|----------|-------|--------|
| **Progression** | > 1%/h | 0.05%/h |
| **PnL** | > 0 | -$10.20 |
| **Rewards** | > 0 | -0.1961 |
| **Workers** | 4/4 loggent | 1/4 loggent |
| **Optuna** | Optimisé | Absent |

---

**Prochaine action**: Exécuter PHASE 1 (Diagnostic Approfondi)

**Temps estimé**: 2-4 heures pour résoudre tous les problèmes
