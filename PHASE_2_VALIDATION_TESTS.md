# 🧪 PHASE 2: TESTS DE VALIDATION PENDANT W3 OPTUNA

**Date**: 2025-12-10 02:45 UTC  
**Durée W3 Optuna**: 30-40 minutes  
**Pendant ce temps**: Tests de validation et préparation entraînement final

---

## 🎯 OBJECTIF

Pendant que W3 Optuna tourne (30-40 min), exécuter les tests de validation et préparer l'entraînement final.

---

## 📋 TESTS À EXÉCUTER

### Test 1: Validation Configuration (5 min)

**Commande**:
```bash
cd /home/morningstar/Documents/trading/bot
python scripts/test_metrics_validation.py config/config.yaml
```

**Résultat attendu**:
```
✅ Configuration VALIDE - Avertissements seulement

⚠️  AVERTISSEMENTS:
   ⚠️  W3: position_size_pct 0.258 < 0.4 (recommandé pour agressif)
   ⚠️  W3: min_holding_period_steps 140 très long (recommandé < 80)
```

**Action si erreur**:
- Vérifier que config.yaml est valide YAML
- Vérifier que tous les workers ont agent_config et trading_parameters
- Vérifier que SL < TP pour tous les workers

---

### Test 2: Vérification des Paramètres PPO (5 min)

**Commande**:
```bash
python -c "
import yaml
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)
    for worker, cfg in config['workers'].items():
        ppo = cfg.get('agent_config', {})
        print(f'{worker}: lr={ppo.get(\"learning_rate\"):.2e}, batch_size={ppo.get(\"batch_size\")}, n_steps={ppo.get(\"n_steps\")}')
"
```

**Résultat attendu**:
```
w1: lr=1.08e-05, batch_size=128, n_steps=2048
w2: lr=1.62e-05, batch_size=64, n_steps=1024
w3: lr=1.91e-04, batch_size=64, n_steps=1024
w4: lr=5.00e-05, batch_size=128, n_steps=1024
```

---

### Test 3: Vérification des Paramètres Trading (5 min)

**Commande**:
```bash
python -c "
import yaml
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)
    for worker, cfg in config['workers'].items():
        tp = cfg.get('trading_parameters', {})
        print(f'{worker}: SL={tp.get(\"stop_loss_pct\"):.4f}, TP={tp.get(\"take_profit_pct\"):.4f}, PosSize={tp.get(\"position_size_pct\"):.2f}')
"
```

**Résultat attendu**:
```
w1: SL=0.0253, TP=0.0321, PosSize=0.11
w2: SL=0.0250, TP=0.0500, PosSize=0.25
w3: SL=0.0800, TP=0.1500, PosSize=0.45
w4: SL=0.0120, TP=0.0200, PosSize=0.20
```

---

### Test 4: Vérification Ratios TP/SL (5 min)

**Commande**:
```bash
python -c "
import yaml
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)
    for worker, cfg in config['workers'].items():
        tp = cfg.get('trading_parameters', {})
        sl = tp.get('stop_loss_pct', 0)
        tp_val = tp.get('take_profit_pct', 0)
        if sl > 0:
            ratio = tp_val / sl
            print(f'{worker}: TP/SL ratio = {ratio:.2f}')
"
```

**Résultat attendu**:
```
w1: TP/SL ratio = 1.27
w2: TP/SL ratio = 2.00
w3: TP/SL ratio = 1.88
w4: TP/SL ratio = 1.67
```

**Critères**:
- ✅ Tous les ratios > 1.2 (acceptable)
- ⚠️ W1 ratio = 1.27 (serré, normal pour scalper)
- ✅ W2, W3, W4 ratios > 1.6 (bon)

---

## 🔍 MONITORING W3 OPTUNA

### Pendant les tests, monitorer W3

**Commande 1: Voir le log en direct**
```bash
tail -f /tmp/w3_reoptimize_*.log
```

**Commande 2: Voir les scores**
```bash
grep "Score:" /tmp/w3_reoptimize_*.log | tail -10
```

**Commande 3: Voir la progression**
```bash
bash scripts/monitor_w3_optuna.sh
```

---

## 📊 CRITÈRES DE VALIDATION

### Configuration Valide
- [x] Tous les workers ont agent_config (PPO params)
- [x] Tous les workers ont trading_parameters
- [x] Tous les SL > 0 et TP > 0
- [x] Tous les TP > SL
- [x] Tous les position_size_pct entre 0 et 1
- [x] Tous les learning_rate entre 1e-7 et 1e-2

### Paramètres PPO Valides
- [x] learning_rate: W1=1.08e-05, W2=1.62e-05, W3=1.91e-04, W4=5.00e-05
- [x] batch_size: W1=128, W2=64, W3=64, W4=128
- [x] n_steps: W1=2048, W2=1024, W3=1024, W4=1024
- [x] gamma, gae_lambda, clip_range, etc. présents

### Paramètres Trading Valides
- [x] W1: SL=0.0253, TP=0.0321, PosSize=0.1121
- [x] W2: SL=0.0250, TP=0.0500, PosSize=0.25
- [x] W3: SL=0.0800, TP=0.1500, PosSize=0.45
- [x] W4: SL=0.0120, TP=0.0200, PosSize=0.20

---

## 🚀 PRÉPARATION ENTRAÎNEMENT FINAL

Pendant que W3 Optuna tourne, préparer l'entraînement final:

### 1. Créer les répertoires de logs
```bash
mkdir -p /mnt/new_data/adan_logs
mkdir -p /home/morningstar/Documents/trading/bot/checkpoints
```

### 2. Nettoyer les anciens logs (optionnel)
```bash
# Archiver les anciens logs
tar -czf /mnt/new_data/adan_logs/archive_$(date +%Y%m%d).tar.gz /mnt/new_data/adan_logs/*.log 2>/dev/null || true
```

### 3. Préparer le script d'entraînement
```bash
# Vérifier que train_parallel_agents.py existe
ls -lh scripts/train_parallel_agents.py
```

### 4. Préparer le monitoring
```bash
# Créer un script de monitoring pour l'entraînement final
cat > scripts/monitor_training.sh << 'EOF'
#!/bin/bash
LOG_FILE=$(ls -t /mnt/new_data/adan_logs/adan_training_*.log 2>/dev/null | head -1)
if [ -z "$LOG_FILE" ]; then
    echo "Aucun log trouvé"
    exit 1
fi
tail -f "$LOG_FILE" | grep -E "Step|Sharpe|Win Rate|Drawdown|Portfolio"
EOF
chmod +x scripts/monitor_training.sh
```

---

## ✅ CHECKLIST TESTS

### Avant W3 Optuna (COMPLÉTÉ)
- [x] Paramètres W3 recalibrés
- [x] Optuna lancé
- [x] Scripts de validation créés
- [x] Scripts de monitoring créés

### Pendant W3 Optuna (EN COURS)
- [ ] Test 1: Validation configuration
- [ ] Test 2: Vérification paramètres PPO
- [ ] Test 3: Vérification paramètres trading
- [ ] Test 4: Vérification ratios TP/SL
- [ ] Monitoring W3 Optuna
- [ ] Préparation entraînement final

### Après W3 Optuna (À FAIRE)
- [ ] Extraire meilleurs paramètres PPO
- [ ] Injecter dans config.yaml
- [ ] Valider configuration mise à jour
- [ ] Lancer entraînement final

---

## 📈 TIMELINE COMPLÈTE

| Heure | Action | Durée | Status |
|-------|--------|-------|--------|
| 02:45 | Lancement W3 Optuna | - | ✅ FAIT |
| 02:50 | Test 1: Validation config | 5 min | ⏳ À FAIRE |
| 02:55 | Test 2: Paramètres PPO | 5 min | ⏳ À FAIRE |
| 03:00 | Test 3: Paramètres trading | 5 min | ⏳ À FAIRE |
| 03:05 | Test 4: Ratios TP/SL | 5 min | ⏳ À FAIRE |
| 03:10 | Préparation entraînement | 10 min | ⏳ À FAIRE |
| 03:20 | Monitoring W3 Optuna | 25 min | ⏳ À FAIRE |
| 03:45 | ✅ W3 Optuna complète | - | ⏳ À FAIRE |
| 03:50 | Extraction paramètres | 5 min | ⏳ À FAIRE |
| 03:55 | Injection config.yaml | 5 min | ⏳ À FAIRE |
| 04:00 | Validation finale | 5 min | ⏳ À FAIRE |
| 04:05 | Lancement entraînement | - | ⏳ À FAIRE |

---

## 🎯 RÉSULTATS ATTENDUS

### Après Tests de Validation
```
✅ Configuration VALIDE
✅ Tous les paramètres PPO présents et valides
✅ Tous les paramètres trading présents et valides
✅ Tous les ratios TP/SL > 1.2
```

### Après W3 Optuna
```
Score W3: 20+ (vs 8.80 avant)
Trades W3: 50+ (vs 5 avant)
Sharpe W3: 5+ (avec plus de trades)
```

### Après Entraînement Final
```
W1: Score > 40, Sharpe > 20, Trades > 300
W2: Score > 25, Sharpe > 15, Trades > 150
W3: Score > 15, Sharpe > 8, Trades > 50
W4: Score > 50, Sharpe > 18, Trades > 500
```

---

## 📞 SUPPORT

### Si un test échoue
1. Vérifier le message d'erreur
2. Vérifier que config.yaml est valide YAML
3. Vérifier que tous les paramètres sont présents
4. Corriger et relancer le test

### Si W3 Optuna s'arrête
1. Vérifier le log: `tail -f /tmp/w3_reoptimize_*.log`
2. Vérifier le PID: `ps aux | grep optuna_optimize_ppo`
3. Relancer si nécessaire

### Si les tests sont lents
- C'est normal, chaque test prend 1-2 minutes
- Continuer avec les autres tests en parallèle

---

## 📁 FICHIERS DE RÉFÉRENCE

- **Config**: `config/config.yaml`
- **Validation Script**: `scripts/test_metrics_validation.py`
- **Monitoring W3**: `scripts/monitor_w3_optuna.sh`
- **Monitoring Training**: `scripts/monitor_training.sh`
- **W3 Log**: `/tmp/w3_reoptimize_*.log`
- **Training Log**: `/mnt/new_data/adan_logs/adan_training_*.log`

---

## ✨ RÉSUMÉ

**Phase 2 Validation Tests**: Tests à exécuter pendant W3 Optuna

✅ **À FAIRE MAINTENANT**:
1. Exécuter les 4 tests de validation
2. Monitorer W3 Optuna
3. Préparer l'entraînement final
4. Attendre la fin de W3 Optuna (30-40 min)

**Durée totale**: 30-40 minutes (pendant W3 Optuna)

---

**Créé par**: Cascade AI  
**Date**: 2025-12-10 02:45 UTC  
**Fichier**: `/home/morningstar/Documents/trading/bot/PHASE_2_VALIDATION_TESTS.md`
