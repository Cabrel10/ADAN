# 🚨 DIAGNOSTIC ADAN - FINDINGS CRITIQUES

## 📊 RÉSUMÉ EXÉCUTIF

**Status Global:** ⚠️ **NEEDS_ATTENTION**

Le diagnostic a révélé **3 problèmes critiques** qui doivent être adressés immédiatement :

## 🔴 PROBLÈMES IDENTIFIÉS

### 1. ⚠️ DISQUE PRESQUE PLEIN (98%)

**Sévérité:** CRITIQUE

**Symptômes:**
- Disque à 98% de capacité
- Risque de crash du système
- Impossible de sauvegarder les logs/checkpoints

**Impact:**
- Les modèles ne peuvent pas sauvegarder les checkpoints
- Les logs peuvent être tronqués
- Le système peut devenir instable

**Actions Immédiates:**
```bash
# 1. Vérifier l'espace disque
df -h

# 2. Nettoyer les fichiers temporaires
rm -rf /tmp/*
rm -rf ~/.cache/*

# 3. Supprimer les anciens logs
find . -name "*.log" -mtime +7 -delete

# 4. Supprimer les anciens checkpoints
find ./checkpoints -name "*.zip" -mtime +14 -delete

# 5. Vérifier l'espace après nettoyage
df -h
```

**Cible:** Ramener à <80% d'utilisation

---

### 2. ⚠️ AUCUN TRADE DÉTECTÉ (0 trades ouverts/fermés)

**Sévérité:** CRITIQUE

**Symptômes:**
- 0 trades ouverts
- 0 trades fermés
- 0 trades avec TP/SL

**Causes Possibles:**
1. Le monitor n'est pas en cours d'exécution
2. Les signaux sont tous HOLD (pas de BUY/SELL)
3. Les positions ne s'ouvrent pas correctement
4. Les logs ne sont pas générés

**Diagnostic:**
```bash
# 1. Vérifier si le monitor est en cours d'exécution
ps aux | grep paper_trading_monitor

# 2. Vérifier les logs
tail -f paper_trading.log

# 3. Vérifier les signaux
grep "Ensemble:" paper_trading.log | tail -20

# 4. Vérifier les trades
grep "Trade Exécuté" paper_trading.log | wc -l
```

**Actions:**
1. Redémarrer le monitor
2. Vérifier que les signaux ne sont pas tous HOLD
3. Vérifier que execute_trade() est appelé

---

### 3. ⚠️ INDICATEURS NON DÉTECTÉS (0 indicateurs)

**Sévérité:** CRITIQUE

**Symptômes:**
- 0 indicateurs détectés dans les logs
- 0 timeframes détectés
- Observation shape invalide

**Causes Possibles:**
1. Les indicateurs ne sont pas calculés
2. Les logs ne contiennent pas les informations d'indicateurs
3. La transmission des indicateurs au CNN/PPO échoue

**Diagnostic:**
```bash
# 1. Vérifier les indicateurs dans les logs
grep -i "rsi\|macd\|bb\|atr\|adx" paper_trading.log | head -20

# 2. Vérifier la forme de l'observation
grep "Built observation" paper_trading.log | tail -5

# 3. Vérifier les timeframes
grep "5m\|1h\|4h" paper_trading.log | head -20
```

**Actions:**
1. Vérifier que FeatureEngineer calcule les indicateurs
2. Vérifier que build_observation() reçoit les données correctes
3. Vérifier que les indicateurs sont transmis au CNN/PPO

---

## ✅ POINTS POSITIFS

### 1. ✅ Gestion des Positions: OK
- Pas d'erreur dans la gestion des positions
- Structure de tracking est en place

### 2. ✅ Confusion CNN/PPO: OK
- Variance des signaux: 0.22 (acceptable)
- Pas de confusion détectée entre les modèles

---

## 🔧 PLAN D'ACTION PRIORITAIRE

### Phase 1: Nettoyage Disque (5 minutes)
```bash
# Libérer de l'espace
df -h
du -sh * | sort -rh | head -10
rm -rf /tmp/*
find . -name "*.log" -mtime +7 -delete
df -h
```

### Phase 2: Vérifier le Monitor (10 minutes)
```bash
# Arrêter et redémarrer
pkill -f paper_trading_monitor.py
sleep 2
python scripts/paper_trading_monitor.py --api_key "..." --api_secret "..." &
sleep 5
tail -f paper_trading.log
```

### Phase 3: Vérifier les Indicateurs (15 minutes)
```bash
# Vérifier que les indicateurs sont calculés
grep "Built observation" paper_trading.log | tail -10
grep "rsi\|macd\|bb" paper_trading.log | head -20
```

### Phase 4: Vérifier les Trades (10 minutes)
```bash
# Vérifier que les trades s'ouvrent
grep "Trade Exécuté" paper_trading.log | tail -10
grep "Position fermée" paper_trading.log | tail -10
```

---

## 📋 CHECKLIST DE VÉRIFICATION

- [ ] Disque < 80% d'utilisation
- [ ] Monitor en cours d'exécution
- [ ] Trades ouverts > 0
- [ ] Trades fermés > 0
- [ ] Indicateurs détectés > 0
- [ ] Timeframes détectés = 3 (5m, 1h, 4h)
- [ ] Observation shape = (20, 14)
- [ ] Variance des signaux > 0.5
- [ ] Pas d'erreur dans les logs

---

## 🚀 PROCHAINES ÉTAPES

1. **Immédiat:** Nettoyer le disque
2. **Court terme:** Redémarrer le monitor et vérifier les trades
3. **Moyen terme:** Vérifier la transmission des indicateurs
4. **Long terme:** Optimiser l'utilisation disque

---

## 📞 SUPPORT

Si les problèmes persistent après ces actions:

1. Vérifier les logs détaillés: `tail -f paper_trading.log`
2. Exécuter le diagnostic à nouveau: `python scripts/diagnostic_adan_health.py`
3. Vérifier les configurations: `cat config/config.yaml`

---

**Généré:** 2025-12-13
**Status:** NEEDS_IMMEDIATE_ATTENTION
