# 🚀 GUIDE DE REDÉMARRAGE ET VÉRIFICATION

## Étape 1 : Arrêter le Système Actuel

```bash
# Arrêter tous les processus ADAN
pkill -9 -f paper_trading_monitor.py
pkill -9 -f python.*adan
sleep 2

# Vérifier que tout est arrêté
ps aux | grep -i adan
ps aux | grep -i paper_trading
```

---

## Étape 2 : Vérifier les Corrections

### Vérification du Code

```bash
# Vérifier que les méthodes helper existent
grep -n "_get_current_tier\|_get_max_concurrent_positions\|_detect_market_regime\|_get_dbe_multipliers" scripts/paper_trading_monitor.py

# Vérifier que les features sont ajoutées
grep -n "portfolio_obs\[8\]\|portfolio_obs\[9\]" scripts/paper_trading_monitor.py

# Vérifier que le DBE est appliqué
grep -n "DBE ACTIVÉ\|dbe_multipliers" scripts/paper_trading_monitor.py

# Vérifier que le blocage hiérarchique existe
grep -n "TRANSFORMATION HIÉRARCHIQUE\|BLOCAGE HIÉRARCHIQUE" scripts/paper_trading_monitor.py
```

### Exécuter les Tests

```bash
# Exécuter le script de test
python scripts/test_hierarchy_corrections.py

# Résultats attendus :
# ✅ TEST 1: Méthode _get_max_concurrent_positions
# ✅ TEST 2: Features num_positions et max_positions
# ✅ TEST 3: Méthodes DBE
```

---

## Étape 3 : Redémarrer le Système

### Option A : Redémarrage Normal

```bash
# Redémarrer le monitor
nohup python scripts/paper_trading_monitor.py > monitor_hierarchy_fixed.log 2>&1 &

# Attendre le démarrage
sleep 30

# Vérifier que le processus est actif
ps aux | grep paper_trading_monitor.py | grep -v grep
```

### Option B : Redémarrage avec Logs Détaillés

```bash
# Redémarrer avec logs en temps réel
python scripts/paper_trading_monitor.py 2>&1 | tee monitor_hierarchy_fixed.log
```

---

## Étape 4 : Vérifier les Logs

### Vérification des Features

```bash
# Chercher les logs de features
tail -100 monitor_hierarchy_fixed.log | grep -A2 "HIÉRARCHIE:"

# Résultat attendu :
# 🔥 HIÉRARCHIE: num_positions=0, max_positions=1
```

### Vérification du DBE

```bash
# Chercher les logs du DBE
tail -100 monitor_hierarchy_fixed.log | grep -A5 "DBE ACTIVÉ"

# Résultat attendu :
# 🌐 DBE ACTIVÉ: Régime BULL, Tier Micro Capital
#    - SL multiplier: 1.30
#    - TP multiplier: 1.60
#    - SL ajusté: 2.60% (base: 2.0%)
#    - TP ajusté: 4.80% (base: 3.0%)
```

### Vérification du Blocage Hiérarchique

```bash
# Chercher les logs de blocage
tail -100 monitor_hierarchy_fixed.log | grep -A2 "BLOCAGE HIÉRARCHIQUE\|TRANSFORMATION HIÉRARCHIQUE"

# Résultat attendu (si position ouverte) :
# 🚫 BLOCAGE HIÉRARCHIQUE: 1/1 positions atteint
#    → Tous les votes BUY seront transformés en HOLD
```

---

## Étape 5 : Vérification Complète du Système

### Checklist de Vérification

```bash
#!/bin/bash

echo "🔍 VÉRIFICATION COMPLÈTE DU SYSTÈME"
echo "===================================="

# 1. Vérifier que le processus est actif
echo ""
echo "1️⃣  Processus actif ?"
if pgrep -f "paper_trading_monitor.py" > /dev/null; then
    echo "✅ Oui"
else
    echo "❌ Non - Redémarrer le système"
    exit 1
fi

# 2. Vérifier les features dans les logs
echo ""
echo "2️⃣  Features [8] et [9] présentes ?"
if grep -q "HIÉRARCHIE: num_positions" monitor_hierarchy_fixed.log; then
    echo "✅ Oui"
else
    echo "❌ Non - Vérifier les logs"
fi

# 3. Vérifier le DBE
echo ""
echo "3️⃣  DBE activé ?"
if grep -q "DBE ACTIVÉ" monitor_hierarchy_fixed.log; then
    echo "✅ Oui"
else
    echo "❌ Non - Vérifier les logs"
fi

# 4. Vérifier le blocage hiérarchique
echo ""
echo "4️⃣  Blocage hiérarchique fonctionnel ?"
if grep -q "BLOCAGE HIÉRARCHIQUE\|TRANSFORMATION HIÉRARCHIQUE" monitor_hierarchy_fixed.log; then
    echo "✅ Oui (position ouverte détectée)"
else
    echo "⚠️  Pas encore activé (aucune position ouverte)"
fi

# 5. Vérifier les erreurs
echo ""
echo "5️⃣  Erreurs critiques ?"
ERROR_COUNT=$(grep -c "❌ Error\|❌ CRITICAL" monitor_hierarchy_fixed.log)
if [ "$ERROR_COUNT" -eq 0 ]; then
    echo "✅ Non"
else
    echo "❌ Oui ($ERROR_COUNT erreurs)"
    grep "❌ Error\|❌ CRITICAL" monitor_hierarchy_fixed.log | head -5
fi

echo ""
echo "===================================="
echo "✅ VÉRIFICATION COMPLÈTE"
```

---

## Étape 6 : Tester les Corrections en Action

### Test 1 : Vérifier que les Features sont Présentes

```bash
# Chercher une observation complète
tail -200 monitor_hierarchy_fixed.log | grep -A10 "DEBUG OBSERVATION"

# Résultat attendu :
# 🔍 [DEBUG OBSERVATION]
#    Observation keys: ['5m', '1h', '4h', 'portfolio_state']
#    Portfolio observation shape: (20,)
#    Portfolio values (first 5): [0.29 0.29 0.88 0.0 0.0]
#    → has_position: 0.0000 (0=non, 1=oui) ⭐
#    → position_count: 0.0000
```

### Test 2 : Vérifier que le DBE est Appliqué

```bash
# Chercher un trade exécuté
tail -200 monitor_hierarchy_fixed.log | grep -B5 -A10 "Trade Exécuté"

# Résultat attendu :
# 🌐 DBE ACTIVÉ: Régime BULL, Tier Micro Capital
#    - SL multiplier: 1.30
#    - TP multiplier: 1.60
#    - SL ajusté: 2.60% (base: 2.0%)
#    - TP ajusté: 4.80% (base: 3.0%)
# 🟢 Trade Exécuté: BUY @ 88073.27
#    TP: 92200.00 (4.8%)
#    SL: 85700.00 (2.6%)
```

### Test 3 : Vérifier le Blocage Hiérarchique

```bash
# Chercher un blocage
tail -200 monitor_hierarchy_fixed.log | grep -B5 -A5 "TRANSFORMATION HIÉRARCHIQUE"

# Résultat attendu :
# 🚫 BLOCAGE HIÉRARCHIQUE: 1/1 positions atteint
#    → Tous les votes BUY seront transformés en HOLD
# 🚫 TRANSFORMATION HIÉRARCHIQUE: BUY → HOLD (1/1 positions)
# DÉCISION FINALE: HOLD (conf=0.10)
```

---

## Étape 7 : Monitoring Continu

### Commandes de Monitoring

```bash
# Afficher les logs en temps réel
tail -f monitor_hierarchy_fixed.log

# Afficher seulement les logs critiques
tail -f monitor_hierarchy_fixed.log | grep -E "🔥|🚫|❌|✅"

# Compter les événements
echo "Nombre de features détectées:"
grep -c "HIÉRARCHIE:" monitor_hierarchy_fixed.log

echo "Nombre de DBE activés:"
grep -c "DBE ACTIVÉ" monitor_hierarchy_fixed.log

echo "Nombre de blocages hiérarchiques:"
grep -c "BLOCAGE HIÉRARCHIQUE\|TRANSFORMATION HIÉRARCHIQUE" monitor_hierarchy_fixed.log
```

---

## Étape 8 : Dépannage

### Problème : Features [8] et [9] manquantes

```bash
# Vérifier que build_observation a été modifié
grep -A5 "portfolio_obs\[8\]" scripts/paper_trading_monitor.py

# Si absent, vérifier que la modification a été appliquée
# Relancer le système
```

### Problème : DBE non appliqué

```bash
# Vérifier que execute_trade a été modifié
grep -A10 "DBE ACTIVÉ" scripts/paper_trading_monitor.py

# Vérifier que config.yaml existe
ls -la config/config.yaml

# Vérifier que le DBE est configuré
grep -A20 "dbe:" config/config.yaml
```

### Problème : Blocage hiérarchique non activé

```bash
# Vérifier que get_ensemble_action a été modifié
grep -A5 "TRANSFORMATION HIÉRARCHIQUE" scripts/paper_trading_monitor.py

# Vérifier que num_positions et max_positions sont extraits
grep -B5 "num_positions >= max_positions" scripts/paper_trading_monitor.py
```

---

## Résumé des Vérifications

| Vérification | Commande | Résultat Attendu |
|--------------|----------|------------------|
| Processus actif | `pgrep -f paper_trading_monitor.py` | PID affiché |
| Features présentes | `grep "HIÉRARCHIE:" monitor_hierarchy_fixed.log` | `num_positions=X, max_positions=Y` |
| DBE activé | `grep "DBE ACTIVÉ" monitor_hierarchy_fixed.log` | `Régime BULL/BEAR/SIDEWAYS` |
| SL/TP ajustés | `grep "SL ajusté" monitor_hierarchy_fixed.log` | `2.6%` (au lieu de 2.0%) |
| Blocage actif | `grep "TRANSFORMATION HIÉRARCHIQUE" monitor_hierarchy_fixed.log` | `BUY → HOLD` |

---

## Prochaines Étapes

1. ✅ Redémarrer le système
2. ✅ Vérifier les logs
3. ✅ Confirmer que les 3 corrections fonctionnent
4. ✅ Monitorer les trades pour vérifier la cohérence
5. ✅ Documenter les résultats

---

**Date :** 2024-12-20  
**Statut :** 🚀 PRÊT À REDÉMARRER  
**Impact :** CRITIQUE - Rétablit la hiérarchie ADAN
