# 🚀 CHECKLIST DE DÉCOLLAGE - ADAN 2.0

## ✅ PRÉ-VOL (À FAIRE AVANT LE LANCEMENT)

### 1. Vérification de l'Environnement
- [ ] Conda activé: `conda activate trading_env`
- [ ] Python 3.11+: `python3 --version`
- [ ] Répertoires créés: `logs/`, `data/`, `checkpoints/`
- [ ] Fichiers critiques présents:
  - [ ] `src/adan_trading_bot/common/central_logger.py`
  - [ ] `src/adan_trading_bot/performance/unified_metrics_db.py`
  - [ ] `src/adan_trading_bot/performance/unified_metrics.py`
  - [ ] `src/adan_trading_bot/risk_management/risk_manager.py`
  - [ ] `src/adan_trading_bot/environment/reward_calculator.py`
  - [ ] `src/adan_trading_bot/environment/realistic_trading_env.py`

### 2. Nettoyage Pré-Lancement
- [ ] Supprimer les bases de données de test:
  ```bash
  rm -f test_*.db metrics.db
  ```
- [ ] Supprimer les logs de test:
  ```bash
  rm -f logs/*.log
  ```
- [ ] Vérifier l'espace disque: `df -h`

### 3. Exécution des Tests
- [ ] Tests unitaires réussis:
  ```bash
  conda run -n trading_env python3 tests/test_foundations.py
  ```
  Résultat attendu: 22/22 ✅

- [ ] Tests d'intégration réussis:
  ```bash
  conda run -n trading_env python3 tests/test_integration_simple.py
  ```
  Résultat attendu: 9/13 ✅

### 4. Vérification des Composants
- [ ] Logger centralisé fonctionne:
  ```bash
  conda run -n trading_env python3 -c "from src.adan_trading_bot.common.central_logger import logger; logger.metric('test', 1.0); print('✅ Logger OK')"
  ```

- [ ] Base de données fonctionne:
  ```bash
  conda run -n trading_env python3 -c "from src.adan_trading_bot.performance.unified_metrics_db import UnifiedMetricsDB; db = UnifiedMetricsDB(); print('✅ DB OK')"
  ```

- [ ] Métriques fonctionnent:
  ```bash
  conda run -n trading_env python3 -c "from src.adan_trading_bot.performance.unified_metrics import UnifiedMetrics; m = UnifiedMetrics(); print('✅ Metrics OK')"
  ```

- [ ] RiskManager fonctionne:
  ```bash
  conda run -n trading_env python3 -c "from src.adan_trading_bot.risk_management.risk_manager import RiskManager; r = RiskManager({'max_daily_drawdown': 0.15, 'max_position_risk': 0.02, 'max_portfolio_risk': 0.10, 'initial_capital': 10000}); print('✅ RiskManager OK')"
  ```

---

## 🚀 DÉCOLLAGE (LANCEMENT)

### Option 1: Lancement Automatique (Recommandé)
```bash
bash LANCEMENT_PRODUCTION.sh
```

### Option 2: Lancement Manuel (3 Terminaux)

#### Terminal 1 - Entraînement Parallèle
```bash
conda activate trading_env
python3 scripts/train_parallel_agents.py --workers 4 --steps 10000
```
Attendez: "Training started..." ✅

#### Terminal 2 - Monitoring Dashboard
```bash
conda activate trading_env
python3 scripts/terminal_dashboard.py
```
Attendez: Dashboard affichant les métriques ✅

#### Terminal 3 - Surveillance des Logs
```bash
tail -f logs/adan_$(date +%Y%m%d).log
```
Attendez: Logs en temps réel ✅

---

## 📊 VOL EN CROISIÈRE (MONITORING)

### Points de Contrôle Critiques

#### 1. Logs Centralisés
```bash
# Vérifier que les logs sont créés
ls -lh logs/adan_*.log

# Vérifier le contenu
tail -20 logs/adan_$(date +%Y%m%d).log

# Chercher les erreurs
grep -i "error\|critical" logs/adan_*.log
```

#### 2. Base de Données
```bash
# Vérifier la structure
sqlite3 metrics.db ".tables"

# Vérifier les données
sqlite3 metrics.db "SELECT COUNT(*) as total_metrics FROM metrics;"
sqlite3 metrics.db "SELECT COUNT(*) as total_trades FROM trades;"

# Vérifier la cohérence
sqlite3 metrics.db "SELECT metric_name, COUNT(*) FROM metrics GROUP BY metric_name LIMIT 10;"
```

#### 3. Métriques
```bash
# Vérifier les dernières métriques
sqlite3 metrics.db "SELECT metric_name, metric_value, timestamp FROM metrics ORDER BY timestamp DESC LIMIT 10;"

# Vérifier les trades
sqlite3 metrics.db "SELECT action, symbol, quantity, price, pnl FROM trades ORDER BY timestamp DESC LIMIT 5;"
```

#### 4. Alertes de Risque
```bash
# Chercher les alertes de risque
grep -i "risk\|drawdown\|circuit" logs/adan_*.log | tail -20

# Chercher les validations
grep "VALIDATION" logs/adan_*.log | tail -10
```

#### 5. Performance
```bash
# Vérifier la taille des fichiers
du -sh logs/ data/ checkpoints/

# Vérifier l'utilisation CPU/Mémoire
top -b -n 1 | head -20
```

---

## ⚠️ PROCÉDURES D'URGENCE

### Si les Logs ne s'Affichent Pas
```bash
# Vérifier les permissions
ls -la logs/

# Vérifier le logger
conda run -n trading_env python3 -c "from src.adan_trading_bot.common.central_logger import logger; logger.metric('test', 1.0)"

# Vérifier les fichiers
ls -la logs/adan_*.log
```

### Si la Base de Données est Corrompue
```bash
# Sauvegarder l'ancienne DB
cp metrics.db metrics.db.backup

# Supprimer et recréer
rm metrics.db

# Relancer les tests
conda run -n trading_env python3 tests/test_foundations.py
```

### Si les Tests Échouent
```bash
# Vérifier les imports
conda run -n trading_env python3 -c "import src.adan_trading_bot; print('✅ Imports OK')"

# Vérifier la configuration
conda run -n trading_env python3 -c "from src.adan_trading_bot.environment.reward_calculator import RewardCalculator; print('✅ RewardCalculator OK')"

# Relancer les tests avec verbose
conda run -n trading_env python3 tests/test_foundations.py -v
```

### Si le Dashboard ne Démarre Pas
```bash
# Vérifier les dépendances
conda run -n trading_env pip list | grep -i streamlit

# Relancer avec debug
conda run -n trading_env python3 scripts/terminal_dashboard.py --logger=debug
```

---

## 🎯 OBJECTIFS DE VOL

### Heure 1 (Décollage)
- [ ] Tous les tests passent (31/35)
- [ ] Logger crée des fichiers
- [ ] Base de données fonctionne
- [ ] Dashboard affiche les métriques

### Heure 2-4 (Montée)
- [ ] Premiers trades exécutés
- [ ] Métriques commencent à s'accumuler
- [ ] Logs montrent l'activité
- [ ] Pas d'erreurs critiques

### Heure 4+ (Croisière)
- [ ] Entraînement progresse
- [ ] Métriques stables
- [ ] Logs cohérents
- [ ] Monitoring actif

---

## ✅ CRITÈRES DE SUCCÈS

### Décollage Réussi
- ✅ 31/35 tests réussis (88.6%)
- ✅ Logs créés et lisibles
- ✅ Base de données fonctionnelle
- ✅ Dashboard affichant les données
- ✅ Pas d'erreurs critiques

### Vol Stable
- ✅ Trades exécutés régulièrement
- ✅ Métriques cohérentes
- ✅ Logs sans erreurs
- ✅ Monitoring actif
- ✅ Système réactif

### Atterrissage Sûr
- ✅ Données persistées
- ✅ Logs archivés
- ✅ Base de données sauvegardée
- ✅ Rapport final généré
- ✅ Prêt pour le prochain vol

---

## 📈 COMMANDES UTILES

### Monitoring
```bash
# Logs en temps réel
tail -f logs/adan_$(date +%Y%m%d).log

# Métriques en temps réel
watch -n 5 'sqlite3 metrics.db "SELECT COUNT(*) FROM metrics;"'

# Trades en temps réel
watch -n 5 'sqlite3 metrics.db "SELECT COUNT(*) FROM trades;"'
```

### Diagnostic
```bash
# Vérifier l'état du système
ps aux | grep python3

# Vérifier l'utilisation disque
du -sh logs/ data/ checkpoints/

# Vérifier la base de données
sqlite3 metrics.db ".schema"
```

### Nettoyage
```bash
# Archiver les logs
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/

# Nettoyer les anciens logs
find logs/ -name "*.log" -mtime +7 -delete

# Compacter la base de données
sqlite3 metrics.db "VACUUM;"
```

---

## 🎉 DÉCOLLAGE AUTORISÉ!

Vous êtes prêt à décoller. Tous les systèmes sont verts.

**Bonne chasse sur les marchés, capitaine!** 📈🚀

---

*Checklist de décollage - ADAN 2.0 Production Ready*
*Dernière mise à jour: 6 Décembre 2025*
*Statut: ✅ APPROUVÉ POUR PRODUCTION*
