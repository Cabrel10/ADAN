# 🚀 GUIDE DES COMMANDES RAPIDES ADAN

**Version :** ADAN v1.2 - Système Opérationnel  
**Date :** 31 Mai 2025  
**Status :** ✅ PRÊT POUR PRODUCTION

---

## 📋 COMMANDES ESSENTIELLES

### 🔧 Activation de l'Environnement
```bash
# Méthode recommandée (compatible avec tous les scripts)
bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate trading_env && [COMMANDE]"

# Méthode directe (si conda est déjà initialisé)
conda activate trading_env
```

### 🧪 Tests de Validation (OBLIGATOIRE avant entraînement)
```bash
# Test OrderManager (doit passer 5/5 tests)
bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate trading_env && python scripts/test_order_manager_only.py"

# Test système complet avec interface moderne
bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate trading_env && python scripts/test_training_improved.py --total_timesteps 1000"
```

### 📊 Vérification des Données
```bash
# Vérifier les données converties (905 MB attendus)
ls -lh data/processed/merged/unified/
# Doit contenir : 1m_train_merged.parquet (632 MB), 1m_val_merged.parquet (211 MB), 1m_test_merged.parquet (106 MB)

# Statistiques des données
python -c "import pandas as pd; df=pd.read_parquet('data/processed/merged/unified/1m_train_merged.parquet'); print(f'Train: {df.shape} | Colonnes: {df.columns[:10].tolist()}')"
```

---

## 🤖 ENTRAÎNEMENT

### 🎯 Entraînement Rapide (Test - 1000 steps)
```bash
bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate trading_env && python scripts/test_training_improved.py --total_timesteps 1000 --initial_capital 15000"
```

### 🏃 Entraînement Court (Validation - 10K steps)
```bash
bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate trading_env && python scripts/train_rl_agent.py --exec_profile cpu --total_timesteps 10000 --initial_capital 15000"
```

### 🚀 Entraînement Production (50K+ steps)
```bash
bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate trading_env && python scripts/train_rl_agent.py --exec_profile cpu --total_timesteps 50000 --initial_capital 15000 --max_episode_steps 2000"
```

### 🎮 Entraînement Long Terme (100K+ steps)
```bash
bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate trading_env && python scripts/train_rl_agent.py --exec_profile cpu --total_timesteps 100000 --initial_capital 15000 --max_episode_steps 5000"
```

---

## 📈 ÉVALUATION ET MONITORING

### 📊 Évaluation d'un Modèle
```bash
bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate trading_env && python scripts/evaluate_performance.py --model_path models/latest_model.zip --exec_profile cpu"
```

### 🔍 Logs d'Entraînement
```bash
# Voir les logs en temps réel
tail -f training_*.log

# Filtrer les métriques importantes
grep -E "(📈 Step|💰 Capital|🎯 Reward)" training_*.log
```

---

## 🛠️ MAINTENANCE ET DIAGNOSTIC

### 🔄 Reconversion des Données (si nécessaire)
```bash
bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate trading_env && python scripts/convert_real_data.py --exec_profile cpu --timeframe 1m"
```

### 🧹 Nettoyage
```bash
# Nettoyer les anciens modèles (garder les 5 plus récents)
ls -lt models/*.zip | tail -n +6 | awk '{print $NF}' | xargs rm -f

# Nettoyer les logs anciens (plus de 7 jours)
find . -name "training_*.log" -mtime +7 -delete
```

---

## ⚡ RACCOURCIS UTILES

### 📋 Status Rapide du Système
```bash
echo "=== STATUS ADAN ==="
echo "📁 Données: $(ls data/processed/merged/unified/*.parquet 2>/dev/null | wc -l) fichiers"
echo "🤖 Modèles: $(ls models/*.zip 2>/dev/null | wc -l) modèles"
echo "📊 Logs: $(ls training_*.log 2>/dev/null | wc -l) logs"
echo "💾 Espace: $(du -sh data/ models/ 2>/dev/null | awk '{s+=$1} END {print s"MB"}')"
```

### 🎯 Test Complet Express (5 minutes)
```bash
bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate trading_env && python scripts/test_order_manager_only.py && python scripts/test_training_improved.py --total_timesteps 500"
```

---

## 🚨 DÉPANNAGE

### ❌ Erreurs Courantes et Solutions

**Erreur :** `ModuleNotFoundError: No module named 'stable_baselines3'`
```bash
pip install stable-baselines3[extra]
```

**Erreur :** `Configuration 'paths' manquante`
```bash
# Vérifier que main_config.yaml existe
ls -la config/main_config.yaml
```

**Erreur :** `Aucune donnée d'entraînement trouvée`
```bash
# Reconvertir les données
bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate trading_env && python scripts/convert_real_data.py --exec_profile cpu"
```

**Entraînement trop verbeux :**
```bash
# Utiliser le script amélioré
python scripts/test_training_improved.py --total_timesteps [NUMBER]
```

### 🔧 Variables d'Environnement (si nécessaire)
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TERM=dumb
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
```

---

## 📊 MÉTRIQUES DE SUCCÈS

### ✅ Tests Validés
- OrderManager : 5/5 tests réussis
- Données : 905 MB chargées (401k train + 114k val + 57k test)
- Features : 235 colonnes (47 × 5 actifs)
- Agent : Compatible SB3+PPO

### 🎯 Objectifs d'Entraînement
- **Test rapide :** Capital stable, pas de crash
- **Validation :** Récompense moyenne > -1.0
- **Production :** Gains > 5% sur données test
- **Long terme :** Sharpe Ratio > 1.0

---

## 🚀 COMMANDE ULTIME (TOUT-EN-UN)

```bash
# Test complet + Entraînement + Évaluation
bash -c "
source ~/miniconda3/etc/profile.d/conda.sh && 
conda activate trading_env && 
echo '🧪 Tests...' && python scripts/test_order_manager_only.py && 
echo '🤖 Entraînement...' && python scripts/train_rl_agent.py --exec_profile cpu --total_timesteps 10000 --initial_capital 15000 && 
echo '📊 Évaluation...' && python scripts/evaluate_performance.py --model_path models/latest_model.zip --exec_profile cpu &&
echo '✅ ADAN Pipeline terminé avec succès!'
"
```

---

**📞 Support :** Consultez `RAPPORT_MODIFICATIONS_FINAL.md` pour les détails techniques  
**🎯 Prêt pour production :** Système 100% opérationnel avec données réelles