# 🚀 COMMANDES RAPIDES ADAN v2.1

**Version:** 2.1 - Apprentissage Continu & Exchange Integration  
**Date:** Juin 2025  
**Mise à jour:** Toutes les nouvelles fonctionnalités incluses

---

## ⚡ DÉMARRAGE RAPIDE

### Configuration de Base
```bash
# Activation environnement
conda activate trading_env
cd ~/Desktop/ADAN/ADAN

# Variables exchange (OBLIGATOIRE pour paper trading)
export BINANCE_TESTNET_API_KEY="votre_cle_api_testnet"
export BINANCE_TESTNET_SECRET_KEY="votre_cle_secrete_testnet"
```

### Tests Système Rapides
```bash
# Status système complet
python status_adan.py

# Test système avec exchange
python test_complete_system.py --exec_profile cpu

# Test connexion exchange uniquement
python test_exchange_connector.py
```

---

## 🔧 PIPELINE DE DONNÉES

### Traitement Multi-Timeframe
```bash
# Pipeline complet (1m, 1h, 1d)
python scripts/convert_real_data.py --exec_profile cpu

# Fusion données par timeframe
python scripts/merge_processed_data.py --exec_profile cpu --timeframes 1m --splits train val test --training-timeframe 1m
```

### Validation Données
```bash
# Vérifier structure
ls -la data/processed/unified/
ls -la data/scalers_encoders/

# Taille des fichiers
du -sh data/processed/unified/*
du -sh data/scalers_encoders/*
```

---

## 🤖 ENTRAÎNEMENT TRADITIONNEL

### Tests Rapides
```bash
# Test 5 minutes (validation)
python scripts/train_rl_agent.py --exec_profile cpu --total_timesteps 5000 --initial_capital 15.0 --max_episode_steps 200

# Test 15 minutes (performances)
python scripts/train_rl_agent.py --exec_profile cpu --total_timesteps 15000 --initial_capital 15.0 --max_episode_steps 500
```

### Entraînement Production
```bash
# Standard 45 minutes
python scripts/train_rl_agent.py --exec_profile cpu --total_timesteps 50000 --initial_capital 15.0

# Long 2 heures
python scripts/train_rl_agent.py --exec_profile cpu --total_timesteps 100000 --initial_capital 15.0
```

### Évaluation
```bash
# Test rapide modèle
python scripts/test_model_quick.py --model_path models/votre_modele.zip --capital 15.0

# Évaluation complète
python scripts/evaluate_performance.py --model_path models/votre_modele.zip
```

---

## 🔗 PAPER TRADING LIVE

### Mode Inférence (Recommandé pour débuter)
```bash
# Test 30 minutes
python scripts/paper_trade_agent.py \
    --exec_profile cpu \
    --model_path models/votre_modele.zip \
    --initial_capital 15000 \
    --max_iterations 30 \
    --sleep_seconds 60

# Test 2 heures
python scripts/paper_trade_agent.py \
    --exec_profile cpu \
    --model_path models/votre_modele.zip \
    --initial_capital 15000 \
    --max_iterations 120 \
    --sleep_seconds 60
```

### Test Rapide Connexion
```bash
# Test 10 minutes (validation exchange)
python scripts/paper_trade_agent.py \
    --exec_profile cpu \
    --model_path models/votre_modele.zip \
    --initial_capital 10000 \
    --max_iterations 10 \
    --sleep_seconds 30
```

---

## 🧠 APPRENTISSAGE CONTINU

### Mode Conservateur (Débutant)
```bash
# Très prudent - Learning rate faible
python scripts/online_learning_agent.py \
    --exec_profile cpu \
    --model_path models/votre_modele.zip \
    --learning_rate 0.00001 \
    --exploration_rate 0.1 \
    --max_position_size 20.0 \
    --learning_frequency 20 \
    --max_iterations 50
```

### Mode Modéré (Intermédiaire)
```bash
# Équilibré après validation conservateur
python scripts/online_learning_agent.py \
    --exec_profile cpu \
    --model_path models/votre_modele.zip \
    --learning_rate 0.0001 \
    --exploration_rate 0.3 \
    --max_position_size 50.0 \
    --learning_frequency 10 \
    --max_iterations 200
```

### Mode Agressif (Avancé)
```bash
# Performance optimisée
python scripts/online_learning_agent.py \
    --exec_profile cpu \
    --model_path models/votre_modele.zip \
    --learning_rate 0.001 \
    --exploration_rate 0.5 \
    --max_position_size 100.0 \
    --learning_frequency 5 \
    --max_iterations 500
```

---

## 🤝 FEEDBACK HUMAIN

### Mode Interactif Complet
```bash
# Évaluation manuelle de chaque décision
python scripts/human_feedback_trading.py \
    --exec_profile cpu \
    --model_path models/votre_modele.zip \
    --interactive_mode true \
    --require_confirmation true \
    --learning_from_feedback true \
    --save_feedback_log true \
    --max_iterations 20
```

### Mode Hybride (Automatique + Humain)
```bash
# Validation humaine pour trades importants
python scripts/hybrid_learning_agent.py \
    --exec_profile cpu \
    --model_path models/votre_modele.zip \
    --auto_learning_rate 0.00001 \
    --human_feedback_threshold 50.0 \
    --max_iterations 100
```

---

## 📊 MONITORING ET SURVEILLANCE

### Monitoring Temps Réel
```bash
# Terminal 1: Agent
python scripts/online_learning_agent.py --exec_profile cpu --model_path models/votre_modele.zip

# Terminal 2: Monitoring
python scripts/monitor_training.py --live_mode true

# Terminal 3: Logs exchange
tail -f online_learning_log_$(date +%Y%m%d).json
```

### Surveillance Logs
```bash
# Logs paper trading
tail -f paper_trading_summary_*.json

# Logs apprentissage continu
tail -f online_learning_log_*.json

# Logs système
tail -f training_*.log
```

---

## 🎯 COMMANDES RUN_ADAN INTÉGRÉES

### Modes Principaux
```bash
# Mode automatique complet
python run_adan.py --mode auto --timesteps 50000 --capital 15.0

# Pipeline données uniquement
python run_adan.py --mode data_pipeline

# Test exchange uniquement
python run_adan.py --mode exchange_test

# Entraînement uniquement
python run_adan.py --mode train --timesteps 50000 --capital 15.0

# Paper trading
python run_adan.py --mode paper_trading --capital 15000 --iterations 50

# Apprentissage continu
python run_adan.py --mode online_learning --capital 15000 --iterations 100 --learning_rate 0.00001

# Feedback humain
python run_adan.py --mode human_feedback --capital 15000 --iterations 30
```

### Options Avancées
```bash
# Mode interactif
python run_adan.py --mode auto --interactive

# Mode rapide
python run_adan.py --mode auto --quick

# Mode verbose
python run_adan.py --mode train --verbose

# Paper trading avec apprentissage
python run_adan.py --mode paper_trading --learning --iterations 100
```

---

## 🛠️ COMMANDES DE DIAGNOSTIC

### Tests Complets
```bash
# Diagnostic système complet
python test_complete_system.py --exec_profile cpu

# Status détaillé
python status_adan.py

# Test exchange spécifique
python status_exchange_connector.py
```

### Tests Spécialisés
```bash
# Test basique CCXT
python test_ccxt_connection.py

# Test OrderManager avec exchange
python scripts/test_order_manager_only.py

# Test StateBuilder
python scripts/test_state_builder_features.py

# Validation environnement
python scripts/test_environment_with_merged_data.py
```

---

## 🔧 UTILITAIRES ET MAINTENANCE

### Nettoyage et Reset
```bash
# Nettoyer données processed
rm -rf data/processed/unified/*
rm -rf data/scalers_encoders/*

# Régénérer pipeline
python scripts/convert_real_data.py --exec_profile cpu

# Nettoyer modèles anciens
find models/ -name "*.zip" -mtime +7 -delete
```

### Gestion Modèles
```bash
# Lister modèles disponibles
ls -la models/*.zip

# Taille des modèles
du -sh models/*.zip

# Test chargement modèle
python -c "
from src.adan_trading_bot.training.trainer import load_agent
agent = load_agent('models/votre_modele.zip')
print('✅ Modèle chargé avec succès')
"
```

---

## 🚨 COMMANDES D'URGENCE

### Arrêt Sécurisé
```bash
# Arrêt gracieux (Ctrl+C puis confirmer)
# Ou kill process par PID
ps aux | grep python | grep adan
kill -TERM <PID>
```

### Diagnostic Urgent
```bash
# Vérifier connexions actives
netstat -an | grep :443
lsof -i :443

# Vérifier utilisation ressources
top -p $(pgrep -f python.*adan)
nvidia-smi  # Si GPU

# Logs d'erreur récents
grep -i error *.log | tail -20
```

### Recovery
```bash
# Sauvegarder état actuel
cp -r models/ backup_models_$(date +%Y%m%d)/
cp *.log backup_logs_$(date +%Y%m%d)/

# Reset environnement
conda deactivate
conda activate trading_env
cd ~/Desktop/ADAN/ADAN
```

---

## 📈 COMMANDES DE PERFORMANCE

### Benchmarking
```bash
# Test performance entraînement
time python scripts/train_rl_agent.py --exec_profile cpu --total_timesteps 5000

# Test performance paper trading
time python scripts/paper_trade_agent.py --model_path models/votre_modele.zip --max_iterations 10

# Test latence exchange
python -c "
import time
from src.adan_trading_bot.exchange_api.connector import get_exchange_client
from src.adan_trading_bot.common.utils import load_config
config = load_config('.', 'cpu')
exchange = get_exchange_client(config)
start = time.time()
ticker = exchange.fetch_ticker('BTC/USDT')
print(f'Latence: {(time.time()-start)*1000:.2f}ms')
"
```

### Métriques Système
```bash
# CPU/RAM pendant entraînement
top -p $(pgrep -f train_rl_agent) -b -n1

# Throughput données
python -c "
import time
import pandas as pd
start = time.time()
df = pd.read_parquet('data/processed/unified/1m_train_merged.parquet')
elapsed = time.time() - start
throughput = len(df) / elapsed
print(f'Throughput: {throughput:.0f} rows/sec')
"
```

---

## 🎉 WORKFLOWS COMPLETS

### Workflow Débutant
```bash
# 1. Test système
python test_complete_system.py --exec_profile cpu

# 2. Pipeline données
python scripts/convert_real_data.py --exec_profile cpu

# 3. Entraînement rapide
python scripts/train_rl_agent.py --exec_profile cpu --total_timesteps 5000 --initial_capital 15.0

# 4. Paper trading test
python scripts/paper_trade_agent.py --model_path models/interrupted_model.zip --max_iterations 5
```

### Workflow Production
```bash
# 1. Validation complète
python test_complete_system.py --exec_profile cpu
python test_exchange_connector.py

# 2. Entraînement production
python scripts/train_rl_agent.py --exec_profile cpu --total_timesteps 100000 --initial_capital 15.0

# 3. Paper trading validation 2h
python scripts/paper_trade_agent.py --model_path models/final_model.zip --max_iterations 120

# 4. Apprentissage continu conservateur
python scripts/online_learning_agent.py --model_path models/final_model.zip --learning_rate 0.00001 --max_iterations 200
```

### Workflow Recherche
```bash
# 1. Tests multi-timeframes
for tf in 1m 1h 1d; do
    python scripts/train_rl_agent.py --exec_profile cpu --total_timesteps 10000 --timeframe $tf
done

# 2. Comparaison performances
python scripts/evaluate_performance.py --model_path models/model_1m.zip
python scripts/evaluate_performance.py --model_path models/model_1h.zip

# 3. Apprentissage continu expérimental
python scripts/online_learning_agent.py --learning_rate 0.001 --exploration_rate 0.5 --max_iterations 1000
```

---

**🎯 ADAN v2.1 - Commandes Rapides pour Tous les Niveaux**

*Mise à jour: Toutes les fonctionnalités d'apprentissage continu et d'intégration exchange incluses*  
*Copyright 2025 - Advanced AI Trading Systems*