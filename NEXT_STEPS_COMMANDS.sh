#!/bin/bash

################################################################################
# 🚀 ADAN TRADING BOT - NEXT STEPS COMMANDS
# 
# Ce script contient toutes les commandes à exécuter pour continuer
# après la correction du DBELogger NameError
#
# Date: 2025-11-18
# Status: ✅ Prêt à exécuter
################################################################################

set -e

PROJECT_DIR="/home/morningstar/Documents/trading/bot"
LOGS_DIR="/mnt/new_data/adan_logs"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"

echo "=================================================================================="
echo "🎯 ADAN TRADING BOT - NEXT STEPS EXECUTION"
echo "=================================================================================="
echo ""

# ============================================================================
# ÉTAPE 1: VÉRIFICATION PRÉ-LANCEMENT
# ============================================================================

echo "📋 ÉTAPE 1: Vérification pré-lancement..."
echo ""

cd "$PROJECT_DIR"

# Vérifier que le code est correct
echo "  ✓ Vérification syntaxe Python..."
python -m py_compile src/adan_trading_bot/environment/dynamic_behavior_engine.py
echo "    ✅ Syntaxe OK"

# Vérifier l'import
echo "  ✓ Vérification import DBE..."
python -c "from src.adan_trading_bot.environment.dynamic_behavior_engine import DynamicBehaviorEngine; print('    ✅ Import OK')"

# Vérifier les données
echo "  ✓ Vérification données parquet..."
PARQUET_COUNT=$(find "$PROJECT_DIR/data/processed/indicators" -name "*.parquet" | wc -l)
echo "    ✅ $PARQUET_COUNT fichiers parquet trouvés"

# Vérifier l'espace disque
echo "  ✓ Vérification espace disque..."
DISK_SPACE=$(df "$PROJECT_DIR" | tail -1 | awk '{print $4}')
echo "    ✅ Espace disponible: $((DISK_SPACE / 1024 / 1024)) GB"

echo ""
echo "✅ Vérifications pré-lancement réussies"
echo ""

# ============================================================================
# ÉTAPE 2: VALIDATION 500K TIMESTEPS
# ============================================================================

echo "=================================================================================="
echo "🚀 ÉTAPE 2: VALIDATION 500K TIMESTEPS"
echo "=================================================================================="
echo ""

# Créer le répertoire de logs s'il n'existe pas
mkdir -p "$LOGS_DIR"

# Générer le timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOGS_DIR/validation_$TIMESTAMP.log"

echo "📝 Logs seront sauvegardés dans: $LOG_FILE"
echo ""
echo "⏱️  Durée estimée: 6-8 heures"
echo "🎯 Objectifs:"
echo "   - Aucun crash"
echo "   - Sharpe ratio > 2.0 après 300k steps"
echo "   - Max drawdown < 20%"
echo "   - Win rate > 50%"
echo ""

# Lancer l'entraînement
echo "🚀 Lancement de l'entraînement..."
echo ""

timeout 28800 python scripts/train_parallel_agents.py \
  --config-path config/config.yaml \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --resume \
  --log-level INFO \
  > "$LOG_FILE" 2>&1 &

TRAINING_PID=$!
echo "✅ Entraînement lancé avec PID: $TRAINING_PID"
echo ""

# Monitoring en temps réel
echo "📊 Monitoring en temps réel (Ctrl+C pour arrêter le monitoring)..."
echo ""

sleep 5

# Afficher les premiers logs
echo "Premiers logs:"
tail -20 "$LOG_FILE"
echo ""

# Monitoring continu
echo "Pour suivre les logs en temps réel, utilisez:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Pour chercher des erreurs:"
echo "  grep -i 'error\|exception' $LOG_FILE"
echo ""
echo "Pour chercher les décisions DBE:"
echo "  grep '\[DBE_DECISION\]' $LOG_FILE | tail -20"
echo ""
echo "Pour vérifier le statut du processus:"
echo "  ps aux | grep train_parallel_agents.py"
echo ""

# ============================================================================
# ÉTAPE 3: POST-ENTRAÎNEMENT
# ============================================================================

echo "=================================================================================="
echo "📋 APRÈS L'ENTRAÎNEMENT (6-8 heures plus tard)"
echo "=================================================================================="
echo ""
echo "1. Vérifier les résultats:"
echo "   tail -100 $LOG_FILE"
echo ""
echo "2. Chercher les erreurs:"
echo "   grep -i 'error\|exception' $LOG_FILE | wc -l"
echo ""
echo "3. Vérifier les checkpoints:"
echo "   ls -lh $CHECKPOINT_DIR/"
echo ""
echo "4. Analyser les métriques:"
echo "   grep '\[TERMINATION CHECK\]' $LOG_FILE | tail -5"
echo ""

# ============================================================================
# ÉTAPE 4: OPTIMISATION HYPERPARAMÈTRES
# ============================================================================

echo "=================================================================================="
echo "🔧 ÉTAPE 4: OPTIMISATION HYPERPARAMÈTRES (OPTUNA)"
echo "=================================================================================="
echo ""
echo "Après validation 500k timesteps réussie, lancer Optuna:"
echo ""
echo "# Worker 1 (3-4 heures)"
echo "python scripts/optimize_hyperparams.py --worker w1"
echo ""
echo "# Worker 2 (3-4 heures)"
echo "python scripts/optimize_hyperparams.py --worker w2"
echo ""
echo "# Worker 3 (3-4 heures)"
echo "python scripts/optimize_hyperparams.py --worker w3"
echo ""
echo "# Worker 4 (3-4 heures)"
echo "python scripts/optimize_hyperparams.py --worker w4"
echo ""
echo "Durée totale: 12-16 heures (séquentiel)"
echo ""

# ============================================================================
# COMMANDES UTILES
# ============================================================================

echo "=================================================================================="
echo "🛠️  COMMANDES UTILES"
echo "=================================================================================="
echo ""
echo "# Arrêter l'entraînement"
echo "pkill -f train_parallel_agents.py"
echo ""
echo "# Vérifier les logs en temps réel"
echo "tail -f $LOG_FILE"
echo ""
echo "# Compter les décisions DBE"
echo "grep -c '\[DBE_DECISION\]' $LOG_FILE"
echo ""
echo "# Vérifier les erreurs"
echo "grep -i 'error\|exception' $LOG_FILE | head -20"
echo ""
echo "# Vérifier l'utilisation mémoire"
echo "ps aux | grep train_parallel_agents"
echo ""
echo "# Vérifier l'espace disque"
echo "df -h $PROJECT_DIR"
echo ""

echo "=================================================================================="
echo "✅ PRÊT POUR LANCEMENT"
echo "=================================================================================="
echo ""
echo "Le code est maintenant stable et prêt pour l'entraînement complet."
echo "Aucune erreur DBELogger ne devrait apparaître lors de l'exécution."
echo ""
echo "Repository: https://github.com/Cabrel10/ADAN0"
echo ""
