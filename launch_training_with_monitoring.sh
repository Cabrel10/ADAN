#!/bin/bash

# T10 Final Training Launch with Monitoring
# Launches training in background and starts monitoring

set -e

WORK_DIR="/mnt/new_data/t10_training"
LOG_DIR="$WORK_DIR/logs"
CHECKPOINT_DIR="$WORK_DIR/checkpoints"
BOT_DIR="/home/morningstar/Documents/trading/bot"

# Create directories
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

echo "================================================================================"
echo "T10 : LANCEMENT ENTRAÎNEMENT FINAL AVEC SURVEILLANCE"
echo "================================================================================"
echo ""
echo "📁 Configuration:"
echo "   Répertoire: $BOT_DIR"
echo "   Script: scripts/train_parallel_agents.py"
echo "   Config: config/config.yaml"
echo "   Logs: $LOG_DIR"
echo "   Checkpoints: $CHECKPOINT_DIR"
echo ""

# Activate conda environment and launch training in background
echo "🚀 Lancement de l'entraînement en arrière-plan..."
echo ""

cd "$BOT_DIR"

# Launch training with nohup and log redirection
nohup bash -c "
source ~/miniconda3/bin/activate trading_env
python scripts/train_parallel_agents.py \
    --config config/config.yaml \
    --checkpoint-dir $CHECKPOINT_DIR \
    --steps 1000000 \
    > $LOG_DIR/training_final_\$(date +%Y%m%d_%H%M%S).log 2>&1
" > /dev/null 2>&1 &

TRAIN_PID=$!
echo "✅ Entraînement lancé (PID: $TRAIN_PID)"
echo ""

# Wait a moment for process to start
sleep 2

# Check if process is still running
if ps -p $TRAIN_PID > /dev/null 2>&1; then
    echo "✅ Processus confirmé en cours d'exécution"
else
    echo "❌ Erreur: Le processus n'a pas démarré correctement"
    exit 1
fi

echo ""
echo "📊 Surveillance en cours..."
echo "   Appuyez sur Ctrl+C pour arrêter la surveillance (l'entraînement continue)"
echo ""

# Start monitoring
python3 monitor_t10_longterm.py

