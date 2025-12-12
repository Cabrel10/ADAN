#!/bin/bash

# T10 : LANCEMENT ENTRAÎNEMENT FINAL
# Utilise scripts/train_parallel_agents.py avec /mnt/new_data pour les résultats

set -e

WORK_DIR="/mnt/new_data/t10_training"
LOG_DIR="$WORK_DIR/logs"
CHECKPOINT_DIR="$WORK_DIR/checkpoints"

# Créer répertoires
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

echo "================================================================================"
echo "T10 : LANCEMENT ENTRAÎNEMENT FINAL"
echo "================================================================================"
echo ""
echo "📁 Configuration:"
echo "   Script: scripts/train_parallel_agents.py"
echo "   Config: config/config.yaml"
echo "   Logs: $LOG_DIR"
echo "   Checkpoints: $CHECKPOINT_DIR"
echo "   Disque: /mnt/new_data (21 GB disponible)"
echo ""
echo "🎯 Objectif:"
echo "   - Entraîner 4 workers en parallèle"
echo "   - 250k steps par worker (1M total)"
echo "   - Utiliser hyperparamètres Optuna optimisés"
echo "   - Appliquer hiérarchie complète"
echo ""

# Lancer l'entraînement
echo "🚀 Lancement de l'entraînement..."
echo ""

cd /home/morningstar/Documents/trading/bot

python3 scripts/train_parallel_agents.py \
    --config config/config.yaml \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --steps 250000 \
    --log-level INFO \
    2>&1 | tee "$LOG_DIR/t10_main.log"

echo ""
echo "================================================================================"
echo "✅ ENTRAÎNEMENT COMPLÉTÉ"
echo "================================================================================"
echo ""
echo "📊 Résultats disponibles dans:"
echo "   $CHECKPOINT_DIR/final/"
echo ""
echo "📈 Pour analyser les résultats:"
echo "   python3 validate_results.py"
echo ""
