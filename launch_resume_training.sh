#!/bin/bash

# 🚀 LAUNCH RESUME TRAINING
# Reprend l'entraînement depuis les checkpoints existants

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         🚀 LAUNCHING RESUME TRAINING FROM CHECKPOINTS          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
CONFIG_FILE="config/config.yaml"
CHECKPOINT_DIR="/mnt/new_data/t10_training/checkpoints"
LOG_DIR="/mnt/new_data/t10_training/logs"

# Vérifications
echo "📋 PRE-LAUNCH CHECKS"
echo "═══════════════════════════════════════════════════════════════"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi
echo "✅ Config file found: $CONFIG_FILE"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi
echo "✅ Checkpoint directory found: $CHECKPOINT_DIR"

# Vérifier les checkpoints
echo ""
echo "📊 CHECKPOINT STATUS"
echo "═══════════════════════════════════════════════════════════════"

for worker in w1 w2 w3 w4; do
    worker_dir="$CHECKPOINT_DIR/$worker"
    if [ -d "$worker_dir" ]; then
        latest=$(ls -t "$worker_dir"/${worker}_model_*.zip 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            steps=$(basename "$latest" | grep -oP '\d+(?=_steps)' | tail -1)
            size=$(du -h "$latest" | cut -f1)
            echo "✅ $worker: $steps steps ($size)"
        else
            echo "⚠️  $worker: No checkpoints found"
        fi
    else
        echo "❌ $worker: Directory not found"
    fi
done

echo ""
echo "🚀 STARTING RESUME TRAINING"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Command:"
echo "  python scripts/train_parallel_agents.py \\"
echo "    --config $CONFIG_FILE \\"
echo "    --resume \\"
echo "    --checkpoint-dir $CHECKPOINT_DIR"
echo ""
echo "Expected behavior:"
echo "  1. Script finds latest checkpoint for each worker"
echo "  2. Loads model with PPO.load() (preserves weights)"
echo "  3. Reads current num_timesteps (e.g., 170,000 for W1)"
echo "  4. Calculates remaining steps (e.g., 80,000 for W1)"
echo "  5. Trains for remaining steps with reset_num_timesteps=False"
echo "  6. Logs metrics for ALL workers to central_logger"
echo ""
echo "Timeline:"
echo "  W1: ~12 hours (80,000 steps)"
echo "  W2: ~13 hours (85,000 steps)"
echo "  W3: ~15 hours (100,000 steps)"
echo "  W4: ~14 hours (90,000 steps)"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Créer le répertoire de logs s'il n'existe pas
mkdir -p "$LOG_DIR"

# Lancer l'entraînement
python scripts/train_parallel_agents.py \
    --config "$CONFIG_FILE" \
    --resume \
    --checkpoint-dir "$CHECKPOINT_DIR"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                  ✅ TRAINING COMPLETED                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
