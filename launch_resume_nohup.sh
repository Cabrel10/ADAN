#!/bin/bash

# 🚀 LAUNCH RESUME TRAINING WITH NOHUP
# Relance l'entraînement détaché avec 100k steps supplémentaires

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║    🚀 LAUNCHING RESUME TRAINING WITH NOHUP (DETACHED)          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
CONFIG_FILE="config/config.yaml"
CHECKPOINT_DIR="/mnt/new_data/t10_training/checkpoints"
LOG_DIR="/mnt/new_data/t10_training/logs"
NOHUP_LOG="nohup_training_$(date +%Y%m%d_%H%M%S).log"

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

# Vérifier les checkpoints actuels
echo ""
echo "📊 CURRENT CHECKPOINT STATUS"
echo "═══════════════════════════════════════════════════════════════"

for worker in w1 w2 w3 w4; do
    worker_dir="$CHECKPOINT_DIR/$worker"
    if [ -d "$worker_dir" ]; then
        latest=$(ls -t "$worker_dir"/${worker}_model_*.zip 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            steps=$(basename "$latest" | grep -oP '\d+(?=_steps)' | tail -1)
            size=$(du -h "$latest" | cut -f1)
            mtime=$(stat -c %y "$latest" | cut -d' ' -f1,2)
            echo "✅ $worker: $steps steps ($size) - Updated: $mtime"
        else
            echo "⚠️  $worker: No checkpoints found"
        fi
    else
        echo "❌ $worker: Directory not found"
    fi
done

echo ""
echo "🚀 LAUNCHING WITH NOHUP (DETACHED)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Command:"
echo "  nohup python scripts/train_parallel_agents.py \\"
echo "    --config $CONFIG_FILE \\"
echo "    --resume \\"
echo "    --checkpoint-dir $CHECKPOINT_DIR \\"
echo "    > $NOHUP_LOG 2>&1 &"
echo ""
echo "Log file: $NOHUP_LOG"
echo ""
echo "New target steps: 350,000 (100k more for each worker)"
echo "  W1: 210k → 350k (140k remaining)"
echo "  W2: 205k → 350k (145k remaining)"
echo "  W3: 185k → 350k (165k remaining)"
echo "  W4: 200k → 350k (150k remaining)"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Créer le répertoire de logs s'il n'existe pas
mkdir -p "$LOG_DIR"

# Lancer l'entraînement avec nohup (détaché)
nohup python scripts/train_parallel_agents.py \
    --config "$CONFIG_FILE" \
    --resume \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    > "$NOHUP_LOG" 2>&1 &

TRAINING_PID=$!

echo "✅ Training launched with PID: $TRAINING_PID"
echo "✅ Process is detached and will continue even if terminal closes"
echo ""
echo "Monitor with:"
echo "  tail -f $NOHUP_LOG"
echo ""
echo "Check status:"
echo "  ps aux | grep train_parallel_agents"
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              ✅ TRAINING LAUNCHED (DETACHED)                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
