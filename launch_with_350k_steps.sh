#!/bin/bash

# 🚀 LAUNCH RESUME TRAINING WITH 350K STEPS (100K MORE)
# Modifie la config temporairement et relance avec nohup

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  🚀 LAUNCHING RESUME WITH 350K STEPS (100K MORE) - NOHUP       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
CONFIG_FILE="config/config.yaml"
CONFIG_BACKUP="config/config.yaml.backup_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="/mnt/new_data/t10_training/checkpoints"
LOG_DIR="/mnt/new_data/t10_training/logs"
NOHUP_LOG="nohup_training_350k_$(date +%Y%m%d_%H%M%S).log"

# Vérifications
echo "📋 PRE-LAUNCH CHECKS"
echo "═══════════════════════════════════════════════════════════════"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi
echo "✅ Config file found: $CONFIG_FILE"

# Backup config
cp "$CONFIG_FILE" "$CONFIG_BACKUP"
echo "✅ Config backed up to: $CONFIG_BACKUP"

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
            remaining=$((350000 - steps))
            echo "✅ $worker: $steps steps → 350k (remaining: $remaining) - Updated: $mtime"
        else
            echo "⚠️  $worker: No checkpoints found"
        fi
    else
        echo "❌ $worker: Directory not found"
    fi
done

echo ""
echo "🔧 MODIFYING CONFIG"
echo "═══════════════════════════════════════════════════════════════"

# Modifier timesteps_per_instance de 500000 à 350000
sed -i 's/timesteps_per_instance: 500000/timesteps_per_instance: 350000/' "$CONFIG_FILE"
echo "✅ Modified timesteps_per_instance: 500000 → 350000"

# Vérifier la modification
current=$(grep "timesteps_per_instance:" "$CONFIG_FILE" | head -1)
echo "✅ Verified: $current"

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
echo "  W1: 210k → 350k (140k remaining) ~21 hours"
echo "  W2: 205k → 350k (145k remaining) ~22 hours"
echo "  W3: 185k → 350k (165k remaining) ~25 hours"
echo "  W4: 200k → 350k (150k remaining) ~23 hours"
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
echo "Restore config (if needed):"
echo "  cp $CONFIG_BACKUP $CONFIG_FILE"
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         ✅ TRAINING LAUNCHED (DETACHED) - 350K STEPS           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
