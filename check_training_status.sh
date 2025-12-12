#!/bin/bash

# Quick training status check
# Usage: ./check_training_status.sh [tail|monitor|full]

LOG_DIR="/mnt/new_data/t10_training/logs"
CHECKPOINT_DIR="/mnt/new_data/t10_training/checkpoints"

# Get latest log file
LATEST_LOG=$(ls -t "$LOG_DIR"/training_final_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ Aucun fichier log trouvé"
    exit 1
fi

echo "📊 STATUS ENTRAÎNEMENT T10"
echo "=================================="
echo ""

# Check if process is running
if pgrep -f "train_parallel_agents.py" > /dev/null; then
    echo "✅ Processus: EN COURS"
    echo "   Processus actifs: $(pgrep -f 'train_parallel_agents.py' | wc -l)"
else
    echo "❌ Processus: ARRÊTÉ"
fi

echo ""
echo "📁 Fichiers:"
echo "   Log: $(basename "$LATEST_LOG")"
echo "   Taille: $(du -h "$LATEST_LOG" | cut -f1)"
echo ""

# System stats
echo "💾 Système:"
RAM_PERCENT=$(free | grep Mem | awk '{printf("%.1f", $3/$2 * 100)}')
DISK_FREE=$(df /mnt/new_data | tail -1 | awk '{printf("%.1f", $4/1024/1024)}')
echo "   RAM: $RAM_PERCENT%"
echo "   Disque /mnt/new_data: ${DISK_FREE} GB libre"
echo ""

# Checkpoint status
CHECKPOINT_COUNT=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d | wc -l)
CHECKPOINT_SIZE=$(du -sh "$CHECKPOINT_DIR" 2>/dev/null | cut -f1)
echo "📈 Checkpoints:"
echo "   Nombre: $((CHECKPOINT_COUNT - 1))"
echo "   Taille: $CHECKPOINT_SIZE"
echo ""

# Show action based on argument
case "${1:-tail}" in
    tail)
        echo "📝 Dernières lignes du log:"
        echo "---"
        tail -20 "$LATEST_LOG"
        ;;
    monitor)
        echo "🔄 Surveillance en direct (Ctrl+C pour arrêter):"
        echo "---"
        tail -f "$LATEST_LOG"
        ;;
    full)
        echo "📄 Contenu complet du log:"
        echo "---"
        cat "$LATEST_LOG"
        ;;
    *)
        echo "Usage: $0 [tail|monitor|full]"
        echo "  tail    - Affiche les 20 dernières lignes (défaut)"
        echo "  monitor - Surveillance en direct"
        echo "  full    - Affiche le log complet"
        ;;
esac

