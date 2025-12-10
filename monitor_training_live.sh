#!/bin/bash
# Monitoring en temps réel de l'entraînement ADAN

LOG_DIR="/mnt/new_data/adan_logs"
LATEST_LOG=$(ls -t $LOG_DIR/training_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ Aucun log trouvé dans $LOG_DIR"
    exit 1
fi

echo "📊 MONITORING ENTRAÎNEMENT ADAN"
echo "================================"
echo "Log: $(basename $LATEST_LOG)"
echo ""

# Boucle de monitoring
while true; do
    clear
    echo "🎯 MONITORING ENTRAÎNEMENT ADAN - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================"
    echo ""
    
    # Statut des processus
    PROCS=$(ps aux | grep -E "train_parallel|python.*train" | grep -v grep | wc -l)
    echo "📊 Processus actifs: $PROCS"
    echo ""
    
    # Taille du log
    SIZE=$(du -h "$LATEST_LOG" | awk '{print $1}')
    LINES=$(wc -l < "$LATEST_LOG")
    echo "📁 Log: $(basename $LATEST_LOG)"
    echo "   Taille: $SIZE"
    echo "   Lignes: $LINES"
    echo ""
    
    # Espace disque
    DISK=$(df -h /mnt/new_data/ | tail -1 | awk '{print $4}')
    echo "💾 Espace disque libre: $DISK"
    echo ""
    
    # Derniers logs
    echo "📈 Derniers logs (STEP/TRADE/RISK):"
    echo "---"
    tail -20 "$LATEST_LOG" | grep -E "STEP|TRADE|RISK_UPDATE|Portfolio value" | tail -10
    echo ""
    
    # Erreurs
    ERRORS=$(grep -c "ERROR\|Exception\|Traceback" "$LATEST_LOG" 2>/dev/null || echo "0")
    echo "❌ Erreurs détectées: $ERRORS"
    echo ""
    
    # Progression
    LATEST_STEP=$(grep -oP '\[STEP \K[0-9]+' "$LATEST_LOG" | tail -1 || echo "0")
    echo "🎯 Dernier step: $LATEST_STEP / 1000000"
    PROGRESS=$(echo "scale=2; $LATEST_STEP / 1000000 * 100" | bc)
    echo "   Progression: $PROGRESS%"
    echo ""
    
    echo "⏱️  Mise à jour toutes les 30 secondes (Ctrl+C pour arrêter)"
    sleep 30
done
