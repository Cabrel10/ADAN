#!/bin/bash
# Script de monitoring de l'entraînement ADAN en temps réel

LOG_FILE="/mnt/new_data/adan_logs/training_session_*.log"
INTERVAL=30  # Vérifier toutes les 30 secondes

echo "🎯 MONITORING ENTRAÎNEMENT ADAN"
echo "================================"
echo "Logs: $LOG_FILE"
echo "Interval: ${INTERVAL}s"
echo ""

while true; do
    clear
    echo "🎯 MONITORING ENTRAÎNEMENT ADAN - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================"
    
    # Vérifier les processus
    PROC_COUNT=$(ps aux | grep -E "train_parallel|python.*train" | grep -v grep | wc -l)
    echo "📊 Processus actifs: $PROC_COUNT"
    
    # Taille du log
    LOG_SIZE=$(du -sh /mnt/new_data/adan_logs/training_session_*.log 2>/dev/null | tail -1 | awk '{print $1}')
    echo "📁 Taille log: $LOG_SIZE"
    
    # Espace disque
    DISK_FREE=$(df -h /mnt/new_data/ | tail -1 | awk '{print $4}')
    echo "💾 Espace libre: $DISK_FREE"
    
    echo ""
    echo "📈 PROGRESSION PAR WORKER:"
    echo "---"
    
    # Compter les steps par worker
    for worker in w0 w1 w2 w3; do
        STEPS=$(grep -c "\[STEP.*Worker.*$worker" /mnt/new_data/adan_logs/training_session_*.log 2>/dev/null || echo "0")
        TRADES=$(grep -c "\[TRADE\].*Worker.*$worker" /mnt/new_data/adan_logs/training_session_*.log 2>/dev/null || echo "0")
        PORTFOLIO=$(grep "Worker.*$worker.*Portfolio value" /mnt/new_data/adan_logs/training_session_*.log 2>/dev/null | tail -1 | grep -oP 'Portfolio value: \K[0-9.]+' || echo "N/A")
        echo "$worker: Steps=$STEPS, Trades=$TRADES, Portfolio=$PORTFOLIO"
    done
    
    echo ""
    echo "📝 DERNIERS LOGS:"
    echo "---"
    tail -10 /mnt/new_data/adan_logs/training_session_*.log 2>/dev/null | grep -E "STEP|TRADE|Worker|Portfolio" | tail -5
    
    echo ""
    echo "⏱️  Prochaine vérification dans ${INTERVAL}s (Ctrl+C pour arrêter)"
    sleep $INTERVAL
done
