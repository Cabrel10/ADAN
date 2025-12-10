#!/bin/bash

# 📊 SCRIPT DE MONITORING AUTOMATIQUE
# Surveille l'entraînement et alerte en cas de problème

LOG_FILE="/mnt/new_data/adan_logs/training_20251207_140717.log"
CHECK_INTERVAL=300  # 5 minutes
ALERT_EMAIL=""  # Laisser vide si pas d'email

echo "🔍 MONITORING AUTOMATIQUE LANCÉ"
echo "================================"
echo "Log: $LOG_FILE"
echo "Intervalle: $CHECK_INTERVAL secondes"
echo ""

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Vérification..."
    
    # Vérifier que les processus sont actifs
    PROCESS_COUNT=$(ps aux | grep -E "train_parallel_agents.py" | grep -v grep | wc -l)
    
    if [ $PROCESS_COUNT -lt 5 ]; then
        echo "❌ ALERTE: Seulement $PROCESS_COUNT processus actifs (attendu: 5)"
        echo "   Processus:"
        ps aux | grep -E "train_parallel_agents.py" | grep -v grep
    else
        echo "✅ Processus: $PROCESS_COUNT actifs"
    fi
    
    # Vérifier les erreurs NaN
    NAN_COUNT=$(grep -i "nan\|invalid values" "$LOG_FILE" 2>/dev/null | wc -l)
    if [ $NAN_COUNT -gt 0 ]; then
        echo "❌ ALERTE: $NAN_COUNT erreurs NaN détectées!"
        grep -i "nan\|invalid values" "$LOG_FILE" | tail -5
    else
        echo "✅ Pas d'erreurs NaN"
    fi
    
    # Vérifier les exceptions
    ERROR_COUNT=$(grep -i "error\|exception" "$LOG_FILE" 2>/dev/null | grep -v "ERROR -" | wc -l)
    if [ $ERROR_COUNT -gt 10 ]; then
        echo "⚠️  ATTENTION: $ERROR_COUNT erreurs détectées"
    else
        echo "✅ Pas d'erreurs critiques"
    fi
    
    # Afficher les derniers steps
    LAST_STEP=$(grep "STEP" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '\[STEP \K[0-9]+' | head -1)
    LAST_PORTFOLIO=$(grep "Portfolio value:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP 'Portfolio value: \K[0-9.]+')
    
    if [ ! -z "$LAST_STEP" ]; then
        echo "📈 Dernier step: $LAST_STEP"
    fi
    
    if [ ! -z "$LAST_PORTFOLIO" ]; then
        echo "💰 Portfolio value: $LAST_PORTFOLIO"
    fi
    
    # Taille du log
    LOG_SIZE=$(du -h "$LOG_FILE" 2>/dev/null | cut -f1)
    echo "📁 Taille du log: $LOG_SIZE"
    
    echo "---"
    sleep $CHECK_INTERVAL
done
