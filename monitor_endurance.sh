#!/bin/bash

echo "📊 MONITORING TEST D'ENDURANCE (3 HEURES)"
echo "=========================================="
echo ""

PID_FILE="deploy/adan_bot/logs/endurance_test.pid"
LOG_FILE="deploy/adan_bot/logs/endurance_test.log"

if [ ! -f "$PID_FILE" ]; then
    echo "❌ Fichier PID non trouvé"
    exit 1
fi

PID=$(cat "$PID_FILE")
START_TIME=$(date +%s)
DURATION=$((3 * 3600))  # 3 heures en secondes

echo "🚀 PID: $PID"
echo "📝 Log: $LOG_FILE"
echo "⏱️  Durée: 3 heures (10800 secondes)"
echo ""

# Fonction pour afficher le statut
show_status() {
    local elapsed=$(($(date +%s) - START_TIME))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))
    
    echo "⏱️  Temps écoulé: ${hours}h ${minutes}m ${seconds}s"
    
    # Vérifier si le processus est toujours actif
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Processus actif (PID $PID)"
    else
        echo "❌ Processus arrêté!"
        return 1
    fi
    
    # Compter les cycles
    local cycles=$(grep -c "🔄 Cycle" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "🔄 Cycles complétés: $cycles"
    
    # Vérifier les erreurs
    local errors=$(grep -c "ERROR" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "⚠️  Erreurs: $errors"
    
    # Vérifier les décisions ADAN
    local decisions=$(grep -c "ADAN:" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "🤖 Décisions ADAN: $decisions"
    
    # Dernière ligne du log
    echo ""
    echo "📋 Dernière activité:"
    tail -3 "$LOG_FILE" | sed 's/^/   /'
    
    return 0
}

# Boucle de monitoring
iteration=0
while true; do
    clear
    echo "📊 MONITORING TEST D'ENDURANCE (3 HEURES)"
    echo "=========================================="
    echo ""
    
    show_status
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ TEST ÉCHOUÉ: Processus arrêté prématurément"
        exit 1
    fi
    
    elapsed=$(($(date +%s) - START_TIME))
    if [ $elapsed -ge $DURATION ]; then
        echo ""
        echo "✅ TEST RÉUSSI: 3 heures complétées!"
        echo ""
        echo "📊 Résumé final:"
        tail -20 "$LOG_FILE" | sed 's/^/   /'
        exit 0
    fi
    
    iteration=$((iteration + 1))
    echo ""
    echo "🔄 Mise à jour dans 30 secondes... (Itération $iteration)"
    sleep 30
done
