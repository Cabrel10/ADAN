#!/bin/bash

# Script de surveillance de l'entraînement ADAN
# Usage: ./monitor_training.sh [log_file]

LOG_FILE=${1:-"training_final_*.log"}

echo "📊 ADAN - SURVEILLANCE ENTRAÎNEMENT"
echo "⏰ $(date)"
echo "🔍 Monitoring: $LOG_FILE"
echo ""

# Fonction de surveillance continue
monitor_training() {
    while true; do
        clear
        echo "📊 ADAN Training Monitor - $(date)"
        echo "═══════════════════════════════════════════════"
        
        # Status processus
        echo "🔧 Status Processus:"
        if pgrep -f "train_rl_agent" > /dev/null; then
            echo "✅ Entraînement ACTIF"
            echo "   PID: $(pgrep -f train_rl_agent)"
            echo "   CPU: $(ps -p $(pgrep -f train_rl_agent) -o %cpu --no-headers)%"
            echo "   RAM: $(ps -p $(pgrep -f train_rl_agent) -o %mem --no-headers)%"
        else
            echo "❌ Aucun entraînement détecté"
        fi
        echo ""
        
        # Espace disque
        echo "💾 Espace Disque:"
        df -h . | grep -E "(Filesystem|/dev)"
        echo ""
        
        # Dernières métriques d'entraînement
        echo "📈 Dernières Métriques:"
        if ls $LOG_FILE 1> /dev/null 2>&1; then
            tail -20 $(ls -t $LOG_FILE | head -1) | grep -E "(Episode|Training|timesteps|reward|loss)" | tail -5
        else
            echo "   Aucun log trouvé"
        fi
        echo ""
        
        # Modèles sauvegardés
        echo "💾 Modèles Récents:"
        if [ -d "models/" ]; then
            ls -lt models/*.zip 2>/dev/null | head -3 | awk '{print "   " $9 " (" $5 " bytes, " $6 " " $7 " " $8 ")"}'
        else
            echo "   Aucun modèle trouvé"
        fi
        echo ""
        
        # TensorBoard logs
        echo "📊 Logs TensorBoard:"
        if [ -d "reports/tensorboard_logs/" ]; then
            LATEST_TB=$(find reports/tensorboard_logs/ -type f -name "events.out.tfevents.*" | head -1)
            if [ -n "$LATEST_TB" ]; then
                echo "   ✅ Logs disponibles - Lancer: tensorboard --logdir reports/tensorboard_logs/"
            else
                echo "   ❌ Aucun log TensorBoard"
            fi
        fi
        echo ""
        
        echo "🔄 Actualisation dans 30s... (Ctrl+C pour arrêter)"
        sleep 30
    done
}

# Fonction de surveillance simple
monitor_simple() {
    echo "📝 Surveillance simple des logs..."
    if ls $LOG_FILE 1> /dev/null 2>&1; then
        tail -f $(ls -t $LOG_FILE | head -1) | grep -E "(Episode|Training|timesteps|reward|loss|SUCCÈS|ERROR)"
    else
        echo "❌ Aucun fichier log trouvé: $LOG_FILE"
    fi
}

# Menu principal
echo "Choisissez le mode de surveillance:"
echo "1) Surveillance complète (rafraîchissement automatique)"
echo "2) Surveillance simple (suivi des logs)"
echo "3) Status rapide"
read -p "Votre choix [1-3]: " choice

case $choice in
    1)
        monitor_training
        ;;
    2)
        monitor_simple
        ;;
    3)
        echo "📊 Status Rapide:"
        if pgrep -f "train_rl_agent" > /dev/null; then
            echo "✅ Entraînement en cours"
            echo "📈 Dernières lignes:"
            tail -5 $(ls -t $LOG_FILE | head -1) 2>/dev/null || echo "Aucun log"
        else
            echo "❌ Aucun entraînement"
        fi
        ;;
    *)
        echo "❌ Choix invalide"
        exit 1
        ;;
esac