#!/bin/bash

# Script d'entraînement final optimisé pour ADAN
# Usage: ./launch_final_training.sh [config_file]

# Configuration par défaut
CONFIG_FILE=${1:-"config/main_config.yaml"}

echo "🚀 ADAN - ENTRAÎNEMENT FINAL"
echo "⏰ Début: $(date)"
echo "📊 Fichier de configuration: $CONFIG_FILE"
echo ""

# Vérification du fichier de configuration
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Erreur: Le fichier de configuration $CONFIG_FILE n'existe pas"
    exit 1
fi

# Activation de l'environnement
echo "🔧 Activation environnement trading_env..."
source activate trading_env

# Vérification espace disque
echo "💾 Espace disque disponible:"
df -h . | grep -v "Sys"

# Création du fichier de log avec timestamp
LOG_FILE="training_final_$(date +%Y%m%d_%H%M%S).log"

# Nettoyage préventif
echo "🧹 Nettoyage préventif..."
find models/ -name "*.zip" -mtime +3 -delete 2>/dev/null || true
find reports/tensorboard_logs/ -name "*" -mtime +2 -delete 2>/dev/null || true

echo ""
echo "🎯 Lancement de l'entraînement..."
echo "📝 Logs sauvegardés dans: $LOG_FILE"
echo ""

# Lancement avec gestion d'erreur
if nohup python scripts/train_rl_agent.py \
    --config "$CONFIG_FILE" \
    > $LOG_FILE 2>&1 &
then
    TRAIN_PID=$!
    echo "✅ Entraînement lancé avec PID: $TRAIN_PID"
    echo "📊 Surveillance: tail -f $LOG_FILE"
    echo "🛑 Arrêt: kill $TRAIN_PID"
    echo ""
    
    # Surveillance initiale
    echo "🔍 Surveillance des 30 premières secondes..."
    sleep 30
    
    if ps -p $TRAIN_PID > /dev/null; then
        echo "✅ Processus actif"
        echo "📈 Dernières lignes:"
        tail -5 $LOG_FILE 2>/dev/null || echo "Log en cours de création..."
    else
        echo "❌ Processus arrêté - Vérifiez le log:"
        tail -10 $LOG_FILE 2>/dev/null || echo "Aucun log trouvé"
    fi
else
    echo "❌ Échec du lancement"
    exit 1
fi

echo ""
echo "🎯 Entraînement en cours d'exécution"
echo "📊 Commandes utiles:"
echo "   - Surveillance: tail -f $LOG_FILE"
echo "   - Status: ps aux | grep train_rl_agent"
echo "   - TensorBoard: tensorboard --logdir reports/tensorboard_logs/"
echo "   - Espace disque: watch -n 60 'df -h .'"