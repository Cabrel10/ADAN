#!/bin/bash

# 🚀 SCRIPT DE LANCEMENT ADAN - ENTRAÎNEMENT DÉTACHÉ
# L'entraînement continue même après fermeture de la session

set -e

echo "🧹 NETTOYAGE DES ANCIENS PROCESSUS..."
pkill -f "train_parallel_agents.py" 2>/dev/null || true
sleep 2

echo "📁 CRÉATION DES RÉPERTOIRES DE LOGS..."
mkdir -p /mnt/new_data/adan_logs
mkdir -p /mnt/new_data/adan_logs/archive_$(date +%s)

echo "📝 ARCHIVAGE DES ANCIENS LOGS..."
mv /mnt/new_data/adan_logs/training_*.log /mnt/new_data/adan_logs/archive_$(date +%s)/ 2>/dev/null || true

echo ""
echo "🚀 LANCEMENT DE L'ENTRAÎNEMENT ADAN..."
echo "======================================"

# Créer le fichier de log avec timestamp
LOG_FILE="/mnt/new_data/adan_logs/training_$(date +%Y%m%d_%H%M%S).log"

# Lancer l'entraînement en arrière-plan avec nohup
# nohup = continue même après fermeture du terminal
# disown = détache le processus de la session shell
nohup python scripts/train_parallel_agents.py \
    --config config/config.yaml \
    --log-level INFO \
    --steps 1000000 \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
disown $TRAIN_PID

echo "✅ Entraînement lancé!"
echo "   PID: $TRAIN_PID"
echo "   Log: $LOG_FILE"
echo ""
echo "📊 Vérification du démarrage..."
sleep 5

# Vérifier que les processus sont bien lancés
PROCESS_COUNT=$(ps aux | grep -E "train_parallel_agents.py" | grep -v grep | wc -l)
echo "   Processus actifs: $PROCESS_COUNT (1 principal + 4 workers)"

if [ $PROCESS_COUNT -ge 5 ]; then
    echo "✅ Entraînement démarré avec succès!"
    echo ""
    echo "📈 Premiers logs:"
    sleep 3
    tail -20 "$LOG_FILE" | grep -E "STEP|Portfolio|DBE_DECISION" | head -5
else
    echo "⚠️  Attention: Nombre de processus inférieur à 5"
fi

echo ""
echo "🔍 Pour monitorer l'entraînement:"
echo "   tail -f $LOG_FILE"
echo ""
echo "📊 Pour voir les processus:"
echo "   ps aux | grep train_parallel_agents.py"
echo ""
echo "🛑 Pour arrêter l'entraînement:"
echo "   pkill -f train_parallel_agents.py"
