#!/bin/bash

echo "🚀 LANCEMENT T10 EN ARRIÈRE-PLAN AVEC NOHUP"
echo "==========================================="

cd /home/morningstar/Documents/trading/bot

# Créer les répertoires
mkdir -p /mnt/new_data/t10_training/logs
mkdir -p /mnt/new_data/t10_training/checkpoints

# Lancer l'entraînement en arrière-plan avec nohup
echo ""
echo "📌 Lancement de l'entraînement..."
nohup python scripts/train_parallel_agents.py \
    --config config/config.yaml \
    --output-dir /mnt/new_data/t10_training \
    --workers 4 \
    --steps 250000 \
    > /mnt/new_data/t10_training/logs/t10_main.log 2>&1 &

TRAIN_PID=$!
echo "✅ Entraînement lancé (PID: $TRAIN_PID)"
echo $TRAIN_PID > /mnt/new_data/t10_training/t10_train.pid

# Lancer le monitoring en arrière-plan
echo ""
echo "📌 Lancement du monitoring..."
nohup python monitor_t10_longterm.py \
    > /mnt/new_data/t10_training/logs/monitoring.log 2>&1 &

MONITOR_PID=$!
echo "✅ Monitoring lancé (PID: $MONITOR_PID)"
echo $MONITOR_PID > /mnt/new_data/t10_training/t10_monitor.pid

echo ""
echo "🎯 Commandes de suivi:"
echo "  - Voir le log principal: tail -f /mnt/new_data/t10_training/logs/t10_main.log"
echo "  - Voir le monitoring: tail -f /mnt/new_data/t10_training/logs/monitoring.log"
echo "  - Vérifier les processus: ps aux | grep train_parallel"
echo ""
echo "✅ T10 lancé en arrière-plan!"
