#!/bin/bash
# Vérification rapide du statut de l'entraînement

echo "🎯 STATUT ENTRAÎNEMENT ADAN"
echo "================================"
echo "Heure: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Processus
PROCS=$(ps aux | grep -E "train_parallel|python.*train" | grep -v grep)
if [ -z "$PROCS" ]; then
    echo "❌ Aucun processus d'entraînement actif"
else
    echo "✅ Processus actifs:"
    echo "$PROCS" | awk '{print "   PID " $2 ": " $11 " " $12}'
fi

echo ""

# Logs
LOG_FILE=$(ls -t /mnt/new_data/adan_logs/training_*.log 2>/dev/null | head -1)
if [ -n "$LOG_FILE" ]; then
    SIZE=$(du -h "$LOG_FILE" | awk '{print $1}')
    LINES=$(wc -l < "$LOG_FILE")
    echo "📁 Log: $(basename $LOG_FILE)"
    echo "   Taille: $SIZE"
    echo "   Lignes: $LINES"
    echo ""
    echo "📈 Derniers logs:"
    tail -5 "$LOG_FILE" | grep -E "STEP|TRADE|Worker|Portfolio" || tail -5 "$LOG_FILE"
else
    echo "❌ Aucun fichier log trouvé"
fi

echo ""
echo "💾 Espace disque:"
df -h /mnt/new_data/ | tail -1 | awk '{printf "   Utilisé: %s, Libre: %s\n", $3, $4}'
