#!/bin/bash
# Script de monitoring en temps réel de l'entraînement

echo "🚀 MONITORING ENTRAÎNEMENT EN TEMPS RÉEL"
echo "========================================"
echo ""

# Trouver le dernier fichier log
LATEST_LOG=$(ls -t /mnt/new_data/adan_logs/training_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ Aucun fichier log trouvé"
    exit 1
fi

echo "📊 Fichier log: $LATEST_LOG"
echo ""

# Fonction pour afficher les stats
show_stats() {
    echo "📈 STATISTIQUES ACTUELLES:"
    echo ""
    
    # Nombre de steps
    STEPS=$(grep -o "total_timesteps [0-9]*" "$LATEST_LOG" | tail -1 | awk '{print $2}')
    if [ ! -z "$STEPS" ]; then
        echo "  ✅ Steps complétés: $STEPS / 500000"
        PERCENT=$((STEPS * 100 / 500000))
        echo "  📊 Progression: $PERCENT%"
    fi
    
    # Vérifier les NaN
    NaN_COUNT=$(grep -c "NaN\|nan" "$LATEST_LOG")
    if [ "$NaN_COUNT" -gt 0 ]; then
        echo "  ❌ NaN détectés: $NaN_COUNT"
    else
        echo "  ✅ Aucun NaN détecté"
    fi
    
    # Vérifier les erreurs
    ERROR_COUNT=$(grep -c "ERROR\|CRITICAL" "$LATEST_LOG")
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo "  ⚠️  Erreurs: $ERROR_COUNT"
        echo ""
        echo "  Dernières erreurs:"
        grep "ERROR\|CRITICAL" "$LATEST_LOG" | tail -3 | sed 's/^/    /'
    else
        echo "  ✅ Aucune erreur"
    fi
    
    # Dernière ligne du log
    echo ""
    echo "  📝 Dernière activité:"
    tail -1 "$LATEST_LOG" | sed 's/^/    /'
}

# Afficher les stats toutes les 30 secondes
while true; do
    clear
    echo "🚀 MONITORING ENTRAÎNEMENT EN TEMPS RÉEL"
    echo "========================================"
    echo "Heure: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    show_stats
    
    echo ""
    echo "⏱️  Mise à jour dans 30 secondes... (Ctrl+C pour arrêter)"
    sleep 30
done
