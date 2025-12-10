#!/bin/bash
# Monitor W3 Optuna Re-optimization en temps réel

LOG_FILE=$(ls -t /tmp/w3_reoptimize_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ Aucun log W3 trouvé"
    exit 1
fi

echo "📊 MONITORING W3 OPTUNA RE-OPTIMIZATION"
echo "========================================"
echo "Log: $LOG_FILE"
echo ""

# Afficher les mises à jour toutes les 5 secondes
while true; do
    clear
    echo "📊 W3 OPTUNA RE-OPTIMIZATION - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    echo ""
    
    # Compter les trials complétés
    COMPLETED=$(grep -c "Trial.*completed" "$LOG_FILE" 2>/dev/null || echo 0)
    RUNNING=$(grep -c "Trial.*running" "$LOG_FILE" 2>/dev/null || echo 0)
    
    echo "📈 PROGRESSION:"
    echo "   Trials complétés: $COMPLETED/10"
    echo "   Trials en cours: $RUNNING"
    echo ""
    
    # Afficher les 5 derniers scores
    echo "🎯 DERNIERS SCORES:"
    grep "Score:" "$LOG_FILE" 2>/dev/null | tail -5 | while read line; do
        echo "   $line"
    done
    echo ""
    
    # Afficher les dernières métriques
    echo "📊 DERNIÈRES MÉTRIQUES:"
    tail -20 "$LOG_FILE" 2>/dev/null | grep -E "Sharpe|Trades|Drawdown|Win" | tail -5 | while read line; do
        echo "   $line"
    done
    echo ""
    
    # Afficher le meilleur score jusqu'à présent
    echo "⭐ MEILLEUR SCORE JUSQU'À PRÉSENT:"
    BEST=$(grep "Score:" "$LOG_FILE" 2>/dev/null | grep -oE "[0-9]+\.[0-9]+" | sort -rn | head -1)
    if [ -n "$BEST" ]; then
        echo "   $BEST"
    else
        echo "   En attente..."
    fi
    echo ""
    
    # Vérifier si terminé
    if [ $COMPLETED -ge 10 ]; then
        echo "✅ RE-OPTIMISATION COMPLÈTE!"
        echo ""
        echo "Résumé final:"
        grep "Score:" "$LOG_FILE" 2>/dev/null | tail -10
        break
    fi
    
    echo "⏳ Mise à jour dans 5 secondes... (Ctrl+C pour arrêter)"
    sleep 5
done
