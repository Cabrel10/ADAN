#!/bin/bash

# Script de monitoring en temps réel pour Optuna

echo "🔍 MONITORING OPTUNA W3"
echo "======================="

while true; do
    clear
    echo "⏰ $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Nombre de trials terminés
    TRIALS_DONE=$(grep -c "Trial [0-9]* finished" optuna_w3_final.log 2>/dev/null || echo "0")
    echo "📊 Trials terminés: $TRIALS_DONE/20"
    
    # Dernier trial
    LAST_TRIAL=$(grep "Trial [0-9]* finished" optuna_w3_final.log 2>/dev/null | tail -1 | grep -oP "Trial \d+" | tail -1)
    if [ ! -z "$LAST_TRIAL" ]; then
        LAST_SCORE=$(grep "Trial [0-9]* finished" optuna_w3_final.log 2>/dev/null | tail -1 | grep -oP "value: \K[-0-9.]+")
        echo "🏆 Dernier: $LAST_TRIAL | Score: $LAST_SCORE"
    fi
    
    # Meilleur trial
    BEST_TRIAL=$(grep "Best is trial" optuna_w3_final.log 2>/dev/null | tail -1 | grep -oP "trial \d+" | tail -1)
    if [ ! -z "$BEST_TRIAL" ]; then
        BEST_SCORE=$(grep "Best is trial" optuna_w3_final.log 2>/dev/null | tail -1 | grep -oP "value: \K[-0-9.]+")
        echo "⭐ Meilleur: $BEST_TRIAL | Score: $BEST_SCORE"
    fi
    
    echo ""
    echo "📈 Métriques du dernier trial:"
    
    # Nombre de trades
    TRADES=$(grep "Trades complétés:" optuna_w3_final.log 2>/dev/null | tail -1 | grep -oP "Trades complétés: \K[0-9]+")
    echo "   Trades: $TRADES"
    
    # Win rate
    WR=$(grep "Win Rate:" optuna_w3_final.log 2>/dev/null | tail -1 | grep -oP "Win Rate: \K[0-9.%]+")
    echo "   Win Rate: $WR"
    
    # PnL
    PNL=$(grep "PnL:" optuna_w3_final.log 2>/dev/null | tail -1 | grep -oP "PnL: \K\$[-0-9.]+")
    echo "   PnL: $PNL"
    
    # Portfolio Growth
    GROWTH=$(grep "Portfolio Growth:" optuna_w3_final.log 2>/dev/null | tail -1 | grep -oP "Portfolio Growth: \K[-0-9.%]+")
    echo "   Portfolio Growth: $GROWTH"
    
    echo ""
    echo "⚡ Vitesse:"
    
    # Calculer it/s
    FIRST_STEP=$(grep "Starting step 1" optuna_w3_final.log 2>/dev/null | head -1 | grep -oP "\d{2}:\d{2}:\d{2}" | head -1)
    LAST_STEP=$(grep "Starting step" optuna_w3_final.log 2>/dev/null | tail -1 | grep -oP "\d{2}:\d{2}:\d{2}" | tail -1)
    
    if [ ! -z "$FIRST_STEP" ] && [ ! -z "$LAST_STEP" ]; then
        FIRST_SEC=$(date -d "2025-12-06 $FIRST_STEP" +%s 2>/dev/null || echo "0")
        LAST_SEC=$(date -d "2025-12-06 $LAST_STEP" +%s 2>/dev/null || echo "0")
        
        if [ "$LAST_SEC" -gt "$FIRST_SEC" ]; then
            ELAPSED=$((LAST_SEC - FIRST_SEC))
            LAST_STEP_NUM=$(grep "Starting step" optuna_w3_final.log 2>/dev/null | tail -1 | grep -oP "step \K[0-9]+")
            
            if [ ! -z "$LAST_STEP_NUM" ] && [ "$ELAPSED" -gt "0" ]; then
                ITS=$(echo "scale=2; $LAST_STEP_NUM / $ELAPSED" | bc)
                echo "   it/s: $ITS"
                echo "   Temps écoulé: ${ELAPSED}s"
                echo "   Steps: $LAST_STEP_NUM"
            fi
        fi
    fi
    
    echo ""
    echo "🔄 Appuyez sur Ctrl+C pour arrêter"
    sleep 10
done
