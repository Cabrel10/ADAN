#!/bin/bash
# Script pour lancer l'optimisation Optuna séquentielle pour tous les workers

set -e

WORKERS=("W1" "W2" "W3" "W4")
TRIALS=20
STEPS=5000
EVAL_STEPS=2000
OUTPUT_DIR="optuna_results"

echo "╔════════════════════════════════════════════════════════════════════════════════════════════════╗"
echo "║                    T8 : OPTIMISATION OPTUNA SÉQUENTIELLE                                      ║"
echo "║                         4 Workers × 20 Trials = 80 Trials Total                              ║"
echo "╚════════════════════════════════════════════════════════════════════════════════════════════════╝"
echo ""

START_TIME=$(date +%s)

for WORKER in "${WORKERS[@]}"; do
    WORKER_START=$(date +%s)
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 LANCEMENT : $WORKER"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Trials: $TRIALS | Steps: $STEPS | Eval Steps: $EVAL_STEPS"
    echo "Démarrage : $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Lancer l'optimisation
    python optuna_optimize_ppo.py \
        --worker "$WORKER" \
        --trials "$TRIALS" \
        --steps "$STEPS" \
        --eval-steps "$EVAL_STEPS" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR/${WORKER}_optimization.log"
    
    WORKER_END=$(date +%s)
    WORKER_DURATION=$((WORKER_END - WORKER_START))
    WORKER_MINUTES=$((WORKER_DURATION / 60))
    
    echo ""
    echo "✅ $WORKER COMPLÉTÉ"
    echo "   Durée : ${WORKER_MINUTES} minutes"
    echo ""
    
    # Vérifier les résultats
    if [ -f "$OUTPUT_DIR/${WORKER}_ppo_best_params.yaml" ]; then
        echo "📊 Résultats sauvegardés : $OUTPUT_DIR/${WORKER}_ppo_best_params.yaml"
        echo ""
    fi
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))
TOTAL_HOURS=$((TOTAL_MINUTES / 60))

echo "╔════════════════════════════════════════════════════════════════════════════════════════════════╗"
echo "║                         T8 COMPLÉTÉ - TOUS LES WORKERS OPTIMISÉS                             ║"
echo "╚════════════════════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 RÉSUMÉ FINAL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Durée totale : ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
echo "Fichiers générés :"
ls -lh "$OUTPUT_DIR"/*_ppo_best_params.yaml 2>/dev/null || echo "  (Aucun fichier trouvé)"
echo ""
echo "✅ T8 SUCCÈS - Prêt pour T9 (Injection dans config.yaml)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
