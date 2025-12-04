#!/bin/bash
# Script pour lancer l'optimisation des 4 workers séquentiellement
# Usage: bash run_optuna_all_workers.sh

set -e  # Exit on error

echo "=========================================="
echo "OPTUNA OPTIMIZATION - 4 WORKERS SÉQUENTIELS"
echo "=========================================="
echo ""
echo "Config: 20 trials × 3000 steps par worker"
echo "Temps estimé: ~5 heures total"
echo ""

# Activate conda env
source $(conda info --base)/etc/profile.d/conda.sh
conda activate trading_env

# Create output directory
mkdir -p optuna_results

# W1: Scalper
echo "=========================================="
echo "WORKER 1 - SCALPER (11-15 trades/jour)"
echo "=========================================="
python optuna_optimize_worker.py --worker W1 --trials 20 --steps 3000
echo ""
echo "W1 terminé. Résultats dans optuna_results/W1_best_params.yaml"
echo ""
sleep 5

# W2: Swing
echo "=========================================="
echo "WORKER 2 - SWING (3-5 trades/jour)"
echo "=========================================="
python optuna_optimize_worker.py --worker W2 --trials 20 --steps 3000
echo ""
echo "W2 terminé. Résultats dans optuna_results/W2_best_params.yaml"
echo ""
sleep 5

# W3: Trend
echo "=========================================="
echo "WORKER 3 - TREND (1-3 trades/jour)"
echo "=========================================="
python optuna_optimize_worker.py --worker W3 --trials 20 --steps 3000
echo ""
echo "W3 terminé. Résultats dans optuna_results/W3_best_params.yaml"
echo ""
sleep 5

# W4: Market Making
echo "=========================================="
echo "WORKER 4 - MARKET MAKING (15-25 trades/jour)"
echo "=========================================="
python optuna_optimize_worker.py --worker W4 --trials 20 --steps 3000
echo ""
echo "W4 terminé. Résultats dans optuna_results/W4_best_params.yaml"
echo ""

echo "=========================================="
echo "OPTIMISATION COMPLÈTE !"
echo "=========================================="
echo ""
echo "Résultats:"
ls -lh optuna_results/*_best_params.yaml
echo ""
echo "Databases:"
ls -lh optuna_results/*.db
echo ""
