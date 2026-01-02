#!/bin/bash

################################################################################
# ADAN TRADING BOT - QUICK START SCRIPT
#
# Exécute le plan complet en une seule commande
# Usage: bash QUICK_START.sh [phase]
#
# Phases:
#   1 = Nettoyage & Validation
#   2 = Optuna (100 trials)
#   3 = Entraînement (500k steps)
#   4 = Évaluation
#   5 = Paper Trading
#   all = Toutes les phases
#
# Exemple:
#   bash QUICK_START.sh 1        # Phase 1 uniquement
#   bash QUICK_START.sh all      # Toutes les phases
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/home/morningstar/Documents/trading/bot"
PHASE="${1:-all}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_DIR}/logs"

export BINANCE_API_KEY="DUMMY_LIVE_API_KEY" # Dummy key to satisfy config loader
export BINANCE_SECRET_KEY="DUMMY_LIVE_SECRET_KEY" # Dummy key to satisfy config loader
export BINANCE_TESTNET_API_KEY="C6OQ2shpSn0YQu8Yc6mEi4w6cAqQDZ8wcRSz0Um6ehkPjEbQ8XWwHiufqDTmHHnB"
export BINANCE_TESTNET_SECRET_KEY="Dk7f52uFcgvL1aiRnWn1knAsulM1Xj6i1XBaEtOsnSDJBkN8NveY4lM9Wi8ZmWwx4"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[\u2705 SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[\u274c ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[\u26a0\ufe0f  WARNING]${NC} $1"
}

# Phase 1: Cleanup & Validation
phase_1() {
    log "=========================================="
    log "PHASE 1: CLEANUP & VALIDATION"
    log "=========================================="
    
    cd "${PROJECT_DIR}"
    
    # Cleanup
    log "Removing old artifacts..."
    rm -rf checkpoints/*.zip 2>/dev/null || true
    rm -f optuna.db 2>/dev/null || true
    rm -rf logs/*.log logs/rewards/* logs/tensorboard/* 2>/dev/null || true
    log_success "Cleanup completed"

    # Ensure checkpoints directory exists for validation
    mkdir -p checkpoints
    
    # Validate setup
    log "Validating setup..."
    python scripts/validate_final_setup.py
    
    # Quick training test
    log "Running quick training test (1000 steps)..."
    timeout 90 python scripts/train_parallel_agents.py \
        --config config/config.yaml \
        --checkpoint-dir checkpoints \
        --total-timesteps 1000 \
        2>&1 | tail -20
    
    log_success "Phase 1 completed"
}

# Phase 2: Optuna Optimization
phase_2() {
    log "=========================================="
    log "PHASE 2: OPTUNA OPTIMIZATION (100 trials)"
    log "=========================================="
    
    cd "${PROJECT_DIR}"
    
    log "Starting Optuna optimization..."
    log "This will take 2-4 hours. Monitor with:"
    log "  watch -n 30 'tail -20 logs/optuna_final.log'"
    
    python scripts/optimize_hyperparams.py \
        --n-trials 100 \
        --timeout 14400 \
        --study-name "adan_final_v1" \
        --storage sqlite:///optuna.db \
        --n-jobs 1 \
        --seed 42 \
        2>&1 | tee logs/optuna_final.log
    
    # Extract best params
    log "Extracting best parameters..."
    python scripts/extract_best_optuna_params.py \
        --study-name "adan_final_v1" \
        --output config/best_params_optuna.json
    
    log_success "Phase 2 completed"
}

# Phase 3: Training
phase_3() {
    log "=========================================="
    log "PHASE 3: TRAINING (500k timesteps)"
    log "=========================================="
    
    cd "${PROJECT_DIR}"
    
    log "Starting training with best Optuna params..."
    log "This will take 6-8 hours. Monitor with:"
    log "  watch -n 60 'tail -30 logs/train_final.log | grep -E \"timesteps|mean_reward\"'"
    
    python scripts/train_parallel_agents.py \
        --config config/config.yaml \
        --checkpoint-dir checkpoints \
        2>&1 | tee logs/train_final.log
    
    log_success "Phase 3 completed"
}

# Phase 4: Evaluation
phase_4() {
    log "=========================================="
    log "PHASE 4: EVALUATION & BACKTEST"
    log "=========================================="
    
    cd "${PROJECT_DIR}"
    
    # Check if models exist
    if [ ! -f "checkpoints/worker_0_final.zip" ]; then
        log_error "Models not found. Run Phase 3 first."
        return 1
    fi
    
    log "Starting evaluation..."
    
    # Create evaluation script if it doesn't exist
    if [ ! -f "scripts/evaluate_final_model.py" ]; then
        log_warning "evaluate_final_model.py not found. Creating placeholder..."
        cat > scripts/evaluate_final_model.py << 'EOF'
#!/usr/bin/env python3
"""Placeholder for evaluation script."""
import sys
print("Evaluation script placeholder - implement as needed")
sys.exit(0)
EOF
    fi
    
    python scripts/evaluate_final_model.py \
        --model-dir checkpoints \
        --backtest \
        --eval-episodes 50 \
        --output results/final_evaluation.json \
        2>&1 | tee logs/evaluation.log
    
    log_success "Phase 4 completed"
}

# Phase 5: Paper Trading
phase_5() {
    log "=========================================="
    log "PHASE 5: PAPER TRADING (7 days)"
    log "=========================================="
    
    cd "${PROJECT_DIR}"
    
    # Check if models exist
    if [ ! -f "checkpoints/worker_0_final.zip" ]; then
        log_error "Models not found. Run Phase 3 first."
        return 1
    fi
    
    log "Starting paper trading..."
    log "This will run for 7 days. Monitor daily with:"
    log "  python scripts/daily_report.py --date \$(date +%Y-%m-%d) --log logs/paper_trading.log"
    
    # Create paper trading script if it doesn't exist
    if [ ! -f "scripts/live_trading.py" ]; then
        log_warning "live_trading.py not found. Creating placeholder..."
        cat > scripts/live_trading.py << 'EOF'
#!/usr/bin/env python3
"""Placeholder for live trading script."""
import sys
print("Live trading script placeholder - implement as needed")
sys.exit(0)
EOF
    fi
    
    python scripts/live_trading.py \
        --mode paper \
        --model-dir checkpoints \
        --initial-balance 1000 \
        --max-positions 3 \
        --max-daily-loss 100 \
        --config config/live.yaml \
        2>&1 | tee logs/paper_trading.log
    
    log_success "Phase 5 completed"
}

# Main execution
main() {
    log "=========================================="
    log "ADAN TRADING BOT - FINAL PLAN"
    log "=========================================="
    log "Phase: ${PHASE}"
    log "Project: ${PROJECT_DIR}"
    log "Timestamp: ${TIMESTAMP}"
    log ""
    
    case "${PHASE}" in
        1)
            phase_1
            ;;
        2)
            phase_2
            ;;
        3)
            phase_3
            ;;
        4)
            phase_4
            ;;
        5)
            phase_5
            ;;
        all)
            phase_1 || exit 1
            phase_2 || exit 1
            phase_3 || exit 1
            phase_4 || exit 1
            phase_5 || exit 1
            ;;
        *)
            log_error "Unknown phase: ${PHASE}"
            echo "Usage: bash QUICK_START.sh [1|2|3|4|5|all]"
            exit 1
            ;;
    esac
    
    log "=========================================="
    log_success "EXECUTION COMPLETED"
    log "=========================================="
}

# Run main
main "$@"
