#!/bin/bash

# ============================================================================
# QUICK RELAUNCH SCRIPT - OPTUNA ZERO METRICS FIX
# ============================================================================
# Usage: bash QUICK_RELAUNCH.sh [phase]
# Phases: 1 (validation), 2 (mini optuna), 3 (optuna complet)
# ============================================================================

set -e

PHASE=${1:-1}
BOT_DIR="/home/morningstar/Documents/trading/bot"
CONDA_ENV="trading_env"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# FUNCTIONS
# ============================================================================

print_header() {
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# ============================================================================
# PHASE 1: VALIDATION
# ============================================================================

phase_1_validation() {
    print_header "PHASE 1: VALIDATION (5 min)"
    
    cd "$BOT_DIR"
    source ~/miniconda3/bin/activate $CONDA_ENV
    
    print_info "Lancement du test d'intégration..."
    python scripts/test_optuna_training_only.py
    
    if [ $? -eq 0 ]; then
        print_success "Test d'intégration réussi!"
        print_info "Résultats attendus:"
        print_info "  ✅ metrics.trades > 0"
        print_info "  ✅ metrics.closed_positions > 0"
        print_info "  ✅ total_trades > 0"
        print_info "  ✅ sharpe_ratio != 0"
    else
        print_error "Test d'intégration échoué!"
        exit 1
    fi
}

# ============================================================================
# PHASE 2: MINI OPTUNA
# ============================================================================

phase_2_mini_optuna() {
    print_header "PHASE 2: MINI OPTUNA (2h)"
    
    cd "$BOT_DIR"
    source ~/miniconda3/bin/activate $CONDA_ENV
    
    for worker in W1 W2 W3 W4; do
        print_info "Lancement Optuna pour $worker (2 trials)..."
        python optuna_optimize_ppo.py --worker $worker --trials 2 --steps 3000
        
        if [ $? -eq 0 ]; then
            print_success "$worker terminé"
        else
            print_error "$worker échoué"
            exit 1
        fi
    done
    
    print_header "RÉSULTATS MINI OPTUNA"
    for worker in W1 W2 W3 W4; do
        echo -e "${YELLOW}=== $worker ===${NC}"
        cat optuna_results/${worker}_ppo_best_params.yaml | grep -E "score:|sharpe:|trades:|drawdown:|win_rate:" || echo "Fichier non trouvé"
    done
}

# ============================================================================
# PHASE 3: OPTUNA COMPLET
# ============================================================================

phase_3_optuna_complet() {
    print_header "PHASE 3: OPTUNA COMPLET (4-8h)"
    
    cd "$BOT_DIR"
    source ~/miniconda3/bin/activate $CONDA_ENV
    
    print_info "Lancement Optuna complet pour tous les workers..."
    print_info "Ceci prendra 4-8 heures. Les logs seront sauvegardés dans /mnt/new_data/adan_logs/"
    
    for worker in W1 W2 W3 W4; do
        print_info "Lancement Optuna pour $worker (100 trials)..."
        python optuna_optimize_ppo.py --worker $worker --trials 100 --steps 5000
        
        if [ $? -eq 0 ]; then
            print_success "$worker terminé"
        else
            print_error "$worker échoué"
            exit 1
        fi
    done
    
    print_header "OPTUNA COMPLET TERMINÉ"
    print_success "Tous les workers ont été optimisés!"
    
    print_info "Résultats finaux:"
    for worker in W1 W2 W3 W4; do
        echo -e "${YELLOW}=== $worker ===${NC}"
        cat optuna_results/${worker}_ppo_best_params.yaml | grep -E "score:|sharpe:|trades:" || echo "Fichier non trouvé"
    done
}

# ============================================================================
# MAIN
# ============================================================================

print_header "OPTUNA ZERO METRICS FIX - QUICK RELAUNCH"

case $PHASE in
    1)
        phase_1_validation
        print_success "Phase 1 terminée!"
        print_info "Prochaine étape: bash QUICK_RELAUNCH.sh 2"
        ;;
    2)
        phase_1_validation
        phase_2_mini_optuna
        print_success "Phase 2 terminée!"
        print_info "Prochaine étape: bash QUICK_RELAUNCH.sh 3"
        ;;
    3)
        phase_1_validation
        phase_2_mini_optuna
        phase_3_optuna_complet
        print_success "Phase 3 terminée!"
        print_info "Optuna complet est terminé!"
        ;;
    all)
        phase_1_validation
        phase_2_mini_optuna
        phase_3_optuna_complet
        print_success "Toutes les phases terminées!"
        ;;
    *)
        print_error "Phase invalide: $PHASE"
        echo "Usage: bash QUICK_RELAUNCH.sh [1|2|3|all]"
        echo "  1   = Validation (5 min)"
        echo "  2   = Mini Optuna (2h)"
        echo "  3   = Optuna Complet (4-8h)"
        echo "  all = Toutes les phases"
        exit 1
        ;;
esac

print_header "✅ SUCCÈS"
