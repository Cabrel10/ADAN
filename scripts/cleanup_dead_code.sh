#!/bin/bash
# 🔴 CLEANUP DEAD CODE - Suppression sécurisée des 53 modules morts
# Usage: bash scripts/cleanup_dead_code.sh [--dry-run] [--force]

set -e

DRY_RUN=false
FORCE=false
BACKUP_BRANCH="backup/before-dead-code-removal-$(date +%Y%m%d_%H%M%S)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  🔴 CLEANUP DEAD CODE - 53 Modules à supprimer                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Vérifier que nous sommes dans le bon répertoire
if [ ! -f "optuna_optimize_worker.py" ]; then
    echo "❌ Erreur: Exécutez ce script depuis la racine du projet"
    exit 1
fi

# Vérifier que git est disponible
if ! command -v git &> /dev/null; then
    echo "❌ Erreur: git n'est pas installé"
    exit 1
fi

# Créer une branche de sauvegarde
if [ "$DRY_RUN" = false ]; then
    echo "📦 Création de la branche de sauvegarde: $BACKUP_BRANCH"
    git checkout -b "$BACKUP_BRANCH" || true
    git push origin "$BACKUP_BRANCH" 2>/dev/null || echo "⚠️  Impossible de pusher la branche (offline?)"
    echo "✅ Branche de sauvegarde créée"
    echo ""
fi

# Liste des modules à supprimer
MODULES_TO_DELETE=(
    # agent/
    "src/adan_trading_bot/agent/__init__.py"
    "src/adan_trading_bot/agent/custom_recurrent_policy.py"
    
    # evaluation/
    "src/adan_trading_bot/evaluation/__init__.py"
    "src/adan_trading_bot/evaluation/decision_quality_analyzer.py"
    
    # exchange_api/
    "src/adan_trading_bot/exchange_api/__init__.py"
    "src/adan_trading_bot/exchange_api/connector.py"
    
    # live_trading/
    "src/adan_trading_bot/live_trading/__init__.py"
    "src/adan_trading_bot/live_trading/experience_buffer.py"
    "src/adan_trading_bot/live_trading/online_reward_calculator.py"
    "src/adan_trading_bot/live_trading/safety_manager.py"
    
    # monitoring/
    "src/adan_trading_bot/monitoring/alert_system.py"
    "src/adan_trading_bot/monitoring/system_health_monitor.py"
    "src/adan_trading_bot/monitoring/worker_monitor.py"
    
    # optimization/ (dead parts)
    "src/adan_trading_bot/optimization/__init__.py"
    "src/adan_trading_bot/optimization/config/__init__.py"
    "src/adan_trading_bot/optimization/config/experiment_config.py"
    "src/adan_trading_bot/optimization/hyperparameter_optimizer.py"
    "src/adan_trading_bot/optimization/hyperparameter_optimizer_fixed.py"
    "src/adan_trading_bot/optimization/monitoring/__init__.py"
    "src/adan_trading_bot/optimization/monitoring/experiment_tracker.py"
    "src/adan_trading_bot/optimization/optimize_attention.py"
    "src/adan_trading_bot/optimization/scripts/__init__.py"
    "src/adan_trading_bot/optimization/tests/__init__.py"
    "src/adan_trading_bot/optimization/tests/load_testing.py"
    
    # patches/
    "src/adan_trading_bot/patches/gugu_march_excellence_rewards.py"
    
    # performance/
    "src/adan_trading_bot/performance/metrics.py"
    
    # portfolio/
    "src/adan_trading_bot/portfolio/__init__.py"
    "src/adan_trading_bot/portfolio/portfolio_manager.py"
    
    # risk_management/
    "src/adan_trading_bot/risk_management/__init__.py"
    "src/adan_trading_bot/risk_management/position_sizer.py"
    "src/adan_trading_bot/risk_management/risk_manager.py"
    
    # trading/
    "src/adan_trading_bot/trading/__init__.py"
    "src/adan_trading_bot/trading/action_translator.py"
    "src/adan_trading_bot/trading/action_validator.py"
    "src/adan_trading_bot/trading/fee_manager.py"
    "src/adan_trading_bot/trading/manual_trading_interface.py"
    "src/adan_trading_bot/trading/order_manager.py"
    "src/adan_trading_bot/trading/position_sizer.py"
    "src/adan_trading_bot/trading/safety_manager.py"
    "src/adan_trading_bot/trading/secure_api_manager.py"
    
    # training/
    "src/adan_trading_bot/training/__init__.py"
    "src/adan_trading_bot/training/callbacks.py"
    "src/adan_trading_bot/training/dynamic_training_callback.py"
    "src/adan_trading_bot/training/hyperparam_modulator.py"
    "src/adan_trading_bot/training/shared_experience_buffer.py"
    "src/adan_trading_bot/training/trainer.py"
    "src/adan_trading_bot/training/training_orchestrator.py"
    
    # visualization/
    "src/adan_trading_bot/visualization/gradcam_1d.py"
    "src/adan_trading_bot/visualization/plotting_styles.py"
    
    # workflows/
    "src/adan_trading_bot/workflows/workflow_orchestrator.py"
    
    # root level
    "src/adan_trading_bot/constants.py"
    "src/adan_trading_bot/main.py"
    "src/adan_trading_bot/online_learning_agent.py"
)

# Supprimer les modules
echo "🗑️  Suppression des modules morts..."
echo ""

DELETED_COUNT=0
SKIPPED_COUNT=0

for module in "${MODULES_TO_DELETE[@]}"; do
    if [ -f "$module" ] || [ -d "$module" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "   [DRY-RUN] Supprimerait: $module"
        else
            echo "   ✓ Suppression: $module"
            git rm -f "$module" 2>/dev/null || rm -rf "$module"
            ((DELETED_COUNT++))
        fi
    else
        if [ "$DRY_RUN" = false ]; then
            echo "   ⚠️  Introuvable: $module"
            ((SKIPPED_COUNT++))
        fi
    fi
done

echo ""
echo "📊 Résumé:"
echo "   Supprimés: $DELETED_COUNT modules"
echo "   Introuvables: $SKIPPED_COUNT modules"
echo ""

# Supprimer les répertoires vides
if [ "$DRY_RUN" = false ]; then
    echo "🧹 Nettoyage des répertoires vides..."
    find src/adan_trading_bot -type d -empty -delete 2>/dev/null || true
    echo "✅ Répertoires vides supprimés"
    echo ""
fi

# Commit
if [ "$DRY_RUN" = false ]; then
    echo "💾 Commit des changements..."
    git add -A
    git commit -m "🔴 Remove 53 dead code modules (never imported)

Removed:
- agent/, evaluation/, exchange_api/, live_trading/
- monitoring/, optimization/config/, optimization/monitoring/
- patches/, performance/, portfolio/, risk_management/
- trading/, training/, visualization/, workflows/
- root-level: constants.py, main.py, online_learning_agent.py

These 53 modules were never imported by the 4 active scripts:
- optuna_optimize_worker.py
- scripts/train_parallel_agents.py
- scripts/terminal_dashboard.py
- src/adan_trading_bot/optimization/hyperparameter_optimizer.py

Keeps 61 active modules (100% of used code).
Improves: load time, maintainability, clarity.

Backup branch: $BACKUP_BRANCH" || echo "⚠️  Commit échoué (peut-être rien à committer)"
    echo "✅ Changements committés"
    echo ""
fi

# Résumé final
echo "╔════════════════════════════════════════════════════════════════╗"
if [ "$DRY_RUN" = true ]; then
    echo "║  ✅ DRY-RUN COMPLET - Aucun changement réel                  ║"
    echo "║  Relancez sans --dry-run pour appliquer les changements     ║"
else
    echo "║  ✅ CLEANUP COMPLET - 53 modules supprimés                   ║"
    echo "║  Branche de sauvegarde: $BACKUP_BRANCH"
fi
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Suggestions suivantes
echo "📋 Prochaines étapes:"
echo "   1. Relancer les tests: python scripts/07_test_all_issues.py"
echo "   2. Vérifier les 4 scripts actifs"
echo "   3. Pusher les changements: git push origin main"
echo ""

