#!/bin/bash

# 🔍 PHASE 2 READINESS CHECK
# Verifies all systems are ready for Phase 2 deployment

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           🔍 PHASE 2 READINESS CHECK                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

CHECKS_PASSED=0
CHECKS_FAILED=0

# Function to check item
check_item() {
    local name=$1
    local condition=$2
    
    if eval "$condition"; then
        echo "✅ $name"
        ((CHECKS_PASSED++))
    else
        echo "❌ $name"
        ((CHECKS_FAILED++))
    fi
}

echo "📋 INFRASTRUCTURE CHECKS"
echo "═══════════════════════════════════════════════════════════════"

check_item "Checkpoint directory exists" "[ -d /mnt/new_data/t10_training/checkpoints ]"
check_item "Phase 2 output directory" "[ -d /mnt/new_data/t10_training/phase2_results ] || mkdir -p /mnt/new_data/t10_training/phase2_results"
check_item "Scripts directory" "[ -d ./scripts ]"
check_item "Config directory" "[ -d ./config ]"

echo ""
echo "📊 SCRIPT CHECKS"
echo "═══════════════════════════════════════════════════════════════"

check_item "Phase 2 Orchestrator" "[ -f ./scripts/phase2_orchestrator.py ]"
check_item "Paper Trading Monitor" "[ -f ./scripts/paper_trading_monitor.py ]"
check_item "Phase 2 Launcher" "[ -f ./launch_phase2_pipeline.sh ]"
check_item "Phase 2 Config" "[ -f ./config/phase2_config.yaml ]"

echo ""
echo "🔧 PYTHON DEPENDENCIES"
echo "═══════════════════════════════════════════════════════════════"

check_item "Python 3 installed" "command -v python3 &> /dev/null"
check_item "PyYAML available" "python3 -c 'import yaml' 2>/dev/null"
check_item "Pandas available" "python3 -c 'import pandas' 2>/dev/null"
check_item "NumPy available" "python3 -c 'import numpy' 2>/dev/null"

echo ""
echo "📈 TRAINING STATUS"
echo "═══════════════════════════════════════════════════════════════"

for worker in w1 w2 w3 w4; do
    latest=$(ls -t /mnt/new_data/t10_training/checkpoints/$worker/${worker}_model_*.zip 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        steps=$(basename "$latest" | grep -oP '\d+(?=_steps)' | tail -1)
        if [ "$steps" -ge 350000 ]; then
            echo "✅ $worker: Complete ($steps steps)"
            ((CHECKS_PASSED++))
        else
            echo "⏳ $worker: In progress ($steps/350000 steps)"
        fi
    else
        echo "❌ $worker: No checkpoint found"
        ((CHECKS_FAILED++))
    fi
done

echo ""
echo "🔐 PERMISSIONS CHECK"
echo "═══════════════════════════════════════════════════════════════"

check_item "Launcher executable" "[ -x ./launch_phase2_pipeline.sh ] || chmod +x ./launch_phase2_pipeline.sh"
check_item "Orchestrator executable" "[ -x ./scripts/phase2_orchestrator.py ] || chmod +x ./scripts/phase2_orchestrator.py"
check_item "Monitor executable" "[ -x ./scripts/paper_trading_monitor.py ] || chmod +x ./scripts/paper_trading_monitor.py"

echo ""
echo "📊 SUMMARY"
echo "═══════════════════════════════════════════════════════════════"
echo "Checks Passed: $CHECKS_PASSED"
echo "Checks Failed: $CHECKS_FAILED"
echo ""

if [ $CHECKS_FAILED -eq 0 ]; then
    echo "✅ ALL CHECKS PASSED - READY FOR PHASE 2"
    echo ""
    echo "🚀 To launch Phase 2 pipeline:"
    echo "   chmod +x launch_phase2_pipeline.sh"
    echo "   ./launch_phase2_pipeline.sh"
    echo ""
    exit 0
else
    echo "❌ SOME CHECKS FAILED - PLEASE FIX ISSUES BEFORE PROCEEDING"
    exit 1
fi
