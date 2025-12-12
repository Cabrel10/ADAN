#!/bin/bash

# 🚀 PHASE 2 PIPELINE LAUNCHER
# Automated Evaluation, Backtesting & Paper Trading
# Launches automatically when training completes

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         🚀 PHASE 2 PIPELINE LAUNCHER - AUTOMATED              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
CHECKPOINT_DIR="/mnt/new_data/t10_training/checkpoints"
PHASE2_DIR="/mnt/new_data/t10_training/phase2_results"
SCRIPTS_DIR="./scripts"
TARGET_STEPS=350000
CHECK_INTERVAL=300  # 5 minutes

# Create output directory
mkdir -p "$PHASE2_DIR"

echo "📋 CONFIGURATION"
echo "═══════════════════════════════════════════════════════════════"
echo "Checkpoint Directory: $CHECKPOINT_DIR"
echo "Phase 2 Output: $PHASE2_DIR"
echo "Target Steps: $TARGET_STEPS"
echo "Check Interval: $CHECK_INTERVAL seconds"
echo ""

# Function to check training completion
check_training_completion() {
    echo "🔍 Checking training completion..."
    
    all_complete=true
    for worker in w1 w2 w3 w4; do
        latest=$(ls -t "$CHECKPOINT_DIR/$worker/${worker}_model_"*.zip 2>/dev/null | head -1)
        if [ -z "$latest" ]; then
            echo "  ⏳ $worker: No checkpoint found"
            all_complete=false
            continue
        fi
        
        steps=$(basename "$latest" | grep -oP '\d+(?=_steps)' | tail -1)
        if [ "$steps" -ge "$TARGET_STEPS" ]; then
            echo "  ✅ $worker: Complete ($steps steps)"
        else
            echo "  ⏳ $worker: In progress ($steps/$TARGET_STEPS steps)"
            all_complete=false
        fi
    done
    
    return $([ "$all_complete" = true ] && echo 0 || echo 1)
}

# Function to run Phase 2 pipeline
run_phase2_pipeline() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║              🚀 LAUNCHING PHASE 2 PIPELINE                    ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Run orchestrator
    echo "📊 Running Phase 2 Orchestrator..."
    python3 "$SCRIPTS_DIR/phase2_orchestrator.py"
    
    if [ $? -ne 0 ]; then
        echo "❌ Phase 2 Orchestrator failed"
        return 1
    fi
    
    echo ""
    echo "✅ Phase 2 Orchestrator complete"
    echo ""
    
    # Start paper trading monitor
    echo "📈 Starting Paper Trading Monitor..."
    nohup python3 "$SCRIPTS_DIR/paper_trading_monitor.py" \
        > "$PHASE2_DIR/monitor.log" 2>&1 &
    
    MONITOR_PID=$!
    echo "✅ Paper Trading Monitor started (PID: $MONITOR_PID)"
    echo "   Log: $PHASE2_DIR/monitor.log"
    echo ""
    
    return 0
}

# Main loop
echo "⏳ WAITING FOR TRAINING COMPLETION"
echo "═══════════════════════════════════════════════════════════════"
echo ""

iteration=0
while true; do
    iteration=$((iteration + 1))
    
    echo "Check #$iteration - $(date '+%Y-%m-%d %H:%M:%S')"
    
    if check_training_completion; then
        echo ""
        echo "✅ All workers have completed training!"
        echo ""
        
        # Run Phase 2 pipeline
        if run_phase2_pipeline; then
            echo ""
            echo "╔════════════════════════════════════════════════════════════════╗"
            echo "║              ✅ PHASE 2 PIPELINE LAUNCHED                     ║"
            echo "╚════════════════════════════════════════════════════════════════╝"
            echo ""
            echo "📊 Phase 2 Results Directory: $PHASE2_DIR"
            echo "📈 Monitor Log: $PHASE2_DIR/monitor.log"
            echo "🌐 Dashboard: $PHASE2_DIR/dashboard.html"
            echo ""
            echo "✅ Pipeline is now running automatically"
            echo "   - Evaluation: Complete"
            echo "   - Ensemble Model: Created"
            echo "   - Backtesting: Running"
            echo "   - Paper Trading: Active on Binance Testnet"
            echo "   - Monitoring: Real-time tracking"
            echo ""
            
            exit 0
        else
            echo ""
            echo "❌ Phase 2 pipeline launch failed"
            exit 1
        fi
    fi
    
    echo "   ⏳ Waiting $CHECK_INTERVAL seconds before next check..."
    echo ""
    
    sleep "$CHECK_INTERVAL"
done
