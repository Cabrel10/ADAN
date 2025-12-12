#!/bin/bash

# Continuous checkpoint monitoring
# Tracks checkpoint creation and worker progress

CHECKPOINT_DIR="/mnt/new_data/t10_training/checkpoints"
REPORT_FILE="CHECKPOINT_PROGRESS.txt"

echo "🔄 Starting continuous checkpoint monitoring..."
echo "Report file: $REPORT_FILE"
echo ""

while true; do
    clear
    
    echo "╔════════════════════════════════════════════════════════════════════════════════╗"
    echo "║                    📊 CHECKPOINT MONITORING - LIVE                            ║"
    echo "║                    $(date '+%Y-%m-%d %H:%M:%S')                                    ║"
    echo "╚════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Header
    printf "%-6s %-12s %-12s %-12s %-12s\n" "Worker" "Latest Step" "Count" "Size" "Last Time"
    echo "─────────────────────────────────────────────────────────────────────────────────"
    
    for WORKER in w1 w2 w3 w4; do
        WORKER_DIR="$CHECKPOINT_DIR/$WORKER"
        
        if [ ! -d "$WORKER_DIR" ]; then
            printf "%-6s %-12s %-12s %-12s %-12s\n" "$WORKER" "—" "—" "—" "—"
            continue
        fi
        
        # Get latest checkpoint
        LATEST=$(ls -t "$WORKER_DIR"/*.zip 2>/dev/null | head -1)
        
        if [ -z "$LATEST" ]; then
            printf "%-6s %-12s %-12s %-12s %-12s\n" "$WORKER" "—" "—" "—" "—"
            continue
        fi
        
        # Extract info
        LATEST_NAME=$(basename "$LATEST")
        LATEST_STEPS=$(echo "$LATEST_NAME" | sed 's/.*_\([0-9]*\)_steps.*/\1/')
        COUNT=$(ls -1 "$WORKER_DIR"/*.zip 2>/dev/null | wc -l)
        SIZE=$(du -sh "$WORKER_DIR" 2>/dev/null | cut -f1)
        LAST_TIME=$(stat -c %y "$LATEST" 2>/dev/null | cut -d' ' -f2 | cut -d'.' -f1)
        
        printf "%-6s %-12s %-12s %-12s %-12s\n" "$WORKER" "$LATEST_STEPS" "$COUNT" "$SIZE" "$LAST_TIME"
    done
    
    echo ""
    echo "📈 PROGRESS SUMMARY:"
    echo "─────────────────────────────────────────────────────────────────────────────────"
    
    # Calculate totals
    TOTAL_CHECKPOINTS=0
    TOTAL_SIZE=0
    
    for WORKER in w1 w2 w3 w4; do
        WORKER_DIR="$CHECKPOINT_DIR/$WORKER"
        COUNT=$(ls -1 "$WORKER_DIR"/*.zip 2>/dev/null | wc -l)
        TOTAL_CHECKPOINTS=$((TOTAL_CHECKPOINTS + COUNT))
    done
    
    TOTAL_SIZE=$(du -sh "$CHECKPOINT_DIR" 2>/dev/null | cut -f1)
    
    echo "Total Checkpoints: $TOTAL_CHECKPOINTS"
    echo "Total Size: $TOTAL_SIZE"
    echo ""
    
    # Process status
    if pgrep -f "train_parallel_agents.py" > /dev/null; then
        PROC_COUNT=$(pgrep -f "train_parallel_agents.py" | wc -l)
        echo "🔄 Training Status: ✅ RUNNING ($PROC_COUNT processes)"
    else
        echo "🔄 Training Status: ❌ STOPPED"
    fi
    
    echo ""
    echo "═════════════════════════════════════════════════════════════════════════════════"
    echo "⏱️  Refreshing in 60 seconds... (Press Ctrl+C to stop)"
    echo ""
    
    # Save to report
    {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')]"
        for WORKER in w1 w2 w3 w4; do
            WORKER_DIR="$CHECKPOINT_DIR/$WORKER"
            LATEST=$(ls -t "$WORKER_DIR"/*.zip 2>/dev/null | head -1)
            if [ ! -z "$LATEST" ]; then
                LATEST_STEPS=$(basename "$LATEST" | sed 's/.*_\([0-9]*\)_steps.*/\1/')
                echo "$WORKER: $LATEST_STEPS steps"
            fi
        done
        echo ""
    } >> "$REPORT_FILE"
    
    sleep 60
done

