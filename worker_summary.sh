#!/bin/bash

# Quick worker performance summary
# Shows latest metrics for each worker

LOG_DIR="/mnt/new_data/t10_training/logs"
LATEST_LOG=$(ls -t "$LOG_DIR"/training_final_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ No log file found"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║                    📊 WORKER PERFORMANCE SUMMARY                              ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Extract latest metrics for each worker
echo "📈 LATEST METRICS (from METRICS_SYNC):"
echo "─────────────────────────────────────────────────────────────────────────────────"
echo ""

grep -E "METRICS_SYNC.*Worker" "$LATEST_LOG" | tail -20 | while read line; do
    # Extract worker ID
    WORKER=$(echo "$line" | grep -oE "Worker [0-9]+" | head -1)
    
    # Extract step
    STEP=$(echo "$line" | grep -oE "Step [0-9]+" | head -1 | awk '{print $2}')
    
    # Extract metrics
    SHARPE=$(echo "$line" | grep -oE "Sharpe=[0-9.]+" | cut -d= -f2)
    WINRATE=$(echo "$line" | grep -oE "WinRate=[0-9.]+%" | cut -d= -f2)
    TRADES=$(echo "$line" | grep -oE "Trades=[0-9]+" | cut -d= -f2)
    
    if [ ! -z "$WORKER" ]; then
        printf "%-12s Step: %-6s | Sharpe: %-6s | Win Rate: %-8s | Trades: %-6s\n" \
            "$WORKER" "$STEP" "$SHARPE" "$WINRATE" "$TRADES"
    fi
done | sort -u

echo ""
echo "💰 PORTFOLIO VALUES (Latest):"
echo "─────────────────────────────────────────────────────────────────────────────────"
echo ""

grep "Portfolio Value:" "$LATEST_LOG" | tail -50 | grep -oE "\[Worker [0-9]+\].*Portfolio Value: [0-9.]+" | sort -u | while read line; do
    WORKER=$(echo "$line" | grep -oE "Worker [0-9]+")
    VALUE=$(echo "$line" | grep -oE "Portfolio Value: [0-9.]+" | cut -d: -f2 | xargs)
    printf "%-12s Portfolio: \$%-8s\n" "$WORKER" "$VALUE"
done

echo ""
echo "📊 SUMMARY STATS:"
echo "─────────────────────────────────────────────────────────────────────────────────"
echo ""
echo "Total positions opened: $(grep -c "POSITION OUVERTE\|POSITION OPENED" "$LATEST_LOG")"
echo "Total positions closed: $(grep -c "POSITION CLOSED\|POSITION FERMÉE" "$LATEST_LOG")"
echo "Log file size: $(du -h "$LATEST_LOG" | cut -f1)"
echo "Log file: $(basename "$LATEST_LOG")"
echo ""
echo "═════════════════════════════════════════════════════════════════════════════════"
echo ""

