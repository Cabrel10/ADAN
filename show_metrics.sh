#!/bin/bash

# Display current training metrics
# Shows portfolio value, win rate, sharpe ratio for each worker

LOG_DIR="/mnt/new_data/t10_training/logs"
LATEST_LOG=$(ls -t "$LOG_DIR"/training_final_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ No log file found"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║                    🚀 T10 TRAINING - CURRENT METRICS                          ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Show file info
echo "📝 Log File:"
echo "   Name: $(basename "$LATEST_LOG")"
echo "   Size: $(du -h "$LATEST_LOG" | cut -f1)"
echo ""

# Show latest METRICS_SYNC entries
echo "📊 LATEST WORKER METRICS (Last 10 entries):"
echo "─────────────────────────────────────────────────────────────────────────────────"
echo ""

grep -E "METRICS_SYNC.*Worker" "$LATEST_LOG" | tail -10 | while read line; do
    # Parse the line
    TIMESTAMP=$(echo "$line" | cut -d' ' -f1-2)
    WORKER=$(echo "$line" | grep -oE "Worker [0-9]+" | head -1)
    STEP=$(echo "$line" | grep -oE "Step [0-9]+" | head -1)
    SHARPE=$(echo "$line" | grep -oE "Sharpe=[0-9.]+" | cut -d= -f2)
    SORTINO=$(echo "$line" | grep -oE "Sortino=[0-9.]+" | cut -d= -f2)
    WINRATE=$(echo "$line" | grep -oE "WinRate=[0-9.]+%" | cut -d= -f2)
    TRADES=$(echo "$line" | grep -oE "Trades=[0-9]+" | cut -d= -f2)
    
    printf "%-12s %-10s | Sharpe: %-6s | Sortino: %-6s | Win Rate: %-8s | Trades: %-6s\n" \
        "$WORKER" "$STEP" "$SHARPE" "$SORTINO" "$WINRATE" "$TRADES"
done

echo ""
echo "💰 PORTFOLIO VALUES (Last 10 entries):"
echo "─────────────────────────────────────────────────────────────────────────────────"
echo ""

grep "Portfolio Value:" "$LATEST_LOG" | tail -10 | while read line; do
    TIMESTAMP=$(echo "$line" | cut -d' ' -f1-2)
    WORKER=$(echo "$line" | grep -oE "\[Worker [0-9]+\]" | tr -d '[]')
    VALUE=$(echo "$line" | grep -oE "Portfolio Value: [0-9.]+" | cut -d: -f2 | xargs)
    STEP=$(echo "$line" | grep -oE "Step: [0-9]+" | cut -d: -f2 | xargs)
    
    if [ ! -z "$WORKER" ] && [ ! -z "$VALUE" ]; then
        printf "%-12s Step: %-6s | Portfolio: \$%-8s\n" "$WORKER" "$STEP" "$VALUE"
    fi
done

echo ""
echo "📈 TRADING ACTIVITY:"
echo "─────────────────────────────────────────────────────────────────────────────────"
echo ""
echo "Positions opened: $(grep -c "POSITION OUVERTE\|POSITION OPENED" "$LATEST_LOG")"
echo "Positions closed: $(grep -c "POSITION CLOSED\|POSITION FERMÉE" "$LATEST_LOG")"
echo ""

# Show process status
echo "🔄 PROCESS STATUS:"
echo "─────────────────────────────────────────────────────────────────────────────────"
if pgrep -f "train_parallel_agents.py" > /dev/null; then
    PROC_COUNT=$(pgrep -f "train_parallel_agents.py" | wc -l)
    echo "✅ Training running ($PROC_COUNT processes)"
else
    echo "❌ Training stopped"
fi

echo ""
echo "═════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "💡 Commands:"
echo "   tail -f /mnt/new_data/t10_training/logs/training_final_*.log"
echo "   ./check_training_status.sh monitor"
echo ""

