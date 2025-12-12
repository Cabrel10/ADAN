#!/bin/bash

# Extract latest worker metrics from training log
# Shows: Portfolio Value, Drawdown, Win Rate for each worker

LOG_DIR="/mnt/new_data/t10_training/logs"
LATEST_LOG=$(ls -t "$LOG_DIR"/training_final_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ No log file found"
    exit 1
fi

echo ""
echo "📊 WORKER METRICS (Latest from log)"
echo "=================================================="
echo ""

# Get latest Portfolio Value for each worker
echo "💰 PORTFOLIO VALUES:"
echo "---"
grep -i "Portfolio Value:" "$LATEST_LOG" | tail -20 | grep -oE "\[Worker [0-9]+\].*Portfolio Value: [0-9.]+" | sort -u

echo ""
echo "📉 DRAWDOWN:"
echo "---"
grep -iE "Drawdown|DD:" "$LATEST_LOG" | tail -10

echo ""
echo "📈 WIN RATE:"
echo "---"
grep -iE "Win.?Rate|Winrate" "$LATEST_LOG" | tail -10

echo ""
echo "💹 POSITION STATS:"
echo "---"
echo "Positions opened:"
grep -c "POSITION OUVERTE\|POSITION OPENED" "$LATEST_LOG"
echo ""
echo "Positions closed:"
grep -c "POSITION CLOSED\|POSITION FERMÉE" "$LATEST_LOG"

echo ""
echo "=================================================="
echo ""

