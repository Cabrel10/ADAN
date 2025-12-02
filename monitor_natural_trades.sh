#!/usr/bin/env bash
# Monitoring Natural Trades - 15 minutes
# Exit if < 5 natural trades

echo "==================================================="
echo "  MONITORING NATURAL TRADES - 15 MINUTES"
echo "==================================================="
echo "Start time: $(date)"
echo "Target: 5+ natural trades (non-forced)"
echo ""

LOG_FILE="logs/optuna_v3_production.log"

# Baseline force trades
INITIAL_FORCE=$(grep -c "✅ \[FORCE_TRADE\] Success" "$LOG_FILE" 2>/dev/null || echo "0")
echo "Initial force trades: $INITIAL_FORCE"

# Function to count natural trades (daily_total > 0)
count_natural_trades() {
    grep "daily_total:" "$LOG_FILE" | tail -n 100 | grep -oE "daily_total: [1-9][0-9]*" | wc -l
}

INITIAL_NATURAL=$(count_natural_trades)
echo "Initial natural trade signals: $INITIAL_NATURAL"
echo ""
echo "Monitoring for 15 minutes..."
echo "Press Ctrl+C to stop monitoring"
echo ""

START_TIME=$(date +%s)
DURATION=900  # 15 minutes

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    
    if [ $ELAPSED -ge $DURATION ]; then
        echo ""
        echo "==================================================="
        echo "  15 MINUTES ELAPSED - FINAL CHECK"
        echo "==================================================="
        break
    fi
    
    # Check every 30 seconds
    sleep 30
    
    CURRENT_NATURAL=$(count_natural_trades)
    NEW_TRADES=$((CURRENT_NATURAL - INITIAL_NATURAL))
    REMAINING=$((DURATION - ELAPSED))
    
    echo "[$(date +%H:%M:%S)] Natural trades: $NEW_TRADES/5 | Remaining: ${REMAINING}s"
    
    if [ $NEW_TRADES -ge 5 ]; then
        echo ""
        echo "==================================================="
        echo "  ✅ SUCCESS - 5+ NATURAL TRADES DETECTED!"
        echo "==================================================="
        echo "Natural trades found: $NEW_TRADES"
        exit 0
    fi
done

# Final check
FINAL_NATURAL=$(count_natural_trades)
TOTAL_NEW=$((FINAL_NATURAL - INITIAL_NATURAL))

echo "Final natural trades: $TOTAL_NEW"
echo ""

if [ $TOTAL_NEW -ge 5 ]; then
    echo "✅ SUCCESS - Target reached ($TOTAL_NEW trades)"
    exit 0
else
    echo "❌ FAILURE - Only $TOTAL_NEW natural trades (need 5)"
    echo ""
    echo "RECOMMENDATION: Stop Optuna and investigate"
    echo "Command to stop: kill \$(cat optuna.pid)"
    exit 1
fi
