#!/usr/bin/env bash
# Script de Monitoring Optuna V3
# Usage: ./monitor_optuna.sh

PID_FILE="optuna.pid"
LOG_FILE="logs/optuna_v3_production.log"

echo "==========================================  "
echo "  OPTUNA V3 MONITORING DASHBOARD"
echo "=========================================="

# 1. Vérifier si le processus tourne
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Optuna RUNNING (PID: $PID)"
        ps -p $PID -o %cpu,%mem,etime,cmd | tail -n 1
    else
        echo "❌ Optuna STOPPED (PID $PID not found)"
        exit 1
    fi
else
    echo "⚠️  No PID file found"
    exit 1
fi

echo ""
echo "=========================================="
echo "  RECENT ACTIVITY (Last 10 lines)"
echo "=========================================="
tail -n 10 "$LOG_FILE"

echo ""
echo "=========================================="
echo "  FORCE TRADE SUCCESSES"
echo "=========================================="
grep "✅ \[FORCE_TRADE\] Success" "$LOG_FILE" | wc -l | xargs echo "Total:"

echo ""
echo "=========================================="
echo "  TRIALS COMPLETED"
echo "=========================================="
grep -E "Trial.*finished" "$LOG_FILE" | tail -n 5

echo ""
echo "=========================================="
echo "  METRICS SUMMARY"
echo "=========================================="
grep -E "Metric|value=" "$LOG_FILE" | tail -n 10

echo ""
echo "=========================================="
echo "  COMMANDS DE MONITORING"
echo "==========================================  "
echo "# Suivre en temps réel:"
echo "  tail -f $LOG_FILE"
echo ""
echo "# Vérifier les trades:"
echo "  grep 'FORCE_TRADE.*Success' $LOG_FILE | wc -l"
echo ""
echo "# Arrêter Optuna:"
echo "  kill \$(cat $PID_FILE)"
echo ""
echo "=========================================="
