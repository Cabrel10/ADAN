#!/bin/bash
# Analyse rapide des logs d'entraînement

LOG_FILE="/mnt/new_data/adan_logs/training_20251208_072851.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    exit 1
fi

echo "📊 ANALYSE RAPIDE DES LOGS"
echo "=========================="
echo ""
echo "📁 File: $(basename $LOG_FILE)"
echo "   Size: $(du -h $LOG_FILE | cut -f1)"
echo "   Lines: $(wc -l < $LOG_FILE)"
echo ""

echo "👥 WORKERS ACTIFS"
echo "================="
for worker in 0 1 2 3; do
    count=$(grep -c "\[Worker $worker\]" "$LOG_FILE" 2>/dev/null || echo 0)
    if [ $count -gt 0 ]; then
        echo "✅ Worker $worker: $count entries"
    else
        echo "❌ Worker $worker: 0 entries"
    fi
done
echo ""

echo "📈 DERNIÈRES DONNÉES (W0)"
echo "========================"
grep "\[Worker 0\]" "$LOG_FILE" | tail -5 | sed 's/^/  /'
echo ""

echo "💰 RÉSUMÉ PnL"
echo "============="
echo "W0 PnL entries:"
grep -o "PnL[^,]*" "$LOG_FILE" | grep "Worker 0" | tail -3 | sed 's/^/  /'
echo ""

echo "🎯 RÉSUMÉ REWARDS"
echo "================="
echo "W0 Reward entries:"
grep -o "Reward[^,]*" "$LOG_FILE" | grep "Worker 0" | tail -3 | sed 's/^/  /'
echo ""

echo "⚠️  ERREURS DÉTECTÉES"
echo "===================="
error_count=$(grep -i "error\|exception\|nan\|inf" "$LOG_FILE" 2>/dev/null | wc -l)
if [ $error_count -gt 0 ]; then
    echo "❌ Found $error_count error entries:"
    grep -i "error\|exception" "$LOG_FILE" 2>/dev/null | head -3 | sed 's/^/  /'
else
    echo "✅ No errors detected"
fi
echo ""

echo "📊 PROGRESSION"
echo "=============="
last_step=$(grep -o "Step[: ]*[0-9]*" "$LOG_FILE" | tail -1 | grep -o "[0-9]*")
if [ -n "$last_step" ]; then
    echo "✅ Latest step: $last_step"
    echo "   Progression: $(echo "scale=2; $last_step * 100 / 500000" | bc)%"
else
    echo "⚠️  Could not determine latest step"
fi
echo ""

echo "✅ ANALYSE TERMINÉE"
