#!/bin/bash

LOG_FILE="deploy/adan_bot/logs/endurance_test.log"
PID_FILE="deploy/adan_bot/logs/endurance_test.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "❌ PID file not found"
    exit 1
fi

PID=$(cat "$PID_FILE")

echo "📊 TEST D'ENDURANCE - STATUS"
echo "=============================="
echo ""

# Vérifier si le processus est actif
if ps -p $PID > /dev/null 2>&1; then
    echo "✅ Processus actif (PID $PID)"
else
    echo "❌ Processus arrêté"
    exit 1
fi

# Compter les cycles
cycles=$(grep -c "🔄 Cycle" "$LOG_FILE" 2>/dev/null || echo "0")
echo "🔄 Cycles complétés: $cycles"

# Vérifier les décisions ADAN
decisions=$(grep -c "ADAN:" "$LOG_FILE" 2>/dev/null || echo "0")
echo "🤖 Décisions ADAN: $decisions"

# Vérifier les erreurs
errors=$(grep -c "ERROR" "$LOG_FILE" 2>/dev/null || echo "0")
echo "⚠️  Erreurs: $errors"

# Vérifier les warnings
warnings=$(grep -c "WARNING" "$LOG_FILE" 2>/dev/null || echo "0")
echo "⚠️  Warnings: $warnings"

# Dernière activité
echo ""
echo "📋 Dernière activité:"
tail -5 "$LOG_FILE" | sed 's/^/   /'

echo ""
echo "=============================="
